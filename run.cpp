#include<torch/torch.h>
#include <torch/script.h>
#include<regex>
#include<stdio.h>
#include <fstream>
#include<iostream>
#include <nlohmann/json.hpp>
#include <typeinfo>
#include <opencv2/opencv.hpp>
#include<cmath>
#include<algorithm>

using namespace torch::indexing;
using json = nlohmann::json;


struct Tokenizer{

  json data;
  json index_to_word;
  json word_to_index;
  int vocab_size;
  int start_id, end_id;

  Tokenizer(std::string vocab_path){
    std::ifstream f(vocab_path);
    data = json::parse(f);
    vocab_size = int(data["vocab_size"]);
    index_to_word = data["index_to_word"];
    word_to_index = data["word_to_index"];
    start_id = int(word_to_index["<S>"]);
    end_id = int(word_to_index["</S>"]);
 } 

  int getStart_id(){
    return start_id;
  }

  int getEnd_id(){
    return end_id;
  }


  std::string combine(std::vector<std::string> in){
    std::string str = in.front();
    in.erase(in.begin());
    std::smatch m;
    std::regex r ("</w>");
    for(std::string t : in){
      str = str + t;
    }
    str = std::regex_replace(str, r, " ");
    return str;
  }

  std::string  decode(torch::Tensor x){
    std::vector<std::string> out;
    for(int i = 1; i < (x.sizes()[1] - 1); i++){         //remove <S> and </S>
      int t = x.index({0, i}).item<int>();
      std::string s = index_to_word[std::to_string(t)];
      out.push_back(s);
    }
    return combine(out);
 }

};


torch::Tensor letterbox(cv::Mat in, int img_size){                           
  float r;
  cv::Mat im;
  torch::Tensor img;
  int t = std::max(in.cols, in.rows);
  r = (float)img_size/(float)t ;

  int new_unpad[2] = {(int)std::round(in.cols * r), (int)std::round(in.rows * r)};
  float dw = (float)(img_size - new_unpad[0])/(float)2;
  float dh = (float)(img_size - new_unpad[1])/(float)2;

  int top = (int)(std::round(dh - 0.1));
  int bottom = (int)(std::round(dh + 0.1));
  int left = (int)(std::round(dw - 0.1));
  int right = (int)(std::round(dw + 0.1));

  if (im.rows != new_unpad[0] && im.cols != new_unpad[1]){ 
    cv::resize(in, in, cv::Size(new_unpad[0], new_unpad[1]), cv::INTER_LINEAR);
  }

  cv::copyMakeBorder(in, im, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));	
  cv::imshow(" ", im);
  cv::waitKey(0);

  std::vector<int64> size = {im.rows, im.cols, im.channels()};
  img = torch::from_blob(im.data , size, torch::TensorOptions().dtype(torch::kByte));
  img = img.to(torch::kFloat32);  
  img = torch::div(img, 255);
  img = img.unsqueeze(0).permute({0, 3, 1, 2}).contiguous();
  return img;
}


struct TextPositionalEncoding : torch::nn::Module {
  
  torch::Tensor position, div_term, pe;
  TextPositionalEncoding(int d_model = 512, float dropout = 0.0, int max_len = 512){
      position = torch::arange(max_len).unsqueeze(1);
      div_term = torch::pow(10000 , (torch::arange(0, d_model, 2)/ d_model));
      pe = torch::zeros({1, max_len, d_model});
      pe.index_put_({0, Slice(), Slice(0, None, 2)}, torch::sin(position / div_term));
      pe.index_put_({0, Slice(), Slice(1, None, 2)}, torch::cos(position / div_term));
      pe = register_buffer("pe", pe);
  }

  torch::Tensor forward(torch::Tensor x){
    x = x + pe.index({Slice(), Slice(None, x.sizes()[1]), Slice()});
    return x;
  }
};

struct ImagePositionalEncoding : torch::nn::Module {
  
  torch::Tensor pe;
  ImagePositionalEncoding(int emb_dim = 512, float dropout= 0.0, int max_len = 512){
    pe = register_parameter("pe", torch::zeros({1, max_len, emb_dim}));
  }

  torch::Tensor forward(torch::Tensor x){
    x = x + pe.index({Slice(), Slice(None, x.sizes()[1]), Slice()});
    return x;
  }
};

struct ImageEmbedding : torch::nn::Module{
  int patch_size;
  torch::nn::Conv2d patch{nullptr};
  ImageEmbedding(int in_channels = 3, int emb_dim = 512, int patch_size = 64){
      this->patch_size = patch_size;
      patch = register_module("patch", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, emb_dim, patch_size).stride(patch_size).bias(false)));
  }

  torch::Tensor forward(torch::Tensor x){
    torch::Tensor patches = patch(x);
    int B, C, H, W;
    B = patches.sizes()[0];C = patches.sizes()[1];H = patches.sizes()[2];W = patches.sizes()[3];
    patches = patches.permute({0, 2, 3, 1});
    patches = patches.view({B, H * W, C});
    return patches;
  }
};

struct Multi_head_Attention : torch::nn::Module{
  int num_heads, n_embd;
  float dropout;
  bool is_causal, training;
  torch::nn::Linear q{nullptr}, k{nullptr}, v{nullptr}, c_proj{nullptr};
  torch::nn::Dropout dropout_layer{nullptr};

  Multi_head_Attention(int emb_dim = 512, int num_heads= 8, float dropout = 0.0, bool is_causal = false){
    q = register_module("q", torch::nn::Linear(torch::nn::LinearOptions(emb_dim, emb_dim).bias(false)));
    k = register_module("k", torch::nn::Linear(torch::nn::LinearOptions(emb_dim, emb_dim).bias(false)));
    v = register_module("v", torch::nn::Linear(torch::nn::LinearOptions(emb_dim, emb_dim).bias(false)));
    c_proj = register_module("c_proj", torch::nn::Linear(torch::nn::LinearOptions(emb_dim, emb_dim).bias(false)));
    this->num_heads = num_heads;
    this->is_causal = is_causal;
    this->dropout = dropout;
    n_embd = emb_dim;
    dropout_layer = register_module("dropout_layer", torch::nn::Dropout(dropout));
  }

  torch::Tensor forward(torch::Tensor x1, torch::Tensor x2, torch::Tensor x3){
    int B, T1, T2, C;
    B = x1.sizes()[0]; T1 = x1.sizes()[1]; T2 = x2.sizes()[1]; C = x1.sizes()[2];
    torch::Tensor q_value = q(x1);
    torch::Tensor k_value = k(x2);
    torch::Tensor v_value = v(x2);
    q_value = q_value.view({B, T1, num_heads, int(C/num_heads)}).transpose(1, 2);
    k_value = k_value.view({B, T2, num_heads, int(C/num_heads)}).transpose(1, 2);
    v_value = v_value.view({B, T2, num_heads, int(C/num_heads)}).transpose(1, 2);

    torch::Tensor y = at::scaled_dot_product_attention(q_value, k_value, v_value, {}, 0.0, is_causal);
    y = y.transpose(1, 2).contiguous().view({B, T1, C});

    y = c_proj(y);
    y = dropout_layer(y);
    return y;
  }
}; 

struct FeedForward : torch::nn::Module{
  torch::nn::Linear w1{nullptr}, w2{nullptr};
  torch::nn::Dropout dropout{nullptr};
  torch::nn::GELU act;

  FeedForward(){}
  FeedForward(int emb_dim = 512, float dropout= 0.0){
    w1 = register_module("w1", torch::nn::Linear(torch::nn::LinearOptions(emb_dim, emb_dim*4).bias(false)));
    w2 = register_module("w2", torch::nn::Linear(torch::nn::LinearOptions(emb_dim*4, emb_dim).bias(false)));
    this->dropout = register_module("dropout", torch::nn::Dropout(dropout));
  }

  torch::Tensor forward(torch::Tensor x){
    return dropout(w2(act(w1(x))));
  }
};

struct Encoder_block : torch::nn::Module{

  std::shared_ptr<Multi_head_Attention> self_attn ;
  std::shared_ptr<FeedForward> ff ;
  torch::nn::LayerNorm self_attn_ln{nullptr}, ff_ln{nullptr};

  Encoder_block(int emb_dim = 512, int num_head = 8, float dropout = 0.0){
    self_attn = std::make_shared<Multi_head_Attention>(emb_dim, num_head, dropout, false);
    ff = std::make_shared<FeedForward>(emb_dim, dropout);
    self_attn = register_module("self_attn",self_attn);
    ff = register_module("ff", ff);
    self_attn_ln = register_module("self_attn_ln", torch::nn::LayerNorm(torch::nn::LayerNormOptions({emb_dim})));
    ff_ln = register_module("ff_ln", torch::nn::LayerNorm(torch::nn::LayerNormOptions({emb_dim})));
  }

  torch::Tensor forward(torch::Tensor x){
    torch::Tensor z = self_attn_ln(x);
    x = x + self_attn->forward(z , z , z);
    z = ff_ln(x);
    x = x + ff->forward(z);
    return x;
  }

};


struct decoder_block : torch::nn::Module{
  std::shared_ptr<Multi_head_Attention> self_attn, cross_attn;
  std::shared_ptr<FeedForward> ff;
  torch::nn::LayerNorm self_attn_ln{nullptr}, cross_attn_ln{nullptr}, ff_ln{nullptr};
  
  decoder_block(int emb_dim = 512, int num_heads = 8, float dropout = 0.0){
    self_attn = std::make_shared<Multi_head_Attention>(emb_dim, num_heads, dropout, true);
    cross_attn = std::make_shared<Multi_head_Attention>(emb_dim, num_heads, dropout, false);
    ff = std::make_shared<FeedForward>(emb_dim, dropout);
    ff = register_module("ff", ff);
    self_attn = register_module("self_attn", self_attn);
    cross_attn = register_module("cross_attn", cross_attn);
    self_attn_ln = register_module("self_attn_ln", torch::nn::LayerNorm(torch::nn::LayerNormOptions({emb_dim})));
    cross_attn_ln = register_module("cross_attn_ln", torch::nn::LayerNorm(torch::nn::LayerNormOptions({emb_dim})));
    ff_ln = register_module("ff_ln", torch::nn::LayerNorm(torch::nn::LayerNormOptions({emb_dim})));
  }

  torch::Tensor forward(torch::Tensor x1, torch::Tensor x2){
    torch::Tensor z = self_attn_ln(x1);
    torch::Tensor x = x1 + self_attn->forward(z, z, z);
    z = cross_attn_ln(x);
    x = x + cross_attn->forward(z, x2, x2);
    z = ff_ln(x);
    x = x + ff->forward(z);
    return x;
  }

};

struct Encoder: torch::nn::Module{

  torch::nn::ModuleList dec;
  
  Encoder(int emb_dim = 512, int num_heads = 8, int num_layers = 6, float dropout = 0.0){
    for(int i = 0; i < num_layers; i++){
      dec->push_back(Encoder_block(emb_dim, num_heads, dropout));
    }
    dec = register_module("dec", torch::nn::ModuleList(dec));
  }

  torch::Tensor forward(torch::Tensor x){
    for(const auto& module : *dec){
      x = module->as<Encoder_block>()->forward(x);
    }
    return x;
  }
};

struct Decoder: torch::nn::Module{

  torch::nn::ModuleList dec;
  
  Decoder(int emb_dim = 512, int num_heads = 8, int num_layers = 6, float dropout = 0.0){
    for(int i = 0; i < num_layers; i++){
      dec->push_back(decoder_block(emb_dim, num_heads, dropout));
    }
    dec = register_module("dec", torch::nn::ModuleList(dec));
  }

  torch::Tensor forward(torch::Tensor x1, torch::Tensor x2){
    for(const auto& module : *dec){
      x2 = module->as<decoder_block>()->forward(x2, x1);
    }
    return x2;
  }
};

struct Model : torch::nn::Module{

  torch::nn::Embedding txt_embs{nullptr};
  std::shared_ptr<TextPositionalEncoding> txt_pos_enc;
  std::shared_ptr<ImageEmbedding> img_embs;
  std::shared_ptr<ImagePositionalEncoding> img_pos_emb;
  std::shared_ptr<Encoder> encoder;
  std::shared_ptr<Decoder> decoder;
  torch::nn::LayerNorm ln_encoder{nullptr}, ln_decoder{nullptr};
  torch::nn::Linear pred{nullptr};

  Model(int vocab_size, int in_channels = 3, int emb_dim = 512, int patch_size = 64, int num_heads = 8, int num_layers = 6, float dropout = 0.0){
    txt_embs = torch::nn::Embedding(vocab_size, emb_dim);
    pred = torch::nn::Linear(torch::nn::LinearOptions(emb_dim, vocab_size).bias(false));
    txt_embs->weight = pred->weight;           //weight tying
    txt_embs = register_module("txt_embs", txt_embs);
    txt_pos_enc = std::make_shared<TextPositionalEncoding>(emb_dim);
    img_embs = std::make_shared<ImageEmbedding>(in_channels, emb_dim, patch_size);
    img_pos_emb = std::make_shared<ImagePositionalEncoding>(emb_dim);
    encoder = std::make_shared<Encoder>(emb_dim, num_heads, num_layers,dropout);
    decoder = std::make_shared<Decoder>(emb_dim, num_heads, num_layers,dropout);
    txt_pos_enc = register_module("txt_pos_enc", txt_pos_enc);
    img_embs = register_module("img_embs", img_embs);
    img_pos_emb = register_module("img_pos_emb", img_pos_emb);
    encoder = register_module("encoder", encoder);
    decoder = register_module("decoder", decoder);
    ln_encoder = register_module("ln_enocder", torch::nn::LayerNorm(torch::nn::LayerNormOptions({emb_dim})));
    ln_decoder = register_module("ln_decoder", torch::nn::LayerNorm(torch::nn::LayerNormOptions({emb_dim})));
    pred = register_module("pred", pred);
  }

  torch::Tensor forward(torch::Tensor x1 , torch::Tensor x2){
    x1 = img_pos_emb->forward(img_embs->forward(x1));
    x2 = txt_pos_enc->forward(txt_embs(x2));
    x1 = encoder->forward(x1);
    x1 = ln_encoder->forward(x1);                                                                                                            
    torch::Tensor y = decoder->forward(x1, x2);
    y = ln_decoder->forward(y);
    torch::Tensor logits = pred->forward(y);
    return logits;
  }
};


torch::Tensor generate(Model model , Tokenizer tokenizer, torch::Tensor img, int max_len = 64){
  int64 temp[1] = {tokenizer.getStart_id()};
  torch::Tensor indx = torch::from_blob(temp, {1}, torch::TensorOptions().dtype(torch::kInt64));
  indx = indx.unsqueeze(0); 
  for(int i = 0; i < max_len; i ++){
    torch::Tensor logits = model.forward(img, indx);
    logits = logits.squeeze(0);
    logits = logits.index({-1}).unsqueeze(0);
    torch::Tensor probs = torch::softmax(logits, 1);
    torch::Tensor idx_next = torch::argmax(probs, 1);
    idx_next = idx_next.unsqueeze(0);
    indx = torch::cat({indx, idx_next}, 1);
    if(idx_next.item<int>() == tokenizer.getEnd_id()){ 
      return indx;
    }  
  }
  return indx;
}


int main(){

  //torch::manual_seed(420);                                                         

  Tokenizer tokenizer("../vocab.json");
  
  torch::serialize::InputArchive input_archive;
  input_archive.load_from("../saved_model/traced_best.pt");

  Model model(tokenizer.vocab_size, 3, 768, 30, 12, 6);
  model.load(input_archive);

  model.to(torch::kCPU);
  model.eval();
  
  // // for (const auto p : model.named_parameters()){
  // //   std::cout<<p.key()<<"-"<<p.value().sizes()<<std::endl;
  // // } 

  cv::Mat image = cv::imread("../Modern_Warfare.png", cv::IMREAD_COLOR);
  torch::Tensor img = letterbox(image, 480);

  torch::Tensor output = generate(model, tokenizer, img);
  std::string out = tokenizer.decode(output);
  std::cout<<out<<"\n";

  return 0;

}

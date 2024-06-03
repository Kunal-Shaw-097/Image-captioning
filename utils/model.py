from utils.modules import decoder_block, TextPositionalEncoding, ImageEmbedding, Encoder_block, ImagePositionalEncoding
import torch.nn as nn
from torch.nn import Embedding
import torch.nn.functional as F
import torch
    
    
class Decoder(nn.Module) :
    def __init__(self, emb_dim : int = 512, num_heads : int = 8, num_layers : int = 6, dropout : float = 0.0) :
        super().__init__()
        self.dec = nn.ModuleList([decoder_block(emb_dim, num_heads, dropout= dropout) for _ in range(num_layers)])
    
    def forward(self, x1 : torch.Tensor, x2 : torch.Tensor) :
        for layer in self.dec :
            x2 = layer(x2 , x1)
        return x2


class Encoder(nn.Module) :
    def __init__(self, emb_dim : int = 512, num_heads : int = 8, num_layers : int = 6, dropout : float = 0.0) :
        super().__init__()
        self.dec = nn.ModuleList([Encoder_block(emb_dim, num_heads, dropout= dropout) for _ in range(num_layers)])

    def forward(self, x : torch.Tensor) :
        for layer in self.dec :
            x = layer(x)
        return x

class Model(nn.Module) :
    def __init__(self, vocab_size : int, in_channels : int = 3, emb_dim : int = 512, patch_size : int = 64, num_heads = 8,
                num_layers = 6, dropout : float = 0.0) :
        super().__init__()
        self.txt_embs = Embedding(vocab_size, embedding_dim= emb_dim)                                # token embedding layer
        self.txt_pos_enc =  TextPositionalEncoding(emb_dim)  
        self.img_embs =  ImageEmbedding(in_channels, emb_dim, patch_size= patch_size)
        self.img_pos_emb = ImagePositionalEncoding(emb_dim)
        self.encoder = Encoder(emb_dim, num_heads, num_layers,dropout= dropout)  
        self.decoder = Decoder(emb_dim, num_heads, num_layers, dropout = dropout)
        self.ln_enocder = nn.LayerNorm(emb_dim)                                       
        self.ln_decoder = nn.LayerNorm(emb_dim)
        self.pred = nn.Linear(emb_dim, vocab_size, bias= False)                                  # final layer
        self.txt_embs.weight = self.pred.weight                                                  # weights tying

    def forward(self, x1 : torch.Tensor, x2 : torch.Tensor, target : torch.Tensor = None) :
        x1 = self.img_pos_emb(self.img_embs(x1))
        x2 = self.txt_pos_enc(self.txt_embs(x2))
        x1 = self.encoder(x1)
        x1 = self.ln_enocder(x1)                                                                                                            
        y = self.decoder(x1, x2)
        y = self.ln_decoder(y)
        logits = self.pred(y)
        loss = None
        if target is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)) , target.view(-1), ignore_index= 2)      #cross-entropy between (B*T, C) and (B*T)
        return logits, loss              # just remove the loss with tracing the model with jit     

    @torch.no_grad()
    def generate(self, x : torch.Tensor, tokenizer, device : str = 'cuda', greedy : bool = True, top_k : int = None, maxlen : int = 100) :
        indx = torch.tensor([tokenizer.start_id]).to(device)
        indx = indx.unsqueeze(0)    # indx size is (1,1)

        for _ in range(maxlen):
            logits, loss = self.forward(x, indx)     # forward pass  
            logits = logits[:, -1, :]     
            if top_k is not None:                    # optionally crop the logits to only the top k options
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)     #softmax along last dim
            if greedy :
                probs = probs.unsqueeze(0)
                idx_next = torch.argmax(probs, dim = -1)[-1]     # greedy search
                idx_next = idx_next.unsqueeze(0)
            else : 
                idx_next = torch.multinomial(probs, num_samples=1)   # sample from the distribution
            if idx_next.item() == tokenizer.end_id :
                return indx
            indx = torch.cat((indx, idx_next), dim=1)           # append sampled index to the running sequence and continue
        return indx

    def process_output(self ,x : torch.Tensor) :
        x = torch.argmax(F.softmax(x, dim= -1), dim = -1)
        return x

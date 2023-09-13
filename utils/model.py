from utils.modules import decoder_block, PositionalEncoding, encoder_backbone_up, encoder_backbone_down, encoder_head
import torch.nn as nn
from torch.nn import Embedding
import torch.nn.functional as F
import torch

class Encoder(nn.Module) :
    def __init__(self, c1 : int = 3, c2 : int = 512, ratio : list = [1, 2, 5], depth_ratio : float = 0.33) :
        super().__init__()
        assert c2 % 32 == 0 , "embedding dim must be a multiple of 32"
        self.block1 = encoder_backbone_down(c1, c2, depth_ratio)
        self.block2 = encoder_backbone_up(c2, depth_ratio)
        self.block3 = encoder_head(c2, ratio)

    def forward(self, x : torch.Tensor) :
        x1, x2, x3 = self.block1(x)
        x1, x2, x3 = self.block2(x1, x2, x3)
        y = self.block3(x1, x2, x3)
        return y
    
    
class Decoder(nn.Module) :
    def __init__(self, vocab_size : int, emb_dim : int = 512, num_heads : int = 8, num_layers : int = 6, dropout : float = 0.0) :
        
        super().__init__()
        self.dec = nn.ModuleList([decoder_block(emb_dim, num_heads, dropout= dropout)] * num_layers)
    
    def forward(self, x1 : torch.Tensor, x2 : torch.Tensor) :

        for layer in self.dec :
            x2 = layer(x2 , x1)

        return x2


class Model(nn.Module) :
    def __init__(self, vocab_size : int, c1 : int = 3, emb_dim : int = 512, num_heads = 8, num_layers = 6, dropout : float = 0.0 , channel_ratio : list = 
                 [1, 2, 5], depth_ratio : float = 0.33) :
        super().__init__()
        self.embs = Embedding(vocab_size, embedding_dim= emb_dim)                                # token embedding layer
        self.pos_enc =  PositionalEncoding(emb_dim)                                              
        self.encoder = Encoder(c1, emb_dim, channel_ratio, depth_ratio)
        self.decoder = Decoder(vocab_size, emb_dim, num_heads, num_layers, dropout = dropout)
        self.flatten = nn.Flatten(start_dim= 2, end_dim= -1)                                     #to flatten feature maps to one dimension
        self.pred = nn.Linear(emb_dim, vocab_size)                                               # final layer
        #self.embs.weight = self.pred.weight                                                      # weights sharing

    def forward(self, x1 : torch.Tensor, x2 : torch.Tensor) :
        x1 = self.encoder(x1)
        x1 = self.flatten(x1)
        x1 = x1.transpose(1,-1)                                                                 #switching the channels to the last postion
        x2 = self.pos_enc(self.embs(x2))                                                        # adding postional encoding on embeddings
        y = self.decoder(x1, x2)
        logits = self.pred(y)
        return logits
    
    def loss(self, x : torch.Tensor, y : torch.Tensor) :
        '''x = x.to(torch.float32)
        y = y.to(torch.int64)'''
        B, T, C = x.shape
        loss = torch.nn.functional.cross_entropy(x.view(B, C, T) , y)                                                   #cross-entropy loss requires B,C,T format
        return loss

    @torch.no_grad()
    def generate(self, x : torch.Tensor, tokenizer, greedy : bool = True, top_k : int = None, maxlen : int = 100) :
        
        indx = torch.tensor([tokenizer.start_id])
        indx = torch.unsqueeze(0)    # indx size is (1,1)
        
        for _ in range(maxlen):
            logits = self.forward(indx, x)     # forward pass          
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')    
            probs = F.softmax(logits, dim=-1)[0]  #s oftmax along last dim
            if greedy :
                idx_next = torch.argmax(probs, dim = -1)     # greedy search
            else : 
                idx_next = torch.multinomial(probs, num_samples=1)   # sample from the distribution
            if idx_next.item() == tokenizer.end_id :
                return idx
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx

    def process_output(self ,x : torch.Tensor) :
        x = torch.argmax(F.softmax(x, dim= -1), dim = -1)
        return x

        
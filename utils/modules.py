import torch
import torch.nn as nn
import torch.nn.functional as F



class TextPositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 512, dropout: float = 0.0, max_len: int = 256):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = 10000 ** (torch.arange(0, d_model, 2) / d_model)
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position / div_term)
        pe[0, :, 1::2] = torch.cos(position / div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor):
        # x = (B , T, emb_dim)
        x = x + self.pe[:, : x.size(1), :]
        return x


class ImagePositionalEncoding(nn.Module) :
    def __init__(self, emb_dim : int = 512,  dropout : float = 0.0, max_len : int = 512) :
        super().__init__() 
        self.pe = nn.Parameter(torch.zeros(1, max_len, emb_dim), requires_grad=True)
    
    def forward(self, x : torch.Tensor) :
        x = x + self.pe[: ,:x.shape[1], :]
        return x

class ImageEmbedding(nn.Module):
    def __init__(self, in_channels : int = 3, emb_dim : int = 512, patch_size : int = 64):
        super().__init__()
        self.patch_size = patch_size
        self.patch = nn.Conv2d(in_channels, emb_dim, patch_size, stride=patch_size, bias = False)
        
    def forward(self, x):
        # Input x is an image tensor of shape (batch_size, in_channels, image_size, image_size)
        patches = self.patch(x)  # (batch_size, in_channels, Hx, Wx)
        B, C, H, W = patches.size()
        patches = patches.permute(0, 2, 3, 1).contiguous()  #(B, emb_dim, H, W) to (B, H, W, emb_dim)
        patches = patches.view(B, H * W, C)
        return patches         

class Multi_head_Attention(nn.Module) :
    def __init__(self, emb_dim : int = 512, num_heads : int = 8, dropout : float = 0.0, is_causal : bool = True) :
        super().__init__()
        assert emb_dim % num_heads == 0, "The emb_dim must be divisible by the num_heads"
        # query, key , value value for x1, x2, x3 for all heads
        self.q = nn.Linear(emb_dim, emb_dim, bias = False)
        self.k= nn.Linear(emb_dim, emb_dim, bias = False)
        self.v= nn.Linear(emb_dim, emb_dim, bias = False)
        self.c_proj = nn.Linear(emb_dim, emb_dim, bias= False)
        # regularization
        self.num_head = num_heads
        self.n_embd = emb_dim
        self.dropout= dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.is_causal = is_causal
    
    def forward(self, x1 : torch.Tensor, x2 : torch.Tensor, x3 : torch.Tensor):
        B, T1, C = x1.size() # batch size, sequence length, embedding dimensionality (n_embd)
        B, T2, C = x2.size() # batch size, encoded img length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.q(x1)
        k = self.k(x2)
        v = self.v(x3)
        q = q.view(B, T1, self.num_head, C // self.num_head).transpose(1, 2) # (B, nh, T1, hs)
        k = k.view(B, T2, self.num_head, C // self.num_head).transpose(1, 2) # (B, nh, T2, hs)
        v = v.view(B, T2, self.num_head, C // self.num_head).transpose(1, 2) # (B, nh, T2, hs)
        if self.training :
            dropout = self.dropout
        else :
            dropout = 0.0
        # cross-attention; cross-attend: (B, nh, T1, hs) x (B, nh, hs, T2) -> (B, nh, T1, T2) x (B, nh, T2, hs) -> (B, nh, T1, hs)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p= dropout ,is_causal= self.is_causal) 
        y = y.transpose(1, 2).contiguous().view(B, T1, C) # re-assemble all head outputs side by side      

        # output projection
        y = self.c_proj(y)
        y = self.dropout_layer(y)
        return y

'''class FeedForward(nn.Module):
    def __init__(self, emb_dim: int = 512, dropout : float = 0.0):
        super().__init__()
        hidden_dim = emb_dim * 4
        self.w1 = nn.Linear(emb_dim, hidden_dim, bias = False)
        self.w2 = nn.Linear(emb_dim, hidden_dim, bias = False)
        self.w3 = nn.Linear(hidden_dim, emb_dim, bias = False)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        # x = B, T, emb_dim
        return self.dropout(self.w3(self.act(self.w1(x)) * self.w2(x)))'''
    
class FeedForward(nn.Module):
    def __init__(self, emb_dim: int = 512, dropout : float = 0.0):
        super().__init__()
        hidden_dim = emb_dim * 4
        self.w1 = nn.Linear(emb_dim, hidden_dim, bias = False)
        self.act = nn.GELU()
        self.w2 = nn.Linear(hidden_dim, emb_dim, bias = False)
        self.dropout= nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        # x = B, T, emb_dim
        return self.dropout(self.w2(self.act(self.w1(x))))  
 
class Encoder_block(nn.Module) :
    def __init__(self, emb_dim : int = 512, num_head : int = 8, dropout : float = 0.0):
        super().__init__()
        self.self_attn = Multi_head_Attention(emb_dim, num_head, dropout, is_causal = False)
        self.self_attn_ln = nn.LayerNorm(emb_dim)
        self.ff = FeedForward(emb_dim, dropout)
        self.ff_ln = nn.LayerNorm(emb_dim)
    
    def forward(self, x : torch.Tensor) :
        z = self.self_attn_ln(x)
        x = x + self.self_attn(z, z ,z)
        z = self.ff_ln(x)
        x = x + self.ff(z)
        return x


class decoder_block(nn.Module):
    def __init__(self, emb_dim: int = 512, num_heads: int = 8, dropout : float = 0.0):
        super().__init__()
        self.self_attn = Multi_head_Attention(emb_dim, num_heads, dropout= dropout, is_causal= True)                        # self-attention
        self.cross_attn = Multi_head_Attention(emb_dim, num_heads, dropout= dropout, is_causal = False)                     # cross-attention
        self.self_attn_ln = nn.LayerNorm(emb_dim)
        self.cross_attn_ln = nn.LayerNorm(emb_dim)
        self.ff_ln = nn.LayerNorm(emb_dim)
        self.ff = FeedForward(emb_dim, dropout= dropout)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        z = self.self_attn_ln(x1)
        x = x1 + self.self_attn(z, z, z)
        z = self.cross_attn_ln(x)
        x = x + self.cross_attn(z, x2, x2)
        z = self.ff_ln(x)
        x = x + self.ff(z)
        return x





'''def autopad(k, p=None):  # kernel, padding
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(
        self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5
    ):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)

    def __init__(
        self, c1: int, c2: int, k: int = 1, s: int = 1,
        p=None, g: int = 1, d: int = 1, act: bool = True):
        
        super().__init__()
        self.conv = nn.Conv2d(
            c1, c2, k, s, padding= autopad(k, p), groups=g, dilation=d, bias=False
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor):
        return self.act(self.bn(self.conv(x)))


class C3(nn.Module):
    """Bottleneck with 3 convolutions."""

    def __init__(
        self, c1: int, c2: int, n: int = 1, shortcut=True, g: int = 1, e: float = 0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  
        self.m = nn.Sequential(
            *(
                Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0)
                for _ in range(n)
            )
        )

    def forward(self, x: torch.Tensor):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer by Glenn Jocher."""

    def __init__(self, c1: int, c2: int, k: int = 5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class encoder_backbone_down(nn.Module):
    def __init__(self, c1: int = 3, c2: int = 512, depth_ratio= 0.33):
        """This backbone is used in yolov5, depth ratio and c2 makes the difference between yolov5 small, medium and large models"""
        super().__init__()
        self.conv1 = Conv(c1, c2 // 16, k=7, s=2, p=3)
        self.conv2 = Conv(c2 // 16, c2 // 8, k=3, s=2)
        self.conv3 = Conv(c2 // 8, c2 // 4, k=3, s=2)
        self.conv4 = Conv(c2 // 4, c2 // 2, k=3, s=2)
        self.conv5 = Conv(c2 // 2, c2, k=3, s=2)

        self.c3_1 = C3(c2 // 8, c2 // 8, n=round(3 * depth_ratio))
        self.c3_2 = C3(c2 // 4, c2 // 4, n=round(6 * depth_ratio))
        self.c3_3 = C3(c2 // 2, c2 // 2, n=round(9 * depth_ratio))
        self.c3_4 = C3(c2, c2, n=round(3 * depth_ratio))

        self.sppf = SPPF(c2, c2, k=5)

    def forward(self, x: torch.Tensor):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.c3_1(y)

        y = self.conv3(y)
        out1 = self.c3_2(y)    # (B, c2/4, h/8, w/8)

        y = self.conv4(y)
        out2 = self.c3_3(y)    # (B, c2/2, h/16, w/16)

        y = self.conv5(y)
        y = self.c3_4(y)
        out3 = self.sppf(y)    # (B, c2, h/32, h/32)
        return out1, out2, out3


class encoder_backbone_up(nn.Module):
    def __init__(self, c: int = 512, depth_ratio: float = 0.33):
        "This is also the part of yolov5 backbone"
        super().__init__()
        self.conv1 = Conv(c, c // 2, k=1, s=1)
        self.conv2 = Conv(c // 2, c // 4, k=1, s=1)
        self.conv3 = Conv(c // 4, c // 4, k=3, s=2)
        self.conv4 = Conv(c // 2, c // 2, k=3, s=2)

        self.up = nn.Upsample(scale_factor=2, mode="nearest")

        self.c3_1 = C3(c, c // 2, n=round(3 * depth_ratio), shortcut=False)
        self.c3_2 = C3(c // 2, c // 4, n=round(3 * depth_ratio), shortcut=False)
        self.c3_3 = C3(c // 2, c // 2, n=round(3 * depth_ratio), shortcut=False)
        self.c3_4 = C3(c, c, n=round(3 * depth_ratio), shortcut=False)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor):
        res1 = self.conv1(x3)
        y = self.up(res1)
        y = torch.cat([y, x2], dim=1)
        y = self.c3_1(y)
        res2 = self.conv2(y)
        y = self.up(res2)
        y = torch.cat([y, x1], dim=1)
        out1 = self.c3_2(y)          # (B, c2/4, h/8, w/8)
        y = self.conv3(out1)
        y = torch.cat([y, res2], dim=1)
        out2 = self.c3_3(y)          #  (B, c2/2, h/16, w/ 16)
        y = self.conv4(out2)
        y = torch.cat([y, res1], dim=1)
        out3 = self.c3_4(y)          #  (B, c2, h/32, h/32)
        return out1, out2, out3


class encoder_head(nn.Module):
    def __init__(self, c: int = 512, ratio: list = [1, 2, 5]):
        """A custom head which takes the multiple outputs(3) of the backbone(like a FPN) and gives the ratio as weights for the different backbone outputs and 
        concatenates them"""
        super().__init__()
        assert isinstance(ratio, list), "ratio must be a list"
        assert (
            c % sum(ratio) == 0 and len(ratio) == 3            
        ), "the ratios should be a list of length 3 and sum should be a multiple of c2/emb_dim"       

        self.ratio = [(c // sum(ratio)) * k for k in ratio]

        self.conv1 = nn.Sequential(
            nn.Conv2d(c // 4, c//2, kernel_size=3, stride=2, padding= 1, bias= False),
            nn.Conv2d(c//2, self.ratio[0], kernel_size=3, stride=2, padding= 1, bias= False),
        )
        self.conv2 = nn.Conv2d(c // 2, self.ratio[1], kernel_size=3, stride=2, padding= 1, bias= False)
        self.conv3 = nn.Conv2d(c, self.ratio[2], kernel_size=1, stride= 1, bias= False)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.SiLU()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor):
        y1 = self.conv1(x1)
        y2 = self.conv2(x2)
        y3 = self.conv3(x3)
        return self.act(self.bn(torch.cat([y1, y2, y3], dim=1)))
    

class Yolov5_Encoder(nn.Module) :
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

class CSPDenseNet_encoder(nn.Module) :
    def __init__(self, c1 , c2) :
        super().__init__()  
        growth_rate = c2//16
        in_channels = c2//8
        num_layers = [6, 12, 24]

        self.conv1 = nn.Conv2d(c1 , in_channels, kernel_size= 7, stride= 2, padding= 3, bias= False)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.blocks = nn.ModuleList()
        for i, _ in enumerate(num_layers):
            self.blocks.append(CSPDenseblock(in_channels, growth_rate, _)) 
            in_channels = in_channels + (growth_rate * _)
            if i < len(num_layers) - 1 :
                self.blocks.append(Transitionlayer(in_channels, in_channels//2))
                in_channels = in_channels//2
        self.final_layer = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, kernel_size= 1, stride= 1, bias= False),
            nn.AdaptiveAvgPool2d((5,5))
        ) 
        
    def forward(self, x : torch.Tensor) :
        x = self.conv1(x)
        x = self.maxpool(x)
        for block in self.blocks :
            x = block(x)   
        return self.final_layer(x)
    
class Denselayer(nn.Module) :
    def __init__(self, c1 , c2) :
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, kernel_size= 3, padding= 1, bias= False)
        self.bn = nn.BatchNorm2d(c1)
        self.act = nn.SiLU()
    
    def forward(self, x : torch.Tensor) :
        return self.conv(self.act(self.bn(x)))
    
class Transitionlayer(nn.Module) :
    def __init__(self, c1, c2) :
        super().__init__()
        self.bottleneck = nn.Conv2d(c1, c2, kernel_sizclass Patch_image(nn.Module) :
    def __init__(self, c1 : int = 3, c2 : int = 512, ) -> None:
        super().__init__()
class Denseblock(nn.Module) :            
    def __init__(self, c1, growth_rate, num_layers):
        super().__init__()
        self.layers= nn.ModuleList()
        c_in = c1
        for _ in range(num_layers) :
            self.layers.append(Denselayer(c_in, growth_rate))
            c_in += growth_rate 

    def forward(self, x : torch.Tensor) :
        for layer in self.layers :
            y = layer(x)
            x = torch.cat([x, y], dim = 1)
        return x

class CSPDenseblock(nn.Module) :   #Cross Stage Partial network implementation of Denseblock
    def __init__(self, c1, growth_rate, num_layers) :
        super().__init__()
        self.c_in  = c1
        self.Dense = Denseblock(self.c_in//2, growth_rate, num_layers)
    
    def forward(self, x : torch.Tensor) :
        x1, x2 = x.split(self.c_in//2, dim= 1)
        x2 = self.Dense(x2)
        x = torch.cat([x1, x2], dim = 1)
        return x
        '''

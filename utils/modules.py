import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# NLP

class Faster_Self_Attention(nn.Module) :
    def __init__(self, emb_dim : int = 512, num_heads : int = 8, dropout : float = 0.0):
        super().__init__()
        assert emb_dim % num_heads == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(emb_dim, 3 * emb_dim, bias= False)
        # output projection
        self.c_proj = nn.Linear(emb_dim, emb_dim, bias= False)
        # regularization
        self.num_head = num_heads
        self.n_embd = emb_dim
        self.dropout= dropout
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.num_head, C // self.num_head).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.num_head, C // self.num_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.num_head, C // self.num_head).transpose(1, 2) # (B, nh, sT, hs)

        if self.training :
            is_casual = True
            dropout = self.dropout
        else :
            is_casual = False
            dropout = 0.0

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p= dropout, is_causal= is_casual) 
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        

        # output projection
        y = self.dropout_layer(self.c_proj(y))
        return y

class Faster_Cross_Attention(nn.Module) :
    def __init__(self, emb_dim : int = 512, num_heads : int = 8, dropout : float = 0.0) :
        super().__init__()
        # query value for x1 for all heads
        self.q = nn.Linear(emb_dim, emb_dim, bias = False)
        # key, value from x2 for all heads
        self.kv = nn.Linear(emb_dim, 2*emb_dim, bias = False)
        self.c_proj = nn.Linear(emb_dim, emb_dim, bias= False)
        # regularization
        self.num_head = num_heads
        self.n_embd = emb_dim
        self.dropout= dropout
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, x1 : torch.Tensor, x2 : torch.Tensor):
        B, T1, C = x1.size() # batch size, sequence length, embedding dimensionality (n_embd)
        B, T2, C = x2.size() # batch size, image_size/32(enocder output length) , embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q= self.q(x1)
        k, v = self.kv(x2).split(self.n_embd, dim=2)
        q = q.view(B, T1, self.num_head, C // self.num_head).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T2, self.num_head, C // self.num_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T2, self.num_head, C // self.num_head).transpose(1, 2) # (B, nh, T, hs)
        if self.training :
            dropout = self.dropout
        else :
            dropout = 0.0
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p= dropout ,is_causal= False) 
        y = y.transpose(1, 2).contiguous().view(B, T1, C) # re-assemble all head outputs side by side
        

        # output projection
        y = self.dropout_layer(self.c_proj(y))
        return y


class Attention_head(nn.Module):
    def __init__(self, emb_dim: int = 512, num_head: int = 8):
        super().__init__()
        assert (
            emb_dim % num_head == 0
        ), "Number of head should be a factor of Embedding dimension"

        self.head_dim = int(emb_dim / num_head)

        self.q = nn.Linear(emb_dim, self.head_dim, bias=False)
        self.k = nn.Linear(emb_dim, self.head_dim, bias=False)
        self.v = nn.Linear(emb_dim, self.head_dim, bias=False)


class Self_Attention_head(Attention_head):
    def __init__(self, emb_dim: int = 512, num_head: int = 8):
        super().__init__(emb_dim, num_head)

    def forward(self, x1: torch.Tensor, mask: torch.Tensor):
        # x.size = (B, T, emb_size)
        q, k, v = (
            self.q(x1),
            self.k(x1),
            self.v(x1),
        )  # all (B, T, head_dim)

        qk = q @ k.transpose(-1, -2)  # (B , T , T)
        scaled_qk = qk / self.head_dim**0.5  # scaling
        
        if self.training :
            scaled_qk = scaled_qk.masked_fill(
                mask == 0, float("-inf")
            )  # applying the above matrix as mask over batches
        softmax_qk = F.softmax(scaled_qk, dim=-1)  # softmax

        qkv = softmax_qk @ v  #  (B, T, head_dim)

        return qkv


class Cross_Attention_head(Attention_head):
    def __init__(self, emb_dim: int = 512, num_head: int = 8):
        super().__init__(emb_dim, num_head)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        q, k, v = (
            self.q(x1),
            self.k(x2),
            self.v(x2),
        )  # q = (B, T, head_dim), k=v=(B, T', head_dim)

        qk = q @ k.transpose(-1, -2)  # (B , T , T')
        scaled_qk = qk / self.head_dim**0.5  # scaling

        softmax_qk = F.softmax(scaled_qk, dim=-1)  # softmax

        qkv = softmax_qk @ v  #  (B, T, head_dim)

        return qkv


class FeedForward(nn.Module):
    def __init__(self, emb_dim: int = 512):
        super().__init__()
        self.ext = nn.Linear(emb_dim, emb_dim * 4)
        self.proj = nn.Linear(emb_dim * 4, emb_dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor):
        # x = B, T, emb_dim
        x1 = self.act(self.ext(x))  # (B, T, 4 * emb_dim)
        x2 = self.proj(x1)  # back to (B, T, emb_dim)
        return x2


class PositionalEncoding(nn.Module):
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


class Multi_head_Attention(nn.Module):
    def __init__(self, emb_dim: int = 512, num_heads: int = 8, self_attn: bool = True):
        super().__init__()
        if self_attn:
            self.heads = nn.ModuleList(
                [Self_Attention_head(emb_dim, num_heads) for i in range(num_heads)]
            )
        else:
            self.heads = nn.ModuleList(
                [Cross_Attention_head(emb_dim, num_heads) for i in range(num_heads)]
            )


class Self_Multi_head_Attention(Multi_head_Attention):
    def __init__(self, emb_dim: int = 512, num_heads: int = 8):
        super().__init__(emb_dim, num_heads, True)

    def forward(self, x1: torch.Tensor, mask: torch.Tensor = None):
        # x = (B, T, emb_dim)
        out = []

        for head in self.heads:
            out.append(head(x1, mask))  # Each item is (B, T, emb_dim/num_heads)

        out = torch.cat(out, dim=-1)  # Concat all to make (B, T, emb_dim)
        return out


class Cross_Multi_head_Attention(Multi_head_Attention):
    def __init__(self, emb_dim: int = 512, num_heads: int = 8):
        super().__init__(emb_dim, num_heads, False)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        # x = (B, T, emb_dim)
        out = []

        for head in self.heads:
            out.append(head(x1, x2))  # Each item is (B, T, emb_dim/num_heads)

        out = torch.cat(out, dim=-1)  # Concat all to make (B, T, emb_dim)
        return out


class decoder_block(nn.Module):
    def __init__(self, emb_dim: int = 512, num_heads: int = 8):
        super().__init__()
        self.mha = Faster_Self_Attention(emb_dim, num_heads, dropout= 0.1)  # self-attention
        #self.mha = Self_Multi_head_Attention(emb_dim, num_heads)
        #self.cha = Cross_Multi_head_Attention(emb_dim, num_heads)  # cross-attention
        self.cha = Faster_Cross_Attention(emb_dim, num_heads, dropout= 0.1)
        self.layer_norm = nn.LayerNorm(emb_dim)
        self.ff = FeedForward(emb_dim)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, mask: torch.Tensor):
        out1 = self.mha(x1)
        #out1 = self.mha(x1, mask)
        out1 = self.layer_norm(x1 + out1)
    
        out2 = self.cha(out1, x2)
        out2 = self.layer_norm(out1 + out2)
    
        out3 = self.ff(out2)
        out3 = self.layer_norm(out2 + out3)
        return out3


# CV


def autopad(k, p=None):  # kernel, padding
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
        """'forward()' applies the YOLOv5 FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 1,
        s: int = 1,
        p=None,
        g: int = 1,
        d: int = 1,
        act: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            c1, c2, k, s, padding=autopad(k, p), groups=g, dilation=d, bias=False
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor):
        return self.act(self.bn(self.conv(x)))


class C3(nn.Module):
    """Bottleneck with 3 convolutions."""

    def __init__(
        self, c1: int, c2: int, n: int = 1, shortcut=True, g: int = 1, e: float = 0.5
    ):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(
            *(
                Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0)
                for _ in range(n)
            )
        )
        ##review above

    def forward(self, x: torch.Tensor):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1: int, c2: int, k: int = 5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor):
        """Forward pass through Ghost Convolution block."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class encoder_backbone_down(nn.Module):
    def __init__(self, c1: int = 3, c2: int = 512, depth_ratio=0.33):
        super().__init__()
        self.conv1 = Conv(c1, c2 // 16, k=6, s=2, p=2)
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
        out1 = self.c3_2(y)

        y = self.conv4(y)
        out2 = self.c3_3(y)

        y = self.conv5(y)
        y = self.c3_4(y)
        out3 = self.sppf(y)
        return out1, out2, out3


class encoder_backbone_up(nn.Module):
    def __init__(self, c: int = 512, depth_ratio: float = 0.33):
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
        out1 = self.c3_2(y)
        y = self.conv3(out1)
        y = torch.cat([y, res2], dim=1)
        out2 = self.c3_3(y)
        y = self.conv4(out2)
        y = torch.cat([y, res1], dim=1)
        out3 = self.c3_4(y)
        return out1, out2, out3


class encoder_head(nn.Module):
    def __init__(self, c: int = 512, ratio: list = [1, 2, 5]):
        super().__init__()
        assert isinstance(ratio, list), "ratio must be a list"
        assert (
            sum(ratio) == 8 and len(ratio) == 3
        ), "the ratios should be a list of length 3 and sum up to 8"

        self.ratio = [c // sum(ratio) * k for k in ratio]

        self.conv1 = nn.Sequential(
            Conv(c // 4, self.ratio[0], k=3, s=2),
            Conv(self.ratio[0], self.ratio[0], k=3, s=2),
        )
        self.conv2 = Conv(c // 2, self.ratio[1], k=3, s=2)
        self.conv3 = Conv(c, self.ratio[2])

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor):
        y1 = self.conv1(x1)
        y2 = self.conv2(x2)
        y3 = self.conv3(x3)

        return torch.cat([y1, y2, y3], dim=1)

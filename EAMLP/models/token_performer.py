"""
Take Performer as T2T Transformer
"""
import math
import torch
import torch.nn as nn
from .transformer_block import Mlp


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, in_dim = None, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.in_dim = in_dim


        self.num_heads = 1
        self.dim = dim
        self.k = 128

        # self.linear_proj = nn.Linear(self.dim, self.in_dim)        
        self.q_linear = nn.Linear(self.dim, self.in_dim) 
        
        self.linear_0 = nn.Linear(self.in_dim, self.k, bias=False)
        
        self.linear_1 = nn.Linear(self.k, self.in_dim)
        self.linear_1.weight.data = self.linear_0.weight.data.permute(1, 0)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.in_dim, self.in_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape

        # idn = self.linear_proj(x)

        x = self.q_linear(x) 
        idn = x[:]

        x = x.view(B, N, -1)

        attn = self.linear_0(x)
        attn = attn.softmax(dim=-2)
        attn = attn / (1e-9 + attn.sum(dim=-1, keepdim=True))
        attn = self.attn_drop(attn)
        x = self.linear_1(attn)

        x = self.proj(x) 
        x = self.proj_drop(x)

        # skip connection
        x = idn + x   # because the original x has different size with current x, use v to do skip connection
        return x


class Token_performer(nn.Module):
    def __init__(self, dim, in_dim, head_cnt=1, kernel_ratio=0.5, dp1=0.1, dp2 = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim, in_dim=in_dim, num_heads=head_cnt, attn_drop=dp1, proj_drop=dp2)
        self.norm2 = nn.LayerNorm(in_dim)
        self.mlp = Mlp(in_features=in_dim, hidden_features=int(in_dim*1), out_features=in_dim, act_layer=nn.GELU, drop=0)

    def forward(self, x):
        x = self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


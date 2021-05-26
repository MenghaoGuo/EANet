class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        assert dim % num_heads == 0 
        self.coef = 4
        self.trans_dims = nn.Linear(dim, dim * self.coef)        
        self.num_heads = self.num_heads * self.coef
        self.k = 256 // self.coef
        self.linear_0 = nn.Linear(dim * self.coef // self.num_heads, self.k)
        self.linear_1 = nn.Linear(self.k, dim * self.coef // self.num_heads)
        

        self.attn_drop = nn.Dropout(attn_drop)        
        self.proj = nn.Linear(dim * self.coef, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape

        x = self.trans_dims(x) # B, N, C 
        x = x.view(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        
        attn = self.linear_0(x)

        attn = attn.softmax(dim=-2)
        attn = attn / (1e-9 + attn.sum(dim=-1, keepdim=True))
        attn = self.attn_drop(attn)
        x = self.linear_1(attn).permute(0,2,1,3).reshape(B, N, -1)
        
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

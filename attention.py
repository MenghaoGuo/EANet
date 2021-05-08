import jittor as jt
from jittor import nn
from jittor import Module
from jittor import init
from jittor.contrib import concat
from resnet import resnet50, resnet101

class External_attention(Module):
    def __init__(self, c):
        super(External_attention, self).__init__()
        self.conv1 = nn.Conv2d(c, c, 1)

        self.k = 64
        self.linear_0 = nn.Conv1d(c, self.k, 1, bias=False)

        self.linear_1 = nn.Conv1d(self.k, c, 1, bias=False)
        self.linear_1.weight.data = self.linear_0.weight.data.permute(1, 0, 2)        
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm(c))        


    def execute(self, x):
        idn = x
        x = self.conv1(x)

        b, c, h, w = x.size()
        n = h*w
        x = x.view(b, c, h*w)   # b * c * n 

        attn = self.linear_0(x) # b, k, n
        attn = F.softmax(attn, dim=-1) # b, k, n

        attn = attn / (1e-9 + attn.sum(dim=1, keepdims=True)) #  # b, k, n
        x = self.linear_1(attn) # b, c, n

        x = x.view(b, c, h, w)
        x = self.conv2(x)
        x = x + idn
        x = F.relu(x)
        return x

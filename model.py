import jittor as jt
from jittor import nn
from jittor import Module
from jittor import init
from jittor.contrib import concat
from resnet import resnet50, resnet101

class EAHead(Module):
    def __init__(self, c):
        super(EAHead, self).__init__()
        self.k = 32 
        self.first_conv = nn.Conv2d(c, c, 1)
        self.k_linear = nn.Conv1d(c, self.k, 1, bias=False)
        self.v_linear = nn.Conv1d(self.k, c, 1, bias=False)
        

    def execute(self, x):
        idn = x[:]
        b, c, h, w = x.size()
        x = self.first_conv(x)
        x = x.view(b, c, -1) # b, c, n 
        attn = self.k_linear(x) # b, c, n
        attn = nn.softmax(attn, dim=-1)
        attn = attn / (attn.sum(dim=1, keepdim=True) + 1e-9) 
        x = self.v_linear(attn) # b, c, n 
        x = x.view(b, c, h, w)
        x = x + idn 
        return x  

class EANet(Module):
    def __init__(self, num_classes=21, output_stride=16):
        super(EANet(), self).__init__()
        self.backbone = resnet50(output_stride)
        self.mid_conv = nn.Conv(2048, 512, 1)
        self.head = EAHead(512)
        self.final_conv = nn.Conv(512, num_classes, 1)

    def execute(self, x):
        imsize = x.shape 
        x = self.backbone(x)  
        x = self.mid_conv(x)
        x = self.head(x) 
        x = self.final_conv(x)
        x = nn.resize(x, size=(imsize[2], imsize[3]), mode='bilinear')
        return x 


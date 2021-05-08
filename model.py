import jittor as jt
from jittor import nn
from jittor import Module
from jittor import init
from jittor.contrib import concat
from resnet import resnet50, resnet101

class ConvBNReLU(Module):
    '''Module for the Conv-BN-ReLU tuple.'''

    def __init__(self, c_in, c_out, kernel_size, stride, padding, dilation):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv(
                c_in, c_out, kernel_size=kernel_size, stride=stride, 
                padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm(c_out)
        self.relu = nn.ReLU()

    def execute(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class External_attention(Module):
    '''
    Arguments:
        c (int): The input and output channel number.
    '''
    def __init__(self, c):
        super(External_attention, self).__init__()
        
        self.conv1 = nn.Conv2d(c, c, 1)

        self.k = 64
        self.linear_0 = nn.Conv1d(c, self.k, 1, bias=False)

        self.linear_1 = nn.Conv1d(self.k, c, 1, bias=False)
        self.linear_1.weight = self.linear_0.weight.permute(1, 0, 2)        
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm(c))        

        self.relu = nn.ReLU()

    def execute(self, x):
        idn = x
        x = self.conv1(x)

        b, c, h, w = x.size()
        n = h*w
        x = x.view(b, c, h*w)   # b * c * n 

        attn = self.linear_0(x) # b, k, n
        attn = nn.softmax(attn, dim=-1) # b, k, n

        attn = attn / (1e-9 + attn.sum(dim=1, keepdims=True)) #  # b, k, n
        x = self.linear_1(attn) # b, c, n

        x = x.view(b, c, h, w)
        x = self.conv2(x)
        x = x + idn
        x = self.relu(x)
        return x



class EANet(Module):
    def __init__(self, num_classes=21, output_stride=16):
        super(EANet, self).__init__()
        self.backbone = resnet50(output_stride)
        self.fc0 = ConvBNReLU(2048, 512, 3, 1, 1, 1)
        self.head = External_attention(512)
        self.fc1 = nn.Sequential(
            ConvBNReLU(512, 256, 3, 1, 1, 1),
            nn.Dropout(p=0.1))
        self.fc2 = nn.Conv2d(256, num_classes, 1)

    def execute(self, x):
        imsize = x.shape 
        x = self.backbone(x)  
        x = self.fc0(x)
        x = self.head(x) 
        x = self.fc1(x)
        x = self.fc2(x)

        x = nn.resize(x, size=(imsize[2], imsize[3]), mode='bilinear')
        return x 

    def get_head(self):
        return [self.fc0, self.head, self.fc1, self.fc2]




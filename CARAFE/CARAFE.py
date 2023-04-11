import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchsummary import summary
#paper:https://arxiv.org/pdf/1905.02188.pdf
#using CARAFE with up-sample can improve accuracy by 2%
class CARAFE(nn.Module):
    def __init__(self, in_ch,mid_ch=64,k_encode=3,k_upsample=5,scale=2):
        super(CARAFE,self).__init__()
        self.scale=scale
        #w
        self.comp=Conv(in_ch,mid_ch,k=1,s=1)
        self.encode=Conv(mid_ch,(k_upsample*scale)**2,k=k_encode,s=1,act=False)
        self.pix_shift=nn.PixelShuffle(scale)
        #x
        self.upsample=nn.Upsample(scale_factor=scale,mode='nearest')
        self.unfold=nn.Unfold(kernel_size=k_encode,dilation=scale,padding=scale*k_upsample//2)
    
    def forward(self, x):
        b,c,h,w=x.size()
        h_m,w_m=h*self.scale,w*self.scale
        #w,bkhw
        w=self.comp(x)
        w=self.encode(w)
        w=self.pix_shift(w)
        w=torch.softmax(w,dim=1)
        #x,bckhw
        x=self.upsample(x)
        x=self.unfold(x)
        x=x.view(b,c,-1,h_m,w_m)

        x=torch.einsum('bkhw,bckhw->bchw',[w,x])
        return x

#Conv coming from org code of yolov5 
class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, self.autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

    def autopad(k, p=None):  # kernel, padding
        # Pad to 'same'
        if p is None:
            p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
        return p


if __name__=="__main__":
    # input=torch.ones([2,3,224,224])
    model=CARAFE(10)
    summary(model.to("cuda"),(3,299,299))
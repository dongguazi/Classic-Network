import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
from torchsummary import summary
#paper:https://arxiv.org/pdf/1807.06521.pdf

class ChannelAttention(nn.Module):
    def __init__(self,in_ch,reducation=16) -> None:
        super(ChannelAttention,self).__init__()
        self.maxpool=nn.AdaptiveMaxPool2d(1)
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.mlp=nn.Sequential(
            nn.Conv2d(in_ch,in_ch//reducation,kernel_size=1,bias=False),
            nn.ReLU(),
            nn.Conv2d(in_ch//reducation,in_ch,kernel_size=1,bias=False),
        )
        self.sigmoid=nn.Sigmoid()

    def forward(self,x):
        out_max=self.maxpool(x)
        out_avg=self.avgpool(x)
        out_max=self.mlp(out_max)
        out_avg=self.mlp(out_avg)
        out=self.sigmoid(out_max+out_avg)
        return out

class SpatialAttention(nn.Module):
    def __init__(self,kernel_size=7) -> None:
        super(SpatialAttention,self).__init__()
        self.conv=nn.Conv2d(2,1,kernel_size=kernel_size,padding=kernel_size//2)
        self.sigmoid=nn.Sigmoid()

    def forward(self,x):
        max_out,_=torch.max(x,dim=1,keepdim=True)
        avg_out=torch.mean(x,dim=1,keepdim=True)
        out=torch.cat([max_out,avg_out],dim=1)
        out=self.conv(out)
        out=self.sigmoid(out)
        return out

class CBAMBlock(nn.Module):
    def __init__(self,in_ch=512,reducation=16,kernel_size=49) -> None:
        super(CBAMBlock,self).__init__()
        self.CA=ChannelAttention(in_ch=in_ch,reducation=reducation)
        self.SA=SpatialAttention(kernel_size=kernel_size)
    def init_weights(self):
        for m in self.modules:
            if isinstance(m,nn.Conv2d):
                init.kaiming_normal_(m.weight,mode='fade_out')
                if m.bias is not None:
                    init.constant_(m.bias,0)
            elif isinstance(m,nn.BatchNorm2d):
                init.constant_(m.weight,1)
                init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear):
                init.normal_(m.weight,mean=0,std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias,0)
    def forward(self,x):
        #b,c,h,w=x.size()
        residual=x
        out=x*self.CA(x)
        out=out*self.SA(out)
        out=out+residual
        return out

if __name__=="__main__":
    # input=torch.ones([2,3,224,224])
    model=CBAMBlock()
    summary(model.to("cuda"),(512,7,7))
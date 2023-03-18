import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GlobalAvgPool2d(nn.Module):
    def __init__(self) -> None:
        super(GlobalAvgPool2d,self).__init__()
    def forward(self,x):
        return F.adaptive_avg_pool2d(x,1).view(x.size(0),-1)

class ConvBlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding) -> None:
        super(ConvBlock,self).__init__()
        self.block=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        x=self.block(x)
        return x
class rSoftMax(nn.Module):
    def __init__(self,groups=1,radix=2) -> None:
        super(rSoftMax,self).__init__()
        self.groups=groups
        self.radix=radix
    def forward(self,x):
        bat=x.size(0)
        x=x.view(bat,self.groups,self.radix,-1).transpose(1,2)
        x=F.softmax(x,dim=1)
        x=x.view(bat,-1,1,1)
        return x

#https://github.com/STomoya/ResNeSt/blob/master/resnest/layers.py
class splitAttention(nn.Module):
    def __init__(self,in_channels,
    out_channels,kernel_size,stride=1,
    padding=0,dilation=1,groups=1,
    bias=False,radix=2,
    reduction_factor=4) -> None:
        super(splitAttention,self).__init__()
        self.radix=radix
        self.radix_conv=nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
            out_channels=out_channels*radix,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups*radix,
            bias=bias
            ),
            nn.BatchNorm2d(out_channels*radix),
            nn.ReLU(inplace=True)
        )
        inter_channels=max(32,in_channels*radix//reduction_factor)

        self.attention=nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=inter_channels,
                kernel_size=1,
                groups=groups
            ),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=inter_channels,
            out_channels=out_channels*radix,
            kernel_size=1,
            groups=groups)
        )
        self.rSoftMax=rSoftMax(groups=groups,radix=radix)
    def forward(self,x):
        pass
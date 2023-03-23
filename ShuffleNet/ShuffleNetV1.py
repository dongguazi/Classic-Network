import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchsummary import summary

class ShuffleNetLayer(nn.Module):
    def __init__(self,in_ch,out_ch,expansion=2) -> None:
        super(ShuffleNetLayer,self).__init__()

class DWConv(nn.Module):
    def __init__(self,in_ch,out_ch,expansion=2) -> None:
        super(DWConv,self).__init__()
        self.inter_ch=in_ch*expansion
        self.conv1=nn.Conv2d(in_ch,self.inter_ch,kernel_size=1,stride=1,padding=0,bias=False)
        self.bn1=nn.ReLU(inplace=True)
        self.conv2_dw=nn.Conv2d(self.inter_ch,self.inter_ch,kernel_size=3,stride=1,padding=1,groups=self.inter_ch,bias=False)
        self.bn2=nn.ReLU(inplace=True)       
        self.conv3=nn.Conv2d(self.inter_ch,in_ch,kernel_size=1,stide=1,padding=0,bias=False)
        self.bn3=nn.ReLU(inplace=True) 
        self.relu=nn.ReLU(inplace=True)
    def forward(self,x):
        indentity=x
        out=self.relu(self.bn1(self.conv1(x)))
        out=self.relu(self.bn2(self.conv2_dw(out)))
        out=self.bn3(self.conv3(out))
        out=out+indentity
        return out

class ShuffleNetLayer(nn.Module):
    def __init__(self,in_ch,out_ch,expansion=2) -> None:
        super(ShuffleNetLayer,self).__init__()
        self.inter_ch=in_ch*expansion
        self.conv1=nn.Conv2d(in_ch,self.inter_ch,kernel_size=1,stride=1,padding=0,bias=False)
        self.bn1=nn.ReLU(inplace=True)
        self.conv2_dw=nn.Conv2d(self.inter_ch,self.inter_ch,kernel_size=3,stride=1,padding=1,groups=self.inter_ch,bias=False)
        self.bn2=nn.ReLU(inplace=True)       
        self.conv3=nn.Conv2d(self.inter_ch,in_ch,kernel_size=1,stide=1,padding=0,bias=False)
        self.bn3=nn.ReLU(inplace=True) 
        self.relu=nn.ReLU(inplace=True)
    def forward(self,x):
        indentity=x
        out=self.relu(self.bn1(self.conv1(x)))
        out=self.relu(self.bn2(self.conv2_dw(out)))
        out=self.bn3(self.conv3(out))
        out=out+indentity
        return out



if __name__=="__main__":
    input=torch.ones([2,3,224,224])
    model=SeNet152(10)
    # res=model(input)
    # print(res.shape)
    summary(model.to("cuda"),(3,224,224))

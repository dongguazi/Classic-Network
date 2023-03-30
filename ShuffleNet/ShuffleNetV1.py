import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchsummary import summary


class ShufflenetV1Block(nn.Module):
    def __init__(self,in_ch,out_ch,mid_ch,stride,group,first_group,expansion=2) -> None:
        super(ShufflenetV1Block,self).__init__()
        assert stride in [1,2]
        self.stride=stride
        self.group=group        
        self.inter_ch=in_ch*expansion

        self.conv1=nn.Sequential(
            nn.Conv2d(in_ch,self.inter_ch,kernel_size=1,stride=1,padding=0,bias=False,groups=1 if first_group else group),
            nn.BatchNorm2d(self.inter_ch),
            nn.ReLU(inplace=True)
        )
        
        self.conv2=nn.Sequential(
            nn.Conv2d(self.inter_ch,self.inter_ch,kernel_size=3,stride=self.stride,padding=1,groups=self.inter_ch,bias=False),
            nn.BatchNorm2d(self.inter_ch),
            nn.Conv2d(self.inter_ch,out_ch,kernel_size=1,stide=1,padding=0,bias=False,groups=group),
            nn.BatchNorm2d(out_ch),
        )
        self.relu=nn.ReLU(inplace=True)
        if stride==2:
            self.branch=nn.AvgPool2d(kernel_size=3,stride=2,padding=1)
    def forward(self,x):
        indentity=x
        out=self.conv1(x)
        out=self.channelShuffle(out)
        out=self.conv2(out)
        if self.stride==1:
           out=self.relu(out+indentity)
        else:
            indentity=self.branch(indentity)
            out=self.relu(torch.cat((out,indentity),dim=1))
        return out

    def channelShuffle(self,x):
        bs,channelNums,H,W=x.shape
        groups_ch=channelNums//self.group
        x=torch.reshape(x,(bs,groups_ch,self.group,H,W))
        x=torch.transpose(x,(0,2,1,3,4))
        x=torch.reshape(x,(bs,channelNums,H,W))
        return x

class ShufflenetV1(nn.Module):
    def __init__(self,class_nums) -> None:
        super(ShufflenetV1,self).__init__()
        pass

    def forward(self,x):
        pass

if __name__=="__main__":
    input=torch.ones([2,3,224,224])
    model=ShufflenetV1(10)
    # res=model(input)
    # print(res.shape)
    summary(model.to("cuda"),(3,224,224))

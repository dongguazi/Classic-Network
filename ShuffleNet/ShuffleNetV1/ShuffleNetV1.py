import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchsummary import summary


class ShufflenetV1Block(nn.Module):
    def __init__(self,in_ch,out_ch,mid_ch,stride,groups,first_group) -> None:
        super(ShufflenetV1Block,self).__init__()
        assert stride in [1,2]
        self.stride=stride
        self.group=groups        
        self.mid_ch=mid_ch
        #notes:has tow hands,stride=1 or 2,
        # --1. stride=1 is  add op
        # --2. stride=2 is  cat op
        if stride==2:
            out_ch=out_ch-in_ch
        self.conv1=nn.Sequential(
            nn.Conv2d(in_ch,self.mid_ch,kernel_size=1,stride=1,bias=False,groups=1 if first_group else groups),
            nn.BatchNorm2d(self.mid_ch),
            nn.ReLU(inplace=True)
        )
        
        self.conv2=nn.Sequential(
            nn.Conv2d(self.mid_ch,self.mid_ch,kernel_size=3,stride=self.stride,padding=1,groups=self.mid_ch,bias=False),
            nn.BatchNorm2d(self.mid_ch),
            nn.Conv2d(self.mid_ch,out_ch,kernel_size=1,stride=1,padding=0,bias=False,groups=groups),
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
        x=torch.transpose(x,1,2)
        x=torch.reshape(x,(bs,channelNums,H,W))
        return x

class ShufflenetV1(nn.Module):
    def __init__(self,class_nums,in_ch=3,model_size='1.0x',groups=3) -> None:
        super(ShufflenetV1,self).__init__()
        self.stage_repeats=[4,8,4]
        self.mode_size=model_size
        #there are many other cases,and we ignore them,you can build them by yourself.
        if  groups==1:
            if model_size=='0.5x':
                self.stage_out_ch=[-1,24,72,144,288]
            elif model_size=='1.0x':
                self.stage_out_ch=[-1,24,144,288,570]
            elif model_size=='2.0x':
                self.stage_out_ch=[-1,48,288,570,1140]
        elif groups==3:
            if model_size=='0.5x':
                self.stage_out_ch=[-1,24,120,240,480]
            elif model_size=='1.0x':
                self.stage_out_ch=[-1,24,240,480,960]
            elif model_size=='2.0x':
                self.stage_out_ch=[-1,48,480,960,1920]
        elif groups==8:
            if model_size=='0.5x':
                self.stage_out_ch=[-1,16,192,384,768]
            elif model_size=='1.0x':
                self.stage_out_ch=[-1,24,384,768,1536]
            elif model_size=='2.0x':
                self.stage_out_ch=[-1,48,768,1536,3072]
        input_ch=self.stage_out_ch[1]
        self.input_layer=nn.Sequential(
            nn.Conv2d(in_ch,input_ch,kernel_size=3,stride=2,padding=3//2, bias=False,groups=3),
            nn.BatchNorm2d(input_ch),
            nn.ReLU(inplace=True)
        )
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=3//2)
        features=[]
        #you can build them on by one
        for inx,repeateNums in enumerate(self.stage_repeats):
            out_ch=self.stage_out_ch[inx+2]
            for i in range(repeateNums):
                stride=2 if i==0 else 1
                firstgroup= inx==0 and i==0
                features.append(ShufflenetV1Block(input_ch,out_ch,mid_ch=out_ch//4, stride=stride,groups=groups,first_group=firstgroup))
                input_ch=out_ch
        self.features=nn.Sequential(*features) 
        self.globpool=nn.AdaptiveAvgPool2d(1)  
        self.classify=nn.Conv2d(self.stage_out_ch[-1],class_nums,kernel_size=1, bias=False) 

    def forward(self,x):
        x=self.input_layer(x)
        x=self.maxpool(x)
        x=self.features(x)
        x=self.globpool(x)
        x=self.classify(x)
        
if __name__=="__main__":
    input=torch.ones([2,3,224,224])
    model=ShufflenetV1(10)
    # model=ShufflenetV1Block(240,240,240//4,stride=1,groups=3,first_group=False)
    # res=model(input)
    # print(res.shape)
    summary(model.to("cuda"),(3,224,224))

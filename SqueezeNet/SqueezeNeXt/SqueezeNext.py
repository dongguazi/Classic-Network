import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchsummary import summary



class CONV_BN_RELU(nn.Module):
    def __init__(self,in_channel,out_channel,kernel_size,padding,stride=1):
        super(CONV_BN_RELU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel,out_channel,kernel_size,stride = stride, padding = padding),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
    def forward(self,x):
        x = self.conv(x)
        return x

class BottleNeck(nn.Module):
    def __init__(self,in_ch,out_ch,stride):
        super(BottleNeck, self).__init__()
        self.block = nn.Sequential(
            CONV_BN_RELU(in_ch,in_ch//2,kernel_size=1,stride=stride,padding=0),
            CONV_BN_RELU(in_ch//2,in_ch//4,kernel_size=1,stride=1,padding=0),
            CONV_BN_RELU(in_ch//4,in_ch//2,kernel_size=(1,3),stride=1,padding=(0,3//2)),
            CONV_BN_RELU(in_ch//2,in_ch//2,kernel_size=(3,1),stride=1,padding=(3//2,0)),
            CONV_BN_RELU(in_ch//2,out_ch,kernel_size=1,stride=1,padding=0),
        )
        self.shortcut=nn.Sequential()
        if stride==2 or out_ch!=in_ch:
            self.shortcut.append(nn.Conv2d(in_ch,out_ch,kernel_size=3,stride=stride,padding=1))
            self.shortcut.append(nn.BatchNorm2d(out_ch))

    def forward(self,x):
        out1 = self.block(x)
        out2=self.shortcut(x)
        out=out1+out2
        return out

class SqueezeNext(nn.Module):
    def __init__(self,class_nums,in_ch=3) -> None:
        super(SqueezeNext,self).__init__()
        self.class_nums=class_nums
        channels=[64,32,64,128,256]
        depth=[6,6,8,1]
        self.conv1=nn.Sequential(
            nn.Conv2d(in_ch,channels[0],kernel_size=7,stride=2,padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True)
        )

        self.stage1=self._make_stages(depth[0],channels[0],channels[1],stride=1)
        self.stage2=self._make_stages(depth[1],channels[1],channels[2],stride=2)
        self.stage3=self._make_stages(depth[2],channels[2],channels[3],stride=2)
        self.stage4=self._make_stages(depth[3],channels[3],channels[4],stride=2)       

        self.classify=nn.Sequential(
            nn.MaxPool2d(7) , 
            nn.Conv2d(channels[-1],128,kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,class_nums,kernel_size=1),
            nn.ReLU(inplace=True)               
        )
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.normal_(m.weight,mean=0.0,std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self,x):
        x=self.conv1(x)
        x=self.stage1(x)
        x=self.stage2(x)
        x=self.stage3(x)
        x=self.stage4(x)
        x=self.classify(x)
        return x

    def _make_stages(self,nums_stages,in_ch,out_ch,stride):
        strides=[stride]+[1]*(nums_stages-1)
        layers=[]
        for i in range(nums_stages):
            layers.append(BottleNeck(in_ch,out_ch,strides[i]))
            in_ch=out_ch
        return nn.Sequential(*layers)

if __name__=="__main__":
    # input=torch.ones([2,3,224,224])
    model=SqueezeNext(10)
    summary(model.to("cuda"),(3,224,224))




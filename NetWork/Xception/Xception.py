import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchsummary import summary


#no use
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

class SeparableConv(nn.Module):
    def __init__(self,in_ch,out_ch,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv, self).__init__()
        self.conv=nn.Conv2d(in_ch,in_ch,kernel_size,stride, padding,dilation,groups=in_ch,bias=bias)
        self.pointwise=nn.Conv2d(in_ch,out_ch,kernel_size=1,stride=1, padding =0,dilation=1,groups=1,bias=bias)
    
    def forward(self,x):
        x = self.conv(x)
        x=self.pointwise(x)
        return x

#three block is in the same block,you can seperate to  three different blocks.
class BottleNeck(nn.Module):
    def __init__(self,in_ch,out_ch,repeates,stride=1,start_with_relu=True,grow_first=True):
        super(BottleNeck, self).__init__()
        if stride!=1 or out_ch!=in_ch:
            self.skip=nn.Sequential(
                nn.Conv2d(in_ch,out_ch,1,stride,bias=False),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.skip=None
       
        layers=[]
        mid_ch=in_ch
        if  grow_first:
            layers.append(nn.ReLU(inplace=True))
            layers.append(SeparableConv(in_ch,out_ch,kernel_size=3,stride=1,padding=1,bias=False))
            layers.append(nn.BatchNorm2d(out_ch))
            mid_ch=out_ch
       
        for i in range (repeates-1):
            layers.append(nn.ReLU(inplace=True))
            layers.append(SeparableConv(mid_ch,mid_ch,kernel_size=3,stride=1,padding=1,bias=False))
            layers.append(nn.BatchNorm2d(mid_ch))

        if  not grow_first:
            layers.append(nn.ReLU(inplace=True))
            layers.append(SeparableConv(in_ch,out_ch,kernel_size=3,stride=1,padding=1,bias=False))
            layers.append(nn.BatchNorm2d(out_ch))
            mid_ch=out_ch
        
        if not start_with_relu:
            layers=layers[1:]
       
        if stride!=1:
            layers.append(nn.MaxPool2d(kernel_size=3,stride=2,padding=3//2))
        self.layers=nn.Sequential(*layers)
    def forward(self,x):
        out1=self.layers(x)
        if self.skip!=None:
            out2=self.skip(x)
        else:
            out2=x
        out=out1+out2
        return out

class Xception(nn.Module):
    def __init__(self,class_nums,in_ch=3) -> None:
        super(Xception,self).__init__()
        self.class_nums=class_nums
        self.conv1=nn.Sequential(
            nn.Conv2d(in_ch,32,kernel_size=3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(32,64,kernel_size=3,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.block1_1=BottleNeck(64,128,repeates= 2,stride=2,start_with_relu=False,grow_first=True)
        self.block1_2=BottleNeck(128,256,repeates= 2,stride=2,start_with_relu=True,grow_first=True)
        self.block1_3=BottleNeck(256,728,repeates= 2,stride=2,start_with_relu=True,grow_first=True)
        self.block2_1=BottleNeck(728,728,repeates= 3,stride=1,start_with_relu=True,grow_first=True)
        self.block2_2=BottleNeck(728,728,repeates= 3,stride=1,start_with_relu=True,grow_first=True)
        self.block2_3=BottleNeck(728,728,repeates= 3,stride=1,start_with_relu=True,grow_first=True)
        self.block2_4=BottleNeck(728,728,repeates= 3,stride=1,start_with_relu=True,grow_first=True)
        self.block2_5=BottleNeck(728,728,repeates= 3,stride=1,start_with_relu=True,grow_first=True)
        self.block2_6=BottleNeck(728,728,repeates= 3,stride=1,start_with_relu=True,grow_first=True)
        self.block2_7=BottleNeck(728,728,repeates= 3,stride=1,start_with_relu=True,grow_first=True)
        self.block2_8=BottleNeck(728,728,repeates= 3,stride=1,start_with_relu=True,grow_first=True)
        self.block3=BottleNeck(728,1024,repeates= 2,stride=2,start_with_relu=True,grow_first=False)
        self.conv3=nn.Sequential(
            SeparableConv(1024,1536,kernel_size= 3,stride=1,padding=1),
            nn.BatchNorm2d(1536),
            nn.ReLU(inplace=True),
        )
        self.conv4=nn.Sequential(
            SeparableConv(1536,2048,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
        )
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.fc=nn.Conv2d(2048,class_nums,kernel_size=1,bias=False)
        self.ress=nn.Softmax(dim=-1)

        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.normal_(m.weight,mean=0.0,std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.block1_1(x)
        x=self.block1_2(x)
        x=self.block1_3(x)
        x=self.block2_1(x)
        x=self.block2_2(x)
        x=self.block2_3(x)
        x=self.block2_4(x)
        x=self.block2_5(x)
        x=self.block2_6(x)
        x=self.block2_7(x)
        x=self.block2_8(x)
        x=self.block3(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.avgpool(x)
        x=self.fc(x)
        x=torch.reshape(x,(x.size(0),-1))
        print(x.size())
        x=self.ress(x)
        return x

if __name__=="__main__":
    # input=torch.ones([2,3,224,224])
    model=Xception(10)
    summary(model.to("cuda"),(3,299,299))




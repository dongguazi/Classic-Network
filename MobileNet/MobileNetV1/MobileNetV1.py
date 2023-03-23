import tensorrt as trt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchsummary import summary

class DepthBlock(nn.Module):

    def __init__(self,in_channels,out_channels,stride=1):
        super(DepthBlock,self).__init__()
        self.conv1=nn.Conv2d(in_channels,in_channels
        ,kernel_size=3,stride=stride,padding=1,
         groups=in_channels,bias=False)
        self.bn1=nn.BatchNorm2d(in_channels)
        self.relu=nn.ReLU(inplace=True)
        self.conv2=nn.Conv2d(in_channels,out_channels
        ,kernel_size=1,stride=1,padding=0,bias=False)
        self.bn2=nn.BatchNorm2d(out_channels)
    
    def forward(self,x):
        x=self.relu(self.bn1(self.conv1(x)))
        x=self.relu(self.bn2(self.conv2(x)))
        return x

class conv_bn(nn.Module):
    def __init__(self,in_ch,out_ch,stride) -> None:
        super().__init__()
        self.conv1=nn.Conv2d(in_ch,out_ch
        ,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(out_ch)
        self.relu=nn.ReLU(inplace=True)
    def forward(self,x):
        x=self.relu(self.bn1(self.conv1(x)))
        return x

class MobileNetV1(nn.Module):
    
    def __init__(self,num_classes) -> None:
        super(MobileNetV1,self).__init__()
        self.conv1=conv_bn(in_ch=3,out_ch=32,stride=2)
        self.conv2=DepthBlock(in_channels=32,out_channels=64,stride=1)
        self.conv3=DepthBlock(in_channels=64,out_channels=128,stride=2)
        self.conv4=DepthBlock(in_channels=128,out_channels=128,stride=1)
        self.conv5=DepthBlock(in_channels=128,out_channels=256,stride=2)
        self.conv6=DepthBlock(in_channels=256,out_channels=256,stride=1)
        self.conv7=DepthBlock(in_channels=256,out_channels=512,stride=2)
        self.conv8_1=DepthBlock(in_channels=512,out_channels=512,stride=1)
        self.conv8_2=DepthBlock(in_channels=512,out_channels=512,stride=1)
        self.conv8_3=DepthBlock(in_channels=512,out_channels=512,stride=1)
        self.conv8_4=DepthBlock(in_channels=512,out_channels=512,stride=1)
        self.conv8_5=DepthBlock(in_channels=512,out_channels=512,stride=1)
        self.conv9=DepthBlock(in_channels=512,out_channels=1024,stride=2)
        self.conv10=DepthBlock(in_channels=1024,out_channels=1024,stride=1)

        self.agvpool=nn.AvgPool2d(7)
        self.fc=nn.Conv2d(1024,num_classes,kernel_size=1,bias=False)
        self.flatten=nn.Flatten()
        self.softmax=nn.Softmax(dim=-1)

    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.conv5(x)
        x=self.conv6(x)
        x=self.conv7(x)
        x=self.conv8_1(x)
        x=self.conv8_2(x)
        x=self.conv8_3(x)
        x=self.conv8_4(x)
        x=self.conv8_5(x)
        x=self.conv9(x)
        x=self.conv10(x)
        x=self.agvpool(x)
        # x=torch.reshape(x.shape)
        x=self.fc(x)
        x=self.flatten(x)
        x=self.softmax(x)    
        return x


if __name__ =="__main__":
    input=torch.ones([2,3,224,224])
    model=MobileNetV1(10)
    # res=model(input)
    # print(res.shape)
    summary(model.to("cuda"),(3,224,224))

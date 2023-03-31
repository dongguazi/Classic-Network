import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchsummary import summary


class Fire(nn.Module):
    def __init__(self,in_ch,squeeze_ch,expand1x1_ch,expand3x3_ch) -> None:
        super(Fire,self).__init__()
        self.squeeze=nn.Conv2d(in_ch,squeeze_ch,kernel_size=1,bias=False)
        self.squeeze_relu=nn.ReLU(inplace=True)
        self.expand1x1=nn.Conv2d(squeeze_ch,expand1x1_ch,kernel_size=1)
        self.expand1x1_relu=nn.ReLU(inplace=True)
        self.expand3x3=nn.Conv2d(squeeze_ch,expand3x3_ch,kernel_size=3,padding=1)
        self.expand3x3_relu=nn.ReLU(inplace=True)

    def forward(self,x):
        x=self.squeeze_relu(self.squeeze(x))
        out1=self.expand1x1_relu(self.expand1x1(x))
        out2=self.expand3x3_relu(self.expand3x3(x))
        out=torch.cat([out1,out2],dim=1)
        return out


class SqueezeNet(nn.Module):
    def __init__(self,class_nums,in_ch=3,bypass=True) -> None:
        super(SqueezeNet,self).__init__()
        self.class_nums=class_nums
        self.bypass=bypass
        self.conv1=nn.Conv2d(in_ch,96,kernel_size=7,stride=2,padding=3)
        self.relu1=nn.ReLU(inplace=True)
        self.maxpool1=nn.MaxPool2d(kernel_size=3, stride=2)
        self.fire2=Fire(96,16,64,64)
        self.fire3=Fire(128,16,64,64)
        self.fire4=Fire(128,32,128,128)
        self.maxpool2=nn.MaxPool2d(kernel_size=3,stride=2)
        self.fire5=Fire(256,32,128,128)
        self.fire6=Fire(256,48,192,192)
        self.fire7=Fire(384,48,192,192)
        self.fire8=Fire(384,64,256,256)
        self.maxpool3=nn.MaxPool2d(kernel_size=3,stride=2)
        self.fire9=Fire(512,64,256,256)

        self.classify=nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(512,class_nums,kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1))            
        )
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.normal_(m.weight,mean=0.0,std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self,x):
        if self.bypass:
            x=self.maxpool1(self.relu1(self.conv1(x)))
            x=self.fire2(x)
            # indentity=x
            # x=indentity+self.fire3(x)
            # x=self.fire4(x)
            # x=self.maxpool2(x)
            # indentity=x
            # x=indentity+self.fire5(x)
            # x=self.fire6(x)
            # indentity=x
            # x=indentity+self.fire7(x)
            # x=self.fire8(x)
            # x=self.maxpool3(x)
            # indentity=x
            # x=indentity+self.fire9(x)
            # x=self.classify(x)
        return torch.flatten(x,1)
    

if __name__=="__main__":
    input=torch.ones([2,3,224,224])
    model=SqueezeNet(10)
    summary(model.to("cuda"),(3,224,224))




import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchsummary import summary

class Bottleneck(nn.Module):
    expansion=4
    def __init__(self,in_channels,out_channels,groups=1,width_per_group=64,down_sample=None,stride=1):
        super(Bottleneck,self).__init__()
        width=int(out_channels*(width_per_group/64.))*groups
        self.conv1=nn.Conv2d(in_channels,width,kernel_size=1,stride=1,padding=0  )
        self.bn1=nn.BatchNorm2d(width)

        self.conv2=nn.Conv2d(width,width,kernel_size=3,stride=stride,padding=1,bias=False, groups=groups )
        self.bn2=nn.BatchNorm2d(width)

        self.conv3=nn.Conv2d(width,out_channels*self.expansion,kernel_size=1,stride=1,padding=0  )
        self.bn3=nn.BatchNorm2d(out_channels*self.expansion)

        self.down_sample=down_sample
        self.stride=stride
        self.relu=nn.ReLU(inplace=True)
    
    def forward(self,x):
        identity=x.clone()
        if self.down_sample:
           identity=self.down_sample(identity)

        x=self.relu(self.bn1(self.conv1(x)))
        x=self.relu(self.bn2(self.conv2(x)))
        x=self.bn3(self.conv3(x))
        
        # print(x.shape)
        # print(identity.shape)
        x+=identity
        x=self.relu(x)
        return x


class PlainBlock(nn.Module):
    expansion=1
    def __init__(self,in_channels,out_channels,down_sample=None,stride=1) -> None:
        super(PlainBlock,self).__init__()
        self.conv1=nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1,bias=False  )
        self.bn1=nn.BatchNorm2d(out_channels)

        self.conv2=nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=stride,padding=1,bias=False  )
        self.bn2=nn.BatchNorm2d(out_channels)

        self.down_sample=down_sample
        self.stride=stride
        self.relu=nn.ReLU()
    
    def forward(self,x):
        identity=x.clone()
        x=self.relu(self.bn1(self.conv1(x)))
        x=self.bn2(self.conv2(x))

        if self.down_sample:
            identity=self.down_sample(identity)
        print(x.shape)
        print(identity.shape)
        x+=identity
        x=self.relu(x)

        return x

class Resnext(nn.Module):
    first=False
    cout=1
    def __init__(self,block,layer_list,num_classes,input_channels,groups=1,width_per_group=64,complex=True) -> None:
        super(Resnext,self).__init__()
        self.in_channels=64
        self.groups=groups
        self.width_per_group=width_per_group

        self.conv1=nn.Conv2d(input_channels,self.in_channels,kernel_size=7,stride=2,padding=3)
        self.bn1=nn.BatchNorm2d(self.in_channels)
        self.relu=nn.ReLU(inplace=True)
        self.maxpooling=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.layer1=self.make_layer(block,layer_list[0],planes=64,stride=1)
        self.layer2=self.make_layer(block,layer_list[1],planes=128,stride=2 if complex==True else 1)
        self.layer3=self.make_layer(block,layer_list[2],planes=256,stride=2 if complex==True else 1)
        self.layer4=self.make_layer(block,layer_list[3],planes=512,stride=2 if complex==True else 1)
        
        self.avgpooling=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Linear(block.expansion*512,num_classes)

        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')

    def forward(self,x):
        x=self.relu(self.bn1(self.conv1(x)))
        x=self.maxpooling(x)
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        x=self.avgpooling(x)
        # print("x.shape")
        # print(x.shape)
        # self.cout+=1
        x=x.reshape(x.shape[0],-1)
        print(x.shape)
        x=self.fc (x)
        return x
     #
    def make_layer(self,Resblock,block_nums,planes,stride=1):
        downsample=None
        layers=[]
        #stride!=1说明输入通道要降维，对上一层来的图片大小，缩放一倍
        # self.in_channels!=planes*Resblock.expansion 针对第一个layer输入的图片进行缩放
        # 每个layer的第一层需要把上一级输出维度变成我当前的基准维度，然后重复基准维度和输出维度不变
        #例如当前我的基准是我的基准256，来自上一层是512，把它变成1024，然后后面的Resblock都只需要【1024-256-1024】的重复过程
        
        if complex==True:
            #下面两种if判断的条件是等价的
            #if stride!=1 or self.in_channels!=planes*Resblock.expansion:
            if stride!=1 or self.first==False:
                self.first=True
                downsample=nn.Sequential(nn.Conv2d(self.in_channels,planes*Resblock.expansion,kernel_size=1,stride=stride),
                nn.BatchNorm2d(planes*Resblock.expansion))
            layers.append(Resblock(self.in_channels,planes,downsample,stride=stride))
            self.in_channels=planes*Resblock.expansion
        else:
            downsample=nn.Sequential(nn.Conv2d(self.in_channels,planes*Resblock.expansion,kernel_size=1,stride=stride),
            nn.BatchNorm2d(planes*Resblock.expansion))
            layers.append(Resblock(self.in_channels,planes,downsample,stride=stride))
            self.in_channels=planes*Resblock.expansion

        for i in range(block_nums-1):
             layers.append(Resblock(self.in_channels,planes))
        return nn.Sequential(*layers)



class Resnet_simple(nn.Module):
    first=False
    cout=1
    def __init__(self,Resblock,layer_list,num_classes,input_channels) -> None:
        super(Resnet_simple,self).__init__()
        self.in_channels=64
        self.conv1=nn.Conv2d(input_channels,64,kernel_size=7,stride=2,padding=3)
        self.bn1=nn.BatchNorm2d(64)
        self.relu=nn.ReLU()
        self.maxpooling=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.layer1=self.make_layer(Resblock,layer_list[0],planes=64,stride=1)
        self.layer2=self.make_layer(Resblock,layer_list[1],planes=128,stride=1)
        self.layer3=self.make_layer(Resblock,layer_list[2],planes=256,stride=1)
        self.layer4=self.make_layer(Resblock,layer_list[3],planes=512,stride=1)
        
        self.avgpooling=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Linear(Resblock.expansion*512,num_classes)

    def forward(self,x):

        x=self.relu(self.bn1(self.conv1(x)))
        x=self.maxpooling(x)
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        x=self.avgpooling(x)
        # print("x.shape")
        # print(x.shape)
        # self.cout+=1
        x=x.reshape(x.shape[0],-1)
        print(x.shape)
        x=self.fc (x)
        return x
     #
    def make_layer(self,Resblock,block_nums,planes,stride=1):
        downsample=None
        layers=[]
        #stride!=1说明输入通道要降维，对上一层来的图片大小，缩放一倍
        # self.in_channels!=planes*Resblock.expansion 针对第一个layer输入的图片进行缩放
        # 每个layer的第一层需要把上一级输出维度变成我当前的基准维度，然后重复基准维度和输出维度不变
        #例如当前我的基准是我的基准256，来自上一层是512，把它变成1024，然后后面的Resblock都只需要【1024-256-1024】的重复过程
        
        #下面两种if判断的条件是等价的
        #if stride!=1 or self.in_channels!=planes*Resblock.expansion:

        downsample=nn.Sequential(nn.Conv2d(self.in_channels,planes*Resblock.expansion,kernel_size=1,stride=stride),
        nn.BatchNorm2d(planes*Resblock.expansion))
        layers.append(Resblock(self.in_channels,planes,downsample,stride=stride))
        self.in_channels=planes*Resblock.expansion
        for i in range(block_nums-1):
             layers.append(Resblock(self.in_channels,planes))
        return nn.Sequential(*layers)

# 深度在三层以下的resblock使用resnext没有意义
def Resnext50_32x4d(num_classes=10,channels=3):
    layer_list=[3,4,6,3]
    groups = 32
    width_per_group = 4
    return  Resnext(Bottleneck,layer_list,num_classes,channels,groups=groups,width_per_group=width_per_group)  

def Resnext101_32x8d(num_classes=10,channels=3):
    layer_list=[3,4,23,3]
    groups = 32
    width_per_group = 4
    return  Resnext(Bottleneck,layer_list,num_classes,channels,groups=groups,width_per_group=width_per_group)  


if __name__ =="__main__":
    input=torch.ones([2,3,224,224])
    model=Resnext50_32x4d(10)
    # res=model(input)
    # print(res.shape)
    summary(model.to("cuda"),(3,224,224))

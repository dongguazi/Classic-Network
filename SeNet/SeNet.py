import tensorrt as trt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchsummary import summary

# SeNet get form ResNet :only modify the Bottleneck,other is the same .
class Bottleneck(nn.Module):
    expansion=4
    def __init__(self,in_channels,out_channels,down_sample=None,stride=1):
        super(Bottleneck,self).__init__()

        self.conv1=nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,padding=0  )
        self.bn1=nn.BatchNorm2d(out_channels)

        self.conv2=nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=stride,padding=1  )
        self.bn2=nn.BatchNorm2d(out_channels)

        self.conv3=nn.Conv2d(out_channels,out_channels*self.expansion,kernel_size=1,stride=1,padding=0  )
        self.bn3=nn.BatchNorm2d(out_channels*self.expansion)
        self.relu=nn.ReLU()
        
        if out_channels==64:
            self.globalAvgPool=nn.AvgPool2d(56,stride=1)
        elif out_channels==128:
            self.globalAvgPool=nn.AvgPool2d(28,stride=1)
        elif out_channels==256:
            self.globalAvgPool=nn.AvgPool2d(14,stride=1)
        elif out_channels==512:
            self.globalAvgPool=nn.AvgPool2d(7,stride=1)
        
        self.fc1=nn.Linear(in_features=out_channels*self.expansion,out_features=round(out_channels/self.expansion))
        self.fc2=nn.Linear(in_features=round(out_channels/self.expansion),out_features=round(out_channels*self.expansion))
        self.sigmoid=nn.Sigmoid()

        self.down_sample=down_sample
        self.stride=stride
    
    def forward(self,x):
        identity=x.clone()
        #inception
        x=self.relu(self.bn1(self.conv1(x)))
        x=self.relu(self.bn2(self.conv2(x)))
        x=self.bn3(self.conv3(x))
        if self.down_sample:
            identity=self.down_sample(identity)
        #se 
        original_x=x
        x=self.globalAvgPool(x)
        x=x.view(x.size(0),-1)
        x=self.fc1(x)
        x=self.relu(x)
        x=self.fc2(x)
        x=self.sigmoid(x)
        x=x.view(x.size(0),x.size(1),1,1)
        x=x*original_x
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

class SeNet(nn.Module):
    first=False
    cout=1
    def __init__(self,Resblock,layer_list,num_classes,input_channels,complex=True) -> None:
        super(SeNet,self).__init__()
        self.in_channels=64
        self.conv1=nn.Conv2d(input_channels,64,kernel_size=7,stride=2,padding=3)
        self.bn1=nn.BatchNorm2d(64)
        self.relu=nn.ReLU()
        self.maxpooling=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.layer1=self.make_layer(Resblock,layer_list[0],planes=64,stride=1)
        self.layer2=self.make_layer(Resblock,layer_list[1],planes=128,stride=2 if complex==True else 1)
        self.layer3=self.make_layer(Resblock,layer_list[2],planes=256,stride=2 if complex==True else 1)
        self.layer4=self.make_layer(Resblock,layer_list[3],planes=512,stride=2 if complex==True else 1)
        
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
        

        if complex==True:
            #下面两种if判断的条件是等价的
            if stride!=1 or self.in_channels!=planes*Resblock.expansion:
            #if stride!=1 or self.first==False:
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




def SeNet50(num_classes,channels=3):
    layer_list=[3,4,6,3]
    return  SeNet(Bottleneck,layer_list,num_classes,channels)  

def SeNet101(num_classes,channels=3):
    layer_list=[3,4,23,3]
    return  SeNet(Bottleneck,layer_list,num_classes,channels)  

def SeNet152(num_classes,channels=3):
    layer_list=[3,4,36,3]
    return  SeNet(Bottleneck,layer_list,num_classes,channels)  

if __name__ =="__main__":
    input=torch.ones([2,3,224,224])
    model=SeNet50(10)
    # res=model(input)
    # print(res.shape)
    summary(model.to("cuda"),(3,224,224))

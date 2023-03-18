#import tensorrt as trt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchsummary import summary

  # SeNet get form ResNext :only modify the SELayer,other is the same . 
class SELayer(nn.Module):
    def __init__(self,channel,reduction=16) -> None:
        super(SKLayer,self).__init__()
        self.globalAvgPool=nn.AdaptiveAvgPool2d(1)
        self.fc=nn.Sequential(
            nn.Linear(channel,channel//reduction,bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel//reduction,channel,bias=False),
            nn.Sigmoid()
        )
    def forward(self,x):
        b,c,_,_=x.size()
        original_x=x
        x=self.globalAvgPool(x).view(b,c)
        x=self.fc(x).view(b,c,1,1)
        return  original_x*x
 
 #new version, 
class SKLayer(nn.Module):
    def __init__(self,in_channels,stride=1,groups=32,M=2,r=16,L=32) -> None:
        super(SKLayer,self).__init__()
        d=max(in_channels/r,32)
        self.M=M
        self.in_channels=in_channels
        self.convs=nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(in_channels,in_channels,stride=stride,padding=i+1,dilation=i+1,groups=groups,bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            ))
        self.gap=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Sequential(
            nn.Conv2d(in_channels,d,kernel_size=1,stride=1,bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU()            
            )
        self.fcs=nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.conv2d(d,in_channels,kernel_size=1,stride=1)
            )
        self.softmax=nn.Softmax(dim=1)
                
    def forward(self,x):
        pass

 #new version, 
class SKBlock(nn.Module):
    expansion=4
    def __init__(self,in_channels,plane,down_sample=None,stride=1,reduction=16):
        super(SKBlock,self).__init__()
        #inception
        self.conv1=nn.Conv2d(in_channels,plane,kernel_size=1,stride=1,padding=0  )
        self.bn1=nn.BatchNorm2d(plane)

        self.conv2=nn.Conv2d(plane,plane,kernel_size=3,stride=stride,padding=1  )
        self.bn2=nn.BatchNorm2d(plane)

        self.conv3=nn.Conv2d(plane,plane*self.expansion,kernel_size=1,stride=1,padding=0  )
        self.bn3=nn.BatchNorm2d(plane*self.expansion)
        self.relu=nn.ReLU()
        self.se=SKLayer(plane*self.expansion,reduction)
   
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
        x=self.se(x)        
        x+=identity
        x=self.relu(x)
        return x


class SeNet(nn.Module):
    first=False
    cout=1
    def __init__(self,block,layer_list,num_classes,input_channels,complex=True) -> None:
        super(SeNet,self).__init__()
        self.in_channels=64
        self.conv1=nn.Conv2d(input_channels,64,kernel_size=7,stride=2,padding=3)
        self.bn1=nn.BatchNorm2d(64)
        self.relu=nn.ReLU()
        self.maxpooling=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.layer1=self.make_layer(block,layer_list[0],plane=64,stride=1)
        self.layer2=self.make_layer(block,layer_list[1],plane=128,stride=2 if complex==True else 1)
        self.layer3=self.make_layer(block,layer_list[2],plane=256,stride=2 if complex==True else 1)
        self.layer4=self.make_layer(block,layer_list[3],plane=512,stride=2 if complex==True else 1)
        
        self.avgpooling=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Linear(block.expansion*512,num_classes)

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
    def make_layer(self,Resblock,block_nums,plane,stride=1):
        downsample=None
        layers=[]
        #stride!=1说明输入通道要降维，对上一层来的图片大小，缩放一倍
        # self.in_channels!=planes*Resblock.expansion 针对第一个layer输入的图片进行缩放
        # 每个layer的第一层需要把上一级输出维度变成我当前的基准维度，然后重复基准维度和输出维度不变
        #例如当前我的基准是我的基准256，来自上一层是512，把它变成1024，然后后面的Resblock都只需要【1024-256-1024】的重复过程
        

        if complex==True:
            #下面两种if判断的条件是等价的
            if stride!=1 or self.in_channels!=plane*Resblock.expansion:
            #if stride!=1 or self.first==False:
                self.first=True
                downsample=nn.Sequential(nn.Conv2d(self.in_channels,plane*Resblock.expansion,kernel_size=1,stride=stride),
                nn.BatchNorm2d(plane*Resblock.expansion))
            layers.append(Resblock(self.in_channels,plane,downsample,stride=stride))
            self.in_channels=plane*Resblock.expansion
        else:
            downsample=nn.Sequential(nn.Conv2d(self.in_channels,plane*Resblock.expansion,kernel_size=1,stride=stride),
            nn.BatchNorm2d(plane*Resblock.expansion))
            layers.append(Resblock(self.in_channels,plane,downsample,stride=stride))
            self.in_channels=plane*Resblock.expansion

        for i in range(block_nums-1):
             layers.append(Resblock(self.in_channels,plane))
        return nn.Sequential(*layers)




def SKNet50(num_classes,channels=3):
    layer_list=[3,4,6,3]
    return  SeNet(SKBlock,layer_list,num_classes,channels)  
    # return  SeNet(Bottleneck,layer_list,num_classes,channels)  

 

def SKNet101(num_classes,channels=3):
    layer_list=[3,4,23,3]
    return  SeNet(SKBlock,layer_list,num_classes,channels)  

def SKNet152(num_classes,channels=3):
    layer_list=[3,4,36,3]
    return  SeNet(SKBlock,layer_list,num_classes,channels)  

if __name__ =="__main__":
    input=torch.ones([2,3,224,224])
    model=SeNet50(10)
    # res=model(input)
    # print(res.shape)
    summary(model.to("cuda"),(3,224,224))

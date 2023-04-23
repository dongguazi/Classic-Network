#import tensorrt as trt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchsummary import summary

 #new version, 
class SKLayer(nn.Module):
    def __init__(self,in_channels,stride=1,G=32,M=2,r=16,L=32) -> None:
        super(SKLayer,self).__init__()
        d=max(int(in_channels/r),32)
        self.M=M
        self.in_channels=in_channels
        self.convs=nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(in_channels,in_channels,kernel_size=3, stride=stride,padding=i+1,dilation=i+1,groups=G,bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            ))
        self.gap=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Sequential(
            nn.Conv2d(in_channels,d,kernel_size=1,stride=1,bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True)            
            )
        self.fcs=nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Conv2d(d,in_channels,kernel_size=1,stride=1,bias=False)
            )
        self.softmax=nn.Softmax(dim=1)
                
    def forward(self,x):
        bs=x.size(0)
        #多分组M个[bs,32,224,224]
        feats=[conv(x) for conv in self.convs]
        #多分组合并[bs，32*M，224,224]
        feats=torch.cat(feats,dim=1)
        #按组拆分[bs,M,32,224,224]
        feats=feats.view(bs,self.M,self.in_channels,feats.shape[2],feats.shape[3])
        feats_U=torch.sum(feats,dim=1)
        feats_S=self.gap(feats_U)
        feats_Z=self.fc(feats_S)
        #多分组M个[bs,32,1,1]
        attention=[fc(feats_Z) for fc in self.fcs]
        #多分组合并[bs，32*M，1,1]
        attention=torch.cat(attention,dim=1)

        #按组拆分[bs,M,32,1,1]
        attention=attention.view(bs,self.M,self.in_channels,1,1)

        attention=self.softmax(attention)
        feats_V=torch.sum(feats*attention,dim=1)
        return feats_V


 #new version, 
class SKBlock(nn.Module):
    expansion=4
    def __init__(self,in_channels,middle_features,out_channels,stride=1,M=2,G=32,r=16,L=32):
        super(SKBlock,self).__init__()
        #inception
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels,middle_features,kernel_size=1,stride=1,bias=False),
            nn.BatchNorm2d(middle_features),
            nn.ReLU(inplace=True)
        )
        #in_channels,stride=1,G=32,M=2,r=16,L=32
        self.SK=SKLayer(middle_features,stride=stride,G=G,M=M,r=r,L=L)

        self.conv2=nn.Sequential(
        nn.Conv2d(middle_features,out_channels,kernel_size=1,stride=1,padding=0 ,bias=False ),
        nn.BatchNorm2d(out_channels)
        )
        if in_channels==out_channels:
            self.shortcut=nn.Sequential()
        else:
            self.shortcut=nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.stride=stride
        self.relu=nn.ReLU(inplace=True)
    
    def forward(self,x):
        identity=x
        out=self.conv1(x)
        out=self.SK(out)
        out=self.conv2(out) 

        out+=self.shortcut(identity)
        out=self.relu(out)
        return out


class SKNet(nn.Module):
    first=False
    cout=1
    def __init__(self,block,layer_list,num_classes,input_channels,complex_mode=True) -> None:
        super(SKNet,self).__init__()
        self.middle_channels=64
        self.complex_mode=complex_mode
        self.conv1=nn.Sequential(
            nn.Conv2d(input_channels,self.middle_channels,kernel_size=7,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(self.middle_channels),
            nn.ReLU(inplace=True)
        )

        self.maxpooling=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        #block,block_nums,in_feats,middle_feats,out_feats,stride=1
        self.layer1=self.make_layer(block,layer_list[0],in_feats=64,middle_feats=128,out_feats=256, stride=1)
        self.layer2=self.make_layer(block,layer_list[1],in_feats=256,middle_feats=256,out_feats=512,stride=2 if complex_mode==True else 1)
        self.layer3=self.make_layer(block,layer_list[2],in_feats=512,middle_feats=512,out_feats=1024,stride=2 if complex_mode==True else 1)
        self.layer4=self.make_layer(block,layer_list[3],in_feats=1024,middle_feats=1024,out_feats=2048,stride=2 if complex_mode==True else 1)
        
        self.avgpooling=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Linear(block.expansion*512,num_classes)

    def forward(self,x):
        x=self.conv1(x)
        x=self.maxpooling(x)
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        x=self.avgpooling(x)
        x=x.reshape(x.shape[0],-1)
        x=self.fc (x)
        return x
     #
    def make_layer(self,block,block_nums,in_feats,middle_feats,out_feats,stride=1):
        layers=[block(in_feats,middle_feats,out_feats,stride=stride)]
        for i in range(block_nums-1):
             layers.append(block(out_feats,middle_feats,out_feats))
        return nn.Sequential(*layers)


def SKNet50(num_classes,channels=3):
    layer_list=[3,4,6,3]
    return  SKNet(SKBlock,layer_list,num_classes,channels)  
    # return  SeNet(Bottleneck,layer_list,num_classes,channels)  

 

def SKNet101(num_classes,channels=3):
    layer_list=[3,4,23,3]
    return  SKNet(SKBlock,layer_list,num_classes,channels)  

def SKNet152(num_classes,channels=3):
    layer_list=[3,4,36,3]
    return  SKNet(SKBlock,layer_list,num_classes,channels)  

def SKNettest(num_classes,channels=3):
    layer_list=[3,4,6,3]
    return SKLayer(32)
if __name__ =="__main__":
    input=torch.ones([2,3,224,224])
    model=SKNet50(10)
    # model=SKNettest(10)
    # res=model(input)
    # print(res.shape)
    summary(model.to("cuda"),(3,224,224))

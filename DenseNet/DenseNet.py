
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchsummary import summary
import math
def conv_block(in_channels,out_channels,kernel_size=3,stride=1,padding=0,bias=False):
    return nn.Sequential(
        nn.BatchNorm2d(in_channels),nn.ReLU(inplace=True),
        nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding,bias=bias)
    )


class BottleneckBlock(nn.Module):
    def __init__(self,in_channels,out_channels,dropRate) -> None:
        super(BottleneckBlock,self).__init__()    
        inter_planes=out_channels*4
        # self.conv1=conv_block(in_channels,inter_planes,kernel_size=1,stride=1,padding=0,bias=False)
        # self.conv2=conv_block(inter_planes,out_channels,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)
        
        self.dropRate=dropRate
    def forward(self,x):
        out = self.conv1(self.relu(self.bn1(x)))
        # if self.dropRate>0:
        #     out=F.dropout(out,p=self.dropRate,inplace=False,training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        # if self.dropRate>0:
        #     out=F.dropout(out,p=self.dropRate,inplace=False,training=self.training)
        return torch.cat([x,out],dim=1)

class TransitionBlock(nn.Module):
    def __init__(self,in_planes,out_plains,dropRate) -> None:
        super(TransitionBlock,self).__init__()
        self.conv1=conv_block(in_planes,out_plains,kernel_size=1,stride=1,padding=0,bias=False)
        # self.bn1=nn.BatchNorm2d(in_planes)
        # self.relu=nn.ReLU(inplace=True)
        # self.conv1=nn.Conv2d(in_planes,out_plains,kernel_size=1,stride=1,padding=0,bias=False)
        self.dropRate=dropRate
        self.avgploing=nn.AvgPool2d(kernel_size=2,stride=2)
    def forward(self,x):
        out=self.conv1(x)
        if self.dropRate>0:
            out=F.dropout(out,p=self.dropRate,inplace=False,training=self.training)
        out=self.avgploing(out)
        return out

class DenseBlock(nn.Module):
    def __init__(self,layer_nums,in_channels,growth_rate,block,dropRate=0.0) -> None:
        super(DenseBlock,self).__init__()
        self.layer=self._make_layer(block,in_channels,growth_rate,layer_nums,dropRate)

    def _make_layer(self,block,in_channels,growth_rate,layer_nums,dropRate):
        layers=[]
        for i in range(layer_nums):
            layers.append(block(in_channels+i*growth_rate,growth_rate,dropRate))
        return nn.Sequential(*layers)

    def forward(self,x):
        return self.layer(x)



class DenseNet(nn.Module):
    def __init__(self,depth,class_nums,growth_rate=12,reduction=0.5,bottleneck=True,dropRate=0.0) -> None:
        super(DenseNet,self).__init__()
        in_plains=2*growth_rate
        nDenseBlock=(depth-4)/3
        if bottleneck==True:
            nDenseBlock/=2
            block=BottleneckBlock
        nDenseBlock=int (nDenseBlock)

        self.conv1=nn.Conv2d(3,in_plains,kernel_size=7,stride=2,padding=3,bias=False)
        self.pool=nn.AvgPool2d(kernel_size=3,stride=2,padding=1)
        #1 block
        self.block1=DenseBlock(nDenseBlock,in_plains,growth_rate,block,dropRate)
        in_plains=int(in_plains+nDenseBlock*growth_rate)
        out_plains=int(math.floor(in_plains*reduction))
        self.trans1=TransitionBlock(in_plains,out_plains,dropRate)
        #2 block
        in_plains=out_plains
        self.block2=DenseBlock(nDenseBlock,in_plains,growth_rate,block,dropRate)
        in_plains=int(in_plains+nDenseBlock*growth_rate)
        out_plains=int(math.floor(in_plains*reduction))
        self.trans2=TransitionBlock(in_plains,out_plains,dropRate)
        #3 block
        in_plains=out_plains
        self.block3=DenseBlock(nDenseBlock,in_plains,growth_rate,block,dropRate)
        in_plains=int(in_plains+nDenseBlock*growth_rate)
        out_plains=int(math.floor(in_plains*reduction))
        self.trans3=TransitionBlock(in_plains,out_plains,dropRate)
        #4 block
        in_plains=out_plains
        self.block4=DenseBlock(nDenseBlock,in_plains,growth_rate,block,dropRate)
        in_plains=int(in_plains+nDenseBlock*growth_rate)
        # # classifier
        self.globgalavgpool=nn.AdaptiveAvgPool2d((1,1))
        self.bn1=nn.BatchNorm2d(in_plains)
        self.relu=nn.ReLU(inplace=True)
        self.fc=nn.Linear(in_plains,class_nums)
        self.in_planes=in_plains
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                m.weight.data.normal_(0,1)
            elif isinstance(m,nn.BatchNorm2d):
               m.weight.data.fill_(1)
               m.bias.data.zero_()
            elif isinstance(m,nn.Linear):
                m.bias.data.zero_()


    def forward(self,x):
        out=self.conv1(x)
        out=self.pool(out)
        out =self.block1(out)
        print(out.size())
        out=self.trans1(out)
        print(out.size())
        out=self.block2(out)
        print(out.size())
        out=self.trans2(out)
        print(out.size())
        out=self.block3(out)
        print(out.size())
        out=self.trans3(out)
        print(out.size())
        out=self.block4(out)
        print(out.size())
        out=self.globgalavgpool(out)
        out=out.view(-1,self.in_planes)
        out=self.fc(out)
        return out




if __name__ =="__main__":
    #input=torch.ones([2,3,224,224])
    model=DenseNet(121,10)
    summary(model.to("cuda"),(3,224,224))
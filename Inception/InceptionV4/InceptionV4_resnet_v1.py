#import tensorrt as trt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchsummary import summary


class BasicConv(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride=(1,1), padding=(0,0),bias=False):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=kernel_size, stride=stride, padding=padding,bias=bias)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

class Stem(nn.Module):
    def __init__(self, in_ch=3):
        super(Stem, self).__init__()
        self.conv1=BasicConv(in_ch,32,kernel_size=3,stride=2,bias=False)
        self.bn1=nn.BatchNorm2d(32)
        self.conv2=BasicConv(32,32,kernel_size=3,bias=False)
        self.bn2=nn.BatchNorm2d(32)
        self.conv3=BasicConv(32,64,kernel_size=3,padding=1,bias=False)
        self.bn3=nn.BatchNorm2d(64)
        self.branch1_1=nn.MaxPool2d(kernel_size=3,stride=2)
        self.branch1_2=BasicConv(64,96,kernel_size=3,stride=2,bias=False)
        
        self.branch2_1=nn.Sequential(
            BasicConv(160,64,kernel_size=1,stride=1,bias=False),
            BasicConv(64,96,kernel_size=3,bias=False)
        )
        self.branch2_2=nn.Sequential(
            BasicConv(160,64,kernel_size=1,stride=1,bias=False),
            BasicConv(64,64,kernel_size=(7,1),padding=(3,0),bias=False),
            BasicConv(64,64,kernel_size=(1,7),padding=(0,3),bias=False),
            BasicConv(64,96,kernel_size=3,bias=False)
        )
        self.branch3_1=nn.MaxPool2d(kernel_size=3,stride=2)
        self.branch3_2=BasicConv(192,192,kernel_size=3,stride=2,bias=False)
        
    def forward(self, x):
        x=self.bn1(self.conv1(x))
        x=self.bn2(self.conv2(x))
        x=self.bn3(self.conv3(x))
        b1=torch.cat([self.branch1_1(x),self.branch1_2(x)],dim=1)
        print(b1.size())
        b2=torch.cat([self.branch2_1(b1),self.branch2_2(b1)],dim=1)
        print(b2.size())
        
        b3=torch.cat([self.branch3_1(b2),self.branch3_2(b2)],dim=1)
        print(b3.size())
        
        return b3


#expand the dim
class ReducationA(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(ReducationA, self).__init__()
        self.branch_1 = nn.Sequential(
            BasicConv(in_channels ,192, kernel_size=1),
            BasicConv(in_channels=192,out_channels=224,kernel_size=3,stride=1),
            BasicConv(in_channels=224,out_channels=256,kernel_size=3,stride=2,padding=1)
        )

        self.branch_2 = nn.Sequential(
            BasicConv(in_channels=in_channels, out_channels=384,kernel_size=3,stride=2)
        )
        self.branch_3 = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        x_1 = self.branch_1(x)
        x_2 = self.branch_2(x)
        x_3 = self.branch_3(x)
        # print('reductionA:')
        # print(x_1.size())
        # print(x_2.size())
        # print(x_3.size())
        x = torch.cat([x_1, x_2, x_3],dim=1)
        return x

#expand the dim
class ReducationB(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(ReducationB, self).__init__()
        self.branch_1 = nn.Sequential(
            BasicConv(in_channels ,out_channels=256, kernel_size=1),
            BasicConv(256,256,kernel_size=(1,7),padding=(0,3),bias=False),
            BasicConv(256,256,kernel_size=(7,1),padding=(3,0),bias=False),        
            BasicConv(in_channels=256,out_channels=320,kernel_size=3,stride=2)
        )

        self.branch_2 = nn.Sequential(
            BasicConv(in_channels ,out_channels=192, kernel_size=1),
            BasicConv(in_channels=192, out_channels=192,kernel_size=3,stride=2)
        )
        self.branch_3 = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        x_1 = self.branch_1(x)
        x_2 = self.branch_2(x)
        x_3 = self.branch_3(x)
        print('reductionB:')
        print(x_1.size())
        print(x_2.size())
        print(x_3.size())
        x = torch.cat([x_1, x_2, x_3],dim=1)
        return x

class InceptionBlockA(nn.Module):
    def __init__(self,in_ch,out_ch=384) -> None:
        super(InceptionBlockA,self).__init__()
        out_ch=out_ch//4
        self.branch1=nn.Sequential(
            nn.Conv2d(in_ch,64,kernel_size=1,bias=False),
            nn.Conv2d(64,96,kernel_size=3,padding=1,bias=False),
            nn.Conv2d(96,out_ch,kernel_size=3,padding=1,bias=False)
        )
        self.branch2=nn.Sequential(
            nn.Conv2d(in_ch,64,kernel_size=1,bias=False),
            nn.Conv2d(64,out_ch,kernel_size=3,padding=1,bias=False),
        )
        self.branch3=nn.Sequential(
            nn.MaxPool2d(kernel_size=3,stride=1,padding=1),
            nn.Conv2d(in_ch,out_ch,kernel_size=1,bias=False)
        )
        self.branch4=nn.Sequential(
            nn.Conv2d(in_ch,out_ch,kernel_size=1,bias=False)
        )

    def forward(self,x):
        out_1=self.branch1(x)
        out_2=self.branch2(x)
        out_3=self.branch3(x)
        out_4=self.branch4(x)
        out=torch.cat([out_1,out_2,out_3,out_4],dim=1)
        return out

class InceptionBlockB(nn.Module):
    def __init__(self,in_ch,out_ch=1024) -> None:
        super(InceptionBlockB,self).__init__()
        out_ch=out_ch//8
        inter_layers1=192
        inter_layers2=224
        self.branch1=nn.Sequential(
            nn.Conv2d(in_ch,inter_layers1,kernel_size=1,bias=False),
            nn.Conv2d(inter_layers1,inter_layers1,kernel_size=(1,7),padding=(0,3),bias=False),
            nn.Conv2d(inter_layers1,inter_layers2,kernel_size=(7,1),padding=(3,0),bias=False),
            nn.Conv2d(inter_layers2,inter_layers2,kernel_size=(1,7),padding=(0,3),bias=False),
            nn.Conv2d(inter_layers2,out_ch*2,kernel_size=(7,1),padding=(3,0),bias=False)
        )
        self.branch2=nn.Sequential(
            nn.Conv2d(in_ch,inter_layers1,kernel_size=1,bias=False),
            nn.Conv2d(inter_layers1,inter_layers2,kernel_size=(1,7),padding=(0,3),bias=False),
            nn.Conv2d(inter_layers2,out_ch*2,kernel_size=(7,1),padding=(3,0),bias=False)
        )
        self.branch3=nn.Sequential(
            nn.MaxPool2d(kernel_size=3,stride=1,padding=1),
            nn.Conv2d(in_ch,out_ch,kernel_size=1,bias=False)
        )
        self.branch4=nn.Conv2d(in_ch,out_ch*3,kernel_size=1,bias=False)
                
    def forward(self,x):
        out_1=self.branch1(x)
        out_2=self.branch2(x)
        out_3=self.branch3(x)
        out_4=self.branch4(x)
        
        print(out_1.size())
        print(out_2.size())
        print(out_3.size())
        print(out_4.size())

        out=torch.cat([out_1,out_2,out_3,out_4],dim=1)
        # print(out.size())
        return out

class InceptionBlockC(nn.Module):
    def __init__(self,in_ch,out_ch=1536) -> None:
        super(InceptionBlockC,self).__init__()
        out_ch=out_ch//6
        self.branch1=nn.Sequential(           
            nn.Conv2d(in_ch,384,kernel_size=1,bias=False),
            nn.Conv2d(384,448,kernel_size=(1,3),padding=(0,1),bias=False),
            nn.Conv2d(448,512,kernel_size=(3,1),padding=(1,0),bias=False),
            )

        self.branch1_1=nn.Conv2d(512,out_ch,kernel_size=(1,3),padding=(0,1),bias=False)
        self.branch1_2=nn.Conv2d(512,out_ch,kernel_size=(3,1),padding=(1,0),bias=False)
       
        self.branch2=nn.Conv2d(in_ch,384,kernel_size=1,bias=False)
        self.branch2_1=nn.Conv2d(384,out_ch,kernel_size=(1,3),padding=(0,1),bias=False)
        self.branch2_2=nn.Conv2d(384,out_ch,kernel_size=(3,1),padding=(1,0),bias=False)

        self.branch3=nn.Sequential(
            nn.MaxPool2d(kernel_size=3,stride=1,padding=1),
            nn.Conv2d(in_ch,out_ch,kernel_size=1,bias=False)
        )
            
        self.branch4=nn.Conv2d(in_ch,out_ch,kernel_size=1,bias=False)
        
    def forward(self,x):       
        out_x1=self.branch1(x)
        out_1=self.branch1_1(out_x1)
        out_2=self.branch1_2(out_x1)

        out_x2=self.branch2(x)
        out_3=self.branch2_1(out_x2)
        out_4=self.branch2_2(out_x2)

        out_5=self.branch3(x)
        out_6=self.branch4(x)
        out=torch.cat([out_1,out_2,out_3,out_4,out_5,out_6],dim=1)
        print(out.size())
        return out


class InceptionV4(nn.Module):
    def __init__(self,class_nums,in_ch=3) -> None:
        super(InceptionV4,self).__init__()
        #classify
        self.stem=Stem(in_ch)
        self.inceptionA1=InceptionBlockA(384,384)
        self.inceptionA2=InceptionBlockA(384,384)
        self.inceptionA3=InceptionBlockA(384,384)
        self.inceptionA4=InceptionBlockA(384,384)
        self.reductionA=ReducationA(384,1024)
        self.inceptionB1=InceptionBlockB(1024,1024)
        self.inceptionB2=InceptionBlockB(1024,1024)
        self.inceptionB3=InceptionBlockB(1024,1024)
        self.inceptionB4=InceptionBlockB(1024,1024)
        self.inceptionB5=InceptionBlockB(1024,1024)
        self.inceptionB6=InceptionBlockB(1024,1024)
        self.inceptionB7=InceptionBlockB(1024,1024)
        self.reductionB=ReducationB(1024,1536)
        self.inceptionC1=InceptionBlockC(1536,1536)
        self.inceptionC2=InceptionBlockC(1536,1536)
        self.inceptionC3=InceptionBlockC(1536,1536)

        self.agvpool=nn.AdaptiveAvgPool2d(output_size=1)
        self.dropout=nn.Dropout2d(p=0.8,inplace=True)
        self.fc=nn.Conv2d(1536,class_nums,kernel_size=1)
        self.softmax=nn.Softmax(dim=1)

    def forward(self,x):
        x=self.stem(x)
        x=self.inceptionA1(x)
        x=self.inceptionA2(x)
        x=self.inceptionA3(x)
        x=self.inceptionA4(x)
        x=self.reductionA(x)
        x=self.inceptionB1(x)
        x=self.inceptionB2(x)
        x=self.inceptionB3(x)
        x=self.inceptionB4(x)
        x=self.inceptionB5(x)
        x=self.inceptionB6(x)
        x=self.inceptionB7(x)
        x=self.reductionB(x)
        x=self.inceptionC1(x)
        x=self.inceptionC2(x)
        x=self.inceptionC3(x)
        x=self.agvpool(x)


        x=self.dropout(x)
        x=self.fc(x)
        x=self.softmax(x)
        print(x.size())

        return x


if __name__ =="__main__":
    input=torch.ones([2,3,224,224])
    model=InceptionV4(10)
    # model=Stem()
    # res=model(input)
    # print(res.shape)
    summary(model.to("cuda"),(3,299,299))
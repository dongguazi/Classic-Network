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

#expand the dim
class Model_Expand(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Model_Expand, self).__init__()
        self.branch_1 = nn.Sequential(
            BasicConv(in_channels=in_channels ,out_channels=out_channels, kernel_size=1),
            BasicConv(in_channels=out_channels,out_channels=out_channels,kernel_size=3,stride=1),
            BasicConv(in_channels=out_channels,out_channels=out_channels,kernel_size=3,stride=2, padding=1)
        )

        self.branch_2 = nn.Sequential(
            BasicConv(in_channels=in_channels, out_channels=out_channels,kernel_size=1),
            BasicConv(in_channels=out_channels, out_channels=out_channels,kernel_size=3,stride=2)
        )
        self.branch_3 = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        x_1 = self.branch_1(x)
        x_2 = self.branch_2(x)
        x_3 = self.branch_3(x)
        # print("Model_Expand, x_1.shape = ",x_1.shape)
        # print("Model_Expand, x_2.shape = ",x_2.shape)
        # print("Model_Expand, x_3.shape = ", x_3.shape)
        x = torch.cat([x_1, x_2, x_3],dim=1)
        return x

class InceptionBlockV1(nn.Module):
    def __init__(self,in_ch,out_maxpool) -> None:
        super(InceptionBlockV1,self).__init__()
        self.branch1=nn.Sequential(
            nn.Conv2d(in_ch,64,kernel_size=1,bias=False),
            nn.Conv2d(64,96,kernel_size=3,padding=1,bias=False),
            nn.Conv2d(96,96,kernel_size=3,padding=1,bias=False)
        )
        self.branch2=nn.Sequential(
            nn.Conv2d(in_ch,48,kernel_size=1,bias=False),
            nn.Conv2d(48,64,kernel_size=3,padding=1,bias=False),
        )
        self.branch3=nn.Sequential(
            nn.MaxPool2d(kernel_size=3,stride=1,padding=1),
            nn.Conv2d(in_ch,out_maxpool,kernel_size=1,bias=False)
        )
        self.branch4=nn.Sequential(
            nn.Conv2d(in_ch,64,kernel_size=1,bias=False)
        )

    def forward(self,x):
        out_1=self.branch1(x)
        out_2=self.branch2(x)
        out_3=self.branch3(x)
        out_4=self.branch4(x)
        out=torch.cat([out_1,out_2,out_3,out_4],dim=1)
        return out


class InceptionBlockV2(nn.Module):
    def __init__(self,in_ch,out_ch) -> None:
        super(InceptionBlockV2,self).__init__()
        out_ch=out_ch//4
        self.branch1=nn.Sequential(
            nn.Conv2d(in_ch,out_ch,kernel_size=1,bias=False),
            nn.Conv2d(out_ch,out_ch,kernel_size=(1,7),padding=(0,3),bias=False),
            nn.Conv2d(out_ch,out_ch,kernel_size=(7,1),padding=(3,0),bias=False),
            nn.Conv2d(out_ch,out_ch,kernel_size=(1,7),padding=(0,3),bias=False),
            nn.Conv2d(out_ch,out_ch,kernel_size=(7,1),padding=(3,0),bias=False)
        )
        self.branch2=nn.Sequential(
            nn.Conv2d(in_ch,out_ch,kernel_size=1,bias=False),
            nn.Conv2d(out_ch,out_ch,kernel_size=(1,7),padding=(0,3),bias=False),
            nn.Conv2d(out_ch,out_ch,kernel_size=(7,1),padding=(3,0),bias=False)
        )
        self.branch3=nn.Sequential(
            nn.MaxPool2d(kernel_size=3,stride=1,padding=1),
            nn.Conv2d(in_ch,out_ch,kernel_size=1,bias=False)
        )
        self.branch4=nn.Conv2d(in_ch,out_ch,kernel_size=1,bias=False)
                
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

class InceptionBlockV3(nn.Module):
    def __init__(self,in_ch,out_ch) -> None:
        super(InceptionBlockV3,self).__init__()
        out_ch=out_ch//6
        self.branch1=nn.Sequential(           
            nn.Conv2d(in_ch,out_ch,kernel_size=1,bias=False),
            nn.Conv2d(out_ch,out_ch,kernel_size=3,padding=1,bias=False)
            )

        self.branch1_1=nn.Conv2d(out_ch,out_ch,kernel_size=(1,3),padding=(0,1),bias=False)
        self.branch1_2=nn.Conv2d(out_ch,out_ch,kernel_size=(3,1),padding=(1,0),bias=False)
       
        self.branch2=nn.Conv2d(in_ch,out_ch,kernel_size=1,bias=False)
        self.branch2_1=nn.Conv2d(out_ch,out_ch,kernel_size=(1,3),padding=(0,1),bias=False)
        self.branch2_2=nn.Conv2d(out_ch,out_ch,kernel_size=(3,1),padding=(1,0),bias=False)

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
        return nn.Sequential()

class InceptionV1(nn.Module):
    def __init__(self,class_nums,in_ch=3) -> None:
        super(InceptionV1,self).__init__()
        #input        
        self.conv1=BasicConv(in_ch,32,kernel_size=3, stride=2)
        self.conv2=BasicConv(32,32,kernel_size=3, stride=1)
        self.conv3=BasicConv(32,64,kernel_size=3, stride=1,padding=1)
        self.pool1=nn.MaxPool2d(kernel_size=3,stride=2)
        self.conv4=BasicConv(64,80,kernel_size=3,stride=1)
        self.conv5=BasicConv(80,192,kernel_size=3,stride=2)
        self.conv6=BasicConv(192,288,kernel_size=3,stride=1,padding=1)
        #block1
        self.InceptionA1=InceptionBlockV1(in_ch=288,out_maxpool=64)
        self.InceptionA2=InceptionBlockV1(in_ch=288,out_maxpool=64)
        self.InceptionA3=InceptionBlockV1(in_ch=288,out_maxpool=64)
        self.Expand1=Model_Expand(288,240)
        #block2
        self.InceptionB1=InceptionBlockV2(in_ch=768,out_ch=768)
        self.InceptionB2=InceptionBlockV2(in_ch=768,out_ch=768)
        self.InceptionB3=InceptionBlockV2(in_ch=768,out_ch=768)
        self.InceptionB4=InceptionBlockV2(in_ch=768,out_ch=768)
        self.InceptionB5=InceptionBlockV2(in_ch=768,out_ch=768)
        self.Expand2=Model_Expand(768,256)
        #block3
        self.InceptionC1=InceptionBlockV2(in_ch=1280,out_ch=1280)
        self.InceptionC2=InceptionBlockV2(in_ch=1280,out_ch=2048)
        #classify
        self.pool2=nn.AdaptiveAvgPool2d(output_size=1)
        self.dropout=nn.Dropout2d(p=0.4,inplace=True)
        self.fc=nn.Conv2d(2048,class_nums,kernel_size=1)
        self.softmax=nn.Softmax(dim=1)

    def forward(self,x):
        out=self.conv1(x)
        out=self.conv2(out)
        out=self.conv3(out)
        out=self.pool1(out)
        out=self.conv4(out)
        out=self.conv5(out)
        out=self.conv6(out)
        out=self.InceptionA1(out)
        out=self.InceptionA2(out)
        out=self.InceptionA3(out)
        out=self.Expand1(out)
        out=self.InceptionB1(out)
        out=self.InceptionB2(out)
        out=self.InceptionB3(out)
        out=self.InceptionB4(out)
        out=self.InceptionB5(out)
        out=self.Expand2(out)
        out=self.InceptionC1(out)
        out=self.InceptionC2(out)
        out=self.pool2(out)

        out=self.dropout(out)
        out=self.fc(out)
        out=self.softmax(out)
        return out


if __name__ =="__main__":
    input=torch.ones([2,3,224,224])
    model=InceptionV1(10)
    # res=model(input)
    # print(res.shape)
    summary(model.to("cuda"),(3,299,299))
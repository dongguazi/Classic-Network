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
        self.conv1=nn.Conv2d(in_ch,32,kernel_size=3,stride=2,bias=False)
        self.conv2=nn.Conv2d(32,32,kernel_size=3,bias=False)
        self.conv3=nn.Conv2d(32,64,kernel_size=3,padding=1,bias=False)
        self.maxpool1=nn.MaxPool2d(kernel_size=3,stride=2)

        self.conv4=nn.Conv2d(64,80,kernel_size=1,bias=False)
        self.conv5=nn.Conv2d(80,192,kernel_size=3,bias=False)
        self.conv6=nn.Conv2d(192,256,kernel_size=3,stride=2,bias=False)
        
    def forward(self, x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.maxpool1(x)
        x=self.conv4(x)
        x=self.conv5(x)
        x=self.conv6(x)      
        return x


#expand the dim
class ReducationA(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(ReducationA, self).__init__()
        self.branch_1 = nn.Sequential(
            nn.Conv2d(in_channels ,192, kernel_size=1),
            nn.Conv2d(in_channels=192,out_channels=224,kernel_size=3,stride=1),
            nn.Conv2d(in_channels=224,out_channels=256,kernel_size=3,stride=2,padding=1)
        )

        self.branch_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=384,kernel_size=3,stride=2)
        )
        self.branch_3 = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        x_1 = self.branch_1(x)
        x_2 = self.branch_2(x)
        x_3 = self.branch_3(x)
        print('reductionA:')
        print(x_1.size())
        print(x_2.size())
        print(x_3.size())
        x = torch.cat([x_1, x_2, x_3],dim=1)
        return x

#expand the dim
class ReducationB(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(ReducationB, self).__init__()
        self.branch_1 = nn.Sequential(
            nn.Conv2d(in_channels ,out_channels=256, kernel_size=1),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1),        
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=2,padding=1)
        )

        self.branch_2 = nn.Sequential(
            nn.Conv2d(in_channels ,256, kernel_size=1),
            nn.Conv2d(256, 256,kernel_size=3,stride=2)
        )
        self.branch_3 = nn.Sequential(
            nn.Conv2d(in_channels ,256, kernel_size=1),
            nn.Conv2d(256, 384,kernel_size=3,stride=2)
        )
        self.branch_4 = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        x_1 = self.branch_1(x)
        x_2 = self.branch_2(x)
        x_3 = self.branch_3(x)
        x_4 = self.branch_4(x)

        # print('reductionB:')
        # print(x_1.size())
        # print(x_2.size())
        # print(x_3.size())
        x = torch.cat([x_1, x_2, x_3,x_4],dim=1)
        return x

class InceptionBlockA(nn.Module):
    def __init__(self,in_ch,out_ch=384) -> None:
        super(InceptionBlockA,self).__init__()
        out_ch=out_ch//4
        self.branch1=nn.Sequential(
            nn.Conv2d(in_ch,32,kernel_size=1,bias=False),
            nn.Conv2d(32,32,kernel_size=3,padding=1,bias=False),
            nn.Conv2d(32,32,kernel_size=3,padding=1,bias=False)
        )
        self.branch2=nn.Sequential(
            nn.Conv2d(in_ch,32,kernel_size=1,bias=False),
            nn.Conv2d(32,32,kernel_size=3,padding=1,bias=False),
        )
        self.branch3=nn.Sequential(
            nn.Conv2d(in_ch,32,kernel_size=1,bias=False)
        )        
        self.conv=nn.Conv2d(96,256,kernel_size=3,padding=1,bias=False)
        

    def forward(self,x):
        out_1=self.branch1(x)
        out_2=self.branch2(x)
        out_3=self.branch3(x)
        # print("blockA:")
        # print(out_1.size())
        # print(out_2.size())
        # print(out_3.size())

        out=torch.cat([out_1,out_2,out_3],dim=1)
        out=self.conv(out)
        out=out+x
        return out

class InceptionBlockB(nn.Module):
    def __init__(self,in_ch,out_ch=1024) -> None:
        super(InceptionBlockB,self).__init__()
        
        self.branch1=nn.Sequential(
            nn.Conv2d(in_ch,128,kernel_size=1,bias=False),
            nn.Conv2d(128,128,kernel_size=(1,7),padding=(0,3),bias=False),
            nn.Conv2d(128,128,kernel_size=(7,1),padding=(3,0),bias=False)
        )
        self.branch2=nn.Sequential(
            nn.Conv2d(in_ch,128,kernel_size=1,bias=False),
        )

        self.conv=nn.Conv2d(256,896,kernel_size=1,bias=False)
                
    def forward(self,x):
        out_1=self.branch1(x)
        out_2=self.branch2(x)
        out=torch.cat([out_1,out_2],dim=1)
        out=self.conv(out)
        out=out+x      
        # print(out_1.size())
        # print(out_2.size())
        # print(out.size())
        return out

class InceptionBlockC(nn.Module):
    def __init__(self,in_ch,out_ch=1536) -> None:
        super(InceptionBlockC,self).__init__()
        out_ch=out_ch//6
        self.branch1=nn.Sequential(           
            nn.Conv2d(in_ch,192,kernel_size=1,bias=False),
            nn.Conv2d(192,192,kernel_size=(1,3),padding=(0,1),bias=False),
            nn.Conv2d(192,192,kernel_size=(3,1),padding=(1,0),bias=False),
            )

        self.branch2=nn.Conv2d(in_ch,192,kernel_size=1,bias=False)
  
        self.conv=nn.Conv2d(384,1792,kernel_size=1,bias=False)
             
    def forward(self,x):       
        out_1=self.branch1(x)
        out_2=self.branch2(x)
        out=torch.cat([out_1,out_2],dim=1)
        out=self.conv(out)
        out=out+x
        # print(out.size())
        return out


class Inception_resnet_V1(nn.Module):
    def __init__(self,class_nums,in_ch=3) -> None:
        super(Inception_resnet_V1,self).__init__()
        #features
        self.stem=Stem(in_ch)
        self.relu=nn.ReLU(inplace=True)
        self.inceptionA1=InceptionBlockA(256,256)
        self.inceptionA2=InceptionBlockA(256,256)
        self.inceptionA3=InceptionBlockA(256,256)
        self.inceptionA4=InceptionBlockA(256,256)
        self.inceptionA5=InceptionBlockA(256,256)

        self.reductionA=ReducationA(256,896)
         
        self.inceptionB1=InceptionBlockB(896,896)
        self.inceptionB2=InceptionBlockB(896,896)
        self.inceptionB3=InceptionBlockB(896,896)
        self.inceptionB4=InceptionBlockB(896,896)
        self.inceptionB5=InceptionBlockB(896,896)
        self.inceptionB6=InceptionBlockB(896,896)
        self.inceptionB7=InceptionBlockB(896,896)
        self.inceptionB8=InceptionBlockB(896,896)
        self.inceptionB9=InceptionBlockB(896,896)
        self.inceptionB10=InceptionBlockB(896,896)

        self.reductionB=ReducationB(896,1792)

        self.inceptionC1=InceptionBlockC(1792,1792)
        self.inceptionC2=InceptionBlockC(1792,1792)
        self.inceptionC3=InceptionBlockC(1792,1792)
        self.inceptionC4=InceptionBlockC(1792,1792)
        self.inceptionC5=InceptionBlockC(1792,1792)
        #classify
        self.agvpool=nn.AdaptiveAvgPool2d(output_size=1)
        self.dropout=nn.Dropout2d(p=0.8,inplace=True)
        self.fc=nn.Conv2d(1792,class_nums,kernel_size=1)
        self.softmax=nn.Softmax(dim=1)

    def forward(self,x):
        x=self.stem(x)
        x=self.relu(self.inceptionA1(x))
        x=self.relu(self.inceptionA2(x))
        x=self.relu(self.inceptionA3(x))
        x=self.relu(self.inceptionA4(x))
        x=self.relu(self.inceptionA5(x))

        x=self.reductionA(x)

        x=self.relu(self.inceptionB1(x))
        x=self.relu(self.inceptionB2(x))
        x=self.relu(self.inceptionB3(x))
        x=self.relu(self.inceptionB4(x))
        x=self.relu(self.inceptionB5(x))
        x=self.relu(self.inceptionB6(x))
        x=self.relu(self.inceptionB7(x))
        x=self.relu(self.inceptionB8(x))
        x=self.relu(self.inceptionB9(x))
        x=self.relu(self.inceptionB10(x))
        print(x.size())

        x=self.reductionB(x)
        print(x.size())

        x=self.relu(self.inceptionC1(x))
        x=self.relu(self.inceptionC2(x))
        x=self.relu(self.inceptionC3(x))
        x=self.relu(self.inceptionC4(x))
        x=self.relu(self.inceptionC5(x))

        x=self.agvpool(x)

        x=self.dropout(x)
        x=self.fc(x)
        x=self.softmax(x)

        return x


if __name__ =="__main__":
    input=torch.ones([2,3,224,224])
    model=Inception_resnet_V1(10)
    # model=InceptionBlockA(256)
    # model=Stem()
    # res=model(input)
    # print(res.shape)
    summary(model.to("cuda"),(3,299,299))
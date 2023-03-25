#import tensorrt as trt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchsummary import summary

class InceptionBlockV1(nn.Module):
    def __init__(self,in_ch,out_ch,reduce_1x1,reduce_3x3,out_3x3,reduce_5x5,out_5x5,out_maxpool) -> None:
        super(InceptionBlockV1,self).__init__()
        self.conv1=nn.Conv2d(in_ch,reduce_1x1,kernel_size=1,bias=False)
        self.conv2_1=nn.Conv2d(in_ch,reduce_3x3,kernel_size=1,bias=False)
        self.conv2_2=nn.Conv2d(reduce_3x3,out_3x3,kernel_size=3,padding=1,bias=False)
        self.conv3_1=nn.Conv2d(in_ch,reduce_5x5,kernel_size=1,bias=False)
        self.conv3_2=nn.Conv2d(reduce_5x5,out_5x5,kernel_size=5,padding=2,bias=False)
        self.conv4_1=nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        self.conv4_2=nn.Conv2d(in_ch,out_maxpool,kernel_size=5,padding=2,bias=False)
    def forward(self,x):
        branch1=self.conv1(x)
        branch2=self.conv2_2(self.conv2_1(x))
        branch3=self.conv3_2(self.conv3_1(x))
        branch4=self.conv4_2(self.conv4_1(x))
        out=torch.cat([branch1,branch2,branch3,branch4],dim=1)
        print(out.size())
        return out

class InceptionV1(nn.Module):
    def __init__(self,class_nums,in_ch=3) -> None:
        super(InceptionV1,self).__init__()
        self.conv1=nn.Conv2d(in_ch,64,kernel_size=7, stride=2,padding=3,bias=False)
        self.maxpool1=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.conv2=nn.Conv2d(64,192,kernel_size=3,stride=1,padding=1,bias=False)
        self.maxpool2=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.inception_3a=InceptionBlockV1(192,256,64,96,128,16,32,32)
        self.inception_3b=InceptionBlockV1(256,480,128,128,192,32,96,64)
        self.maxpool3=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.inception_4a=InceptionBlockV1(480,512,160,112,224,24,64,64)
        self.inception_4b=InceptionBlockV1(512,512,160,112,224,24,64,64)
        self.inception_4c=InceptionBlockV1(512,512,128,128,256,24,64,64)
        self.inception_4d=InceptionBlockV1(512,528,112,144,288,32,64,64)
        self.inception_4e=InceptionBlockV1(528,832,256,160,320,32,128,128)
        self.maxpool4=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.inception_5a=InceptionBlockV1(832,832,256,160,320,32,128,128)
        self.inception_5b=InceptionBlockV1(832,1024,384,192,384,48,128,128)
        self.avgpool=nn.AvgPool2d(kernel_size=7)
        self.dropout=nn.Dropout2d(p=0.4,inplace=True)
        self.fc=nn.Conv2d(1024,class_nums,kernel_size=1)
        self.softmax=nn.Softmax(dim=1)

    def forward(self,x):
        out=self.maxpool1(self.conv1(x))
        out=self.maxpool2(self.conv2(out))
        out=self.inception_3a(out)
        out=self.inception_3b(out)
        out=self.maxpool3(out)
        out=self.inception_4a(out)
        out=self.inception_4b(out)
        out=self.inception_4c(out)
        out=self.inception_4d(out)
        out=self.inception_4e(out)
        out=self.maxpool4(out)
        out=self.inception_5a(out)
        out=self.inception_5b(out)
        out=self.dropout(out)
        out=self.avgpool(out)
        out=self.dropout(out)
        # print(out.size())
        out=self.fc(out)
        out=self.softmax(out)
        return out

if __name__ =="__main__":
    input=torch.ones([2,3,224,224])
    # model=InceptionBlock(192,256,64,96,128,16,32,32)
    model=InceptionV1(10)
    # res=model(input)
    # print(res.shape)
    summary(model.to("cuda"),(3,224,224))
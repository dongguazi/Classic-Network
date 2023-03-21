import tensorrt as trt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchsummary import summary


def _make_divisible(v, divisor, min_value=None):
    """
    note:come from other people

    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
class Inverted_Residual_Block(nn.Module):

    def __init__(self,in_ch,out_ch,stride,width_ratio):
        super(Inverted_Residual_Block,self).__init__()
        self.stride=stride
        hidden_ch=int(round(width_ratio*in_ch))
        
        self.connect=stride==1 & in_ch==out_ch
        self.conv1=nn.Conv2d(in_ch,hidden_ch,kernel_size=1),
        self.conv2_dw=conv_bn(hidden_ch,hidden_ch,stride=stride,groups=hidden_ch)
        self.conv3=nn.Conv2d(hidden_ch,out_ch,kernel_size=1,stride=1,bias=False)
        self.relu6= nn.ReLU6(inplace=True)

    def forward(self,x):
        indentity=x
        out=self.relu6(self.conv1(x))
        out=self.relu6(self.conv2(x))
        out=self.conv3(out)
        if self.connect:
            out+=x

        return out

class conv_bn(nn.Module):
    def __init__(self,in_ch,out_ch,kernel_size,stride,groups) -> None:
        super().__init__()
        self.padding=stride//2
        self.conv1=nn.Conv2d(in_ch,out_ch
        ,kernel_size,stride,self.padding,groups=groups, bias=False)
        self.bn1=nn.BatchNorm2d(out_ch)
        self.relu=nn.ReLU6(inplace=True)
    def forward(self,x):
        x=self.relu(self.bn1(self.conv1(x)))
        return x

class MobileNetV2(nn.Module):
    
    def __init__(self,num_classes,width_ratio,settingList,round_nearest=8) -> None:
        super(MobileNetV2,self).__init__()
        in_ch=32
        out_ch=1280
        if settingList is None:
            settingList = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]
        if len(settingList) == 0 or len(settingList[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(settingList))
        in_ch=_make_divisible(in_ch*width_ratio,round_nearest)
        out_ch=_make_divisible(out_ch*width_ratio,round_nearest)

        self.conv1=conv_bn(in_ch=3,out_ch=in_ch,stride=2)
        for t,c,n,s in settingList:
            out_ch=_make_divisible(c*width_ratio,round_nearest)
            for i in range(n):
                



        self.agvpool=nn.AvgPool2d(7)
        self.fc=nn.Conv2d(1024,num_classes,kernel_size=1,bias=False)
        self.flatten=nn.Flatten()
        self.softmax=nn.Softmax(dim=-1)

    def forward(self,x):
        x=self.conv1(x)
        x=self.agvpool(x)
        # x=torch.reshape(x.shape)
        x=self.fc(x)
        x=self.flatten(x)
        x=self.softmax(x)    
        return x


if __name__ =="__main__":
    input=torch.ones([2,3,224,224])
    model=MobileNetV2(10)
    # res=model(input)
    # print(res.shape)
    summary(model.to("cuda"),(3,224,224))

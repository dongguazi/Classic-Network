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
        #倒残差网络结构两个要点：详见MobileNetV2—_block图
        # 第一个：1x1conv升维，3x3conv Dwise，1x1conv降维
        # 第二个：判断res连接的条件是stride==1 and in_ch==out_ch

        hidden_ch=int(round(width_ratio*in_ch))        
        self.res_connect= stride==1 and in_ch==out_ch
        layers=[]

        # if width_ratio!=1:
        layers.append(conv_BNRelu(in_ch,hidden_ch,kernel_size=1))
        layers.extend(
            [conv_BNRelu(hidden_ch,hidden_ch,kernel_size=3, stride=stride,groups=hidden_ch),
        nn.Conv2d(hidden_ch,out_ch,kernel_size=1,stride=1,padding=0,bias=False),
        nn.BatchNorm2d(out_ch),]
        )
        self.conv=nn.Sequential(*layers)
        # self.relu6= nn.ReLU6(inplace=True)

    def forward(self,x):
        if self.res_connect:
            return x+self.conv(x)
        else:
            return self.conv(x)

class conv_BNRelu(nn.Module):
    def __init__(self,in_ch,out_ch,kernel_size=3,stride=1,groups=1) -> None:
        super(conv_BNRelu,self).__init__()
        self.padding=(kernel_size-1)//2
        self.conv=nn.Conv2d(in_ch,out_ch
        ,kernel_size,stride,self.padding,groups=groups, bias=False)
        self.bn=nn.BatchNorm2d(out_ch)
        self.relu=nn.ReLU6(inplace=True)
    def forward(self,x):
        x=self.relu(self.bn(self.conv(x)))
        return x

class MobileNetV2(nn.Module):
    
    def __init__(self,num_classes=2,width_ratio=1.0,settingList=None,round_nearest=8) -> None:
        super(MobileNetV2,self).__init__()
        in_ch=32
        last_ch=1280
        #width_ratio是倒残差网络宽度的倍率
        # 按照论文的设置t, c, n, s
        if settingList is None:
            #e：expand size；c：out size；
            #se：0(无)1(有se),nl:0(relu)1(Hswish);s:stride
            settingList = [
                # e, c, se, nl，s
                [16, 16, 1, 0, 2],
                [72, 24, 0, 0, 2],
                [88, 24, 0, 0, 1],
                [96, 40, 1, 1, 2],
                [240, 40, 1, 1, 1],
                [240, 40, 1, 1, 1],
                [120, 48, 1, 1, 1],
                [144, 48, 1, 1, 1],
                [288, 96, 1, 1, 2],
                [576, 96, 1, 1, 1],
                [576, 96, 1, 1, 1],


                [6, 320, 1, 1],
            ]
        if len(settingList) == 0 or len(settingList[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(settingList))
        in_ch=_make_divisible(in_ch*width_ratio,round_nearest)
        self.last_ch=_make_divisible(last_ch*max(1,width_ratio),round_nearest)

        layers=[]
        layers.append(conv_BNRelu(in_ch=3,out_ch=in_ch,kernel_size=3,stride=2))
        for t,c,n,s in settingList:
            out_ch=_make_divisible(c*width_ratio,round_nearest)
            for i in range(n):
                stride=s if i==0 else 1
                layers.append(Inverted_Residual_Block(in_ch,out_ch,stride,width_ratio))
                in_ch=out_ch
        layers.append(conv_BNRelu(in_ch,self.last_ch,kernel_size=1,stride=1))
       
        self.features=nn.Sequential(*layers)
        self.classify=nn.Sequential(
            nn.AvgPool2d(7),
            nn.Conv2d(self.last_ch,num_classes,kernel_size=1,bias=False),
            nn.Flatten(),nn.Softmax(dim=-1)

        )
        # self.agvpool=
        # self.fc=
        # self.flatten=
        # self.softmax=
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    def forward(self,x):
        x=self.features(x)
        x=self.classify(x)  
        return x


if __name__ =="__main__":
    input=torch.ones([2,3,224,224])
    model=MobileNetV2(10)
    # res=model(input)
    # print(res.shape)
    summary(model.to("cuda"),(3,224,224))

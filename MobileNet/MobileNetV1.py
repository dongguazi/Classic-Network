import tensorrt as trt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchsummary import summary

class Bottleneck(nn.Module):

    def __init__(self,in_channels,out_channels,down_sample=None,stride=1):
        super(Bottleneck,self).__init__()
    
    def forward(self,x):

        return 0


class MobileNet(nn.Module):
    first=False
    cout=1
    def __init__(self,Resblock,layer_list,num_classes,input_channels,complex=True) -> None:
        super(MobileNet,self).__init__()
       

    def forward(self,x):

       
        return 0


if __name__ =="__main__":
    input=torch.ones([2,3,224,224])
    model=MobileNet(10)
    # res=model(input)
    # print(res.shape)
    summary(model.to("cuda"),(3,224,224))

#import tensorrt as trt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchsummary import summary

  # SeNet get form ResNext :only modify the SELayer,other is the same . 
class FasterNet(nn.Module):
    def __init__(self) -> None:
        super(FasterNet,self).__init__()
        pass
    def init_weights(self):
        pass

    def forward(self,x):
        pass





 

def SeNet101(num_classes,channels=3):
    layer_list=[3,4,23,3]
    return  SeNet(Bottleneck,layer_list,num_classes,channels)  


if __name__ =="__main__":
    input=torch.ones([2,3,224,224])
    model=SeNet152(10)
    # res=model(input)
    # print(res.shape)
    summary(model.to("cuda"),(3,224,224))

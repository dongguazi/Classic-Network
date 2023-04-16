
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchsummary import summary
import math

#paper:https://arxiv.org/pdf/2210.02093.pdf
#该网络是目标检测Neck的组件，适用于小目标检测，用在第一个上采样结构之后。


if __name__ =="__main__":
    #input=torch.ones([2,3,224,224])
    # model=DenseNet(121,10)
    # summary(model.to("cuda"),(3,224,224))
    pass
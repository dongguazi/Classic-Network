import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchsummary import summary
import resnet 
from cuda import cudart  # 使用 cuda runtime API
import numpy as np
import os
import tensorrt as trt


if __name__ =="__main__":
    input=torch.ones([2,3,224,224])
    model=resnet.Resnet18(10)
    # res=model(input)
    # print(res.shape)
    summary(model.to("cuda"),(3,224,224))
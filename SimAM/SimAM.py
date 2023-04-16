import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
from torchsummary import summary
#paper:http://proceedings.mlr.press/v139/yang21o/yang21o.pdf

class SimAM(torch.nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        var = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        # print(var.size()) #torch.Size([3, 64, 7, 7])
        # print(x.mean(dim=[2, 3], keepdim=True).size()) #torch.Size([3, 64, 1, 1])
        mean=var.sum(dim=[2, 3], keepdim=True) / n
        y = var / (4 * (mean + self.e_lambda)) + 0.5      
        return x * self.activaton(y)

if __name__=="__main__":
    input=torch.ones([3,64,7,7])
    model=SimAM()
    output=model(input)
    print(output.shape)
    # summary(model.to("cuda"),(64,7,7))
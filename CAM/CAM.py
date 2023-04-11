import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchsummary import summary
#paper:https://openreview.net/pdf?id=q2ZaVU6bEsT
class CAM(nn.Module):
    def __init__(self, inc, fusion='weight'):
        super().__init__()
        
        assert fusion in ['weight', 'adaptive', 'concat']
        self.fusion = fusion
        
        self.conv1 = nn.Conv(inc, inc, 3, 1, None, 1, 1)
        self.conv2 = nn.Conv(inc, inc, 3, 1, None, 1, 3)
        self.conv3 = nn.Conv(inc, inc, 3, 1, None, 1, 5)
        
        self.fusion_1 = nn.Conv(inc, inc, 1)
        self.fusion_2 = nn.Conv(inc, inc, 1)
        self.fusion_3 = nn.Conv(inc, inc, 1)

        if self.fusion == 'adaptive':
            self.fusion_4 = nn.Conv(inc * 3, 3, 1)
    
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        
        if self.fusion == 'weight':
            return self.fusion_1(x1) + self.fusion_2(x2) + self.fusion_3(x3)
        elif self.fusion == 'adaptive':
            fusion = torch.softmax(self.fusion_4(torch.cat([self.fusion_1(x1), self.fusion_2(x2), self.fusion_3(x3)], dim=1)), dim=1)
            x1_weight, x2_weight, x3_weight = torch.split(fusion, [1, 1, 1], dim=1)
            return x1 * x1_weight + x2 * x2_weight + x3 * x3_weight
        else:
            return torch.cat([self.fusion_1(x1), self.fusion_2(x2), self.fusion_3(x3)], dim=1)

if __name__=="__main__":
    # input=torch.ones([2,3,224,224])
    model=CAM(10)
    summary(model.to("cuda"),(3,299,299))
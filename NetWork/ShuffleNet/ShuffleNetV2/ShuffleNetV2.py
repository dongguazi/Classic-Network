import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchsummary import summary


class ShufflenetV2Block(nn.Module):
    def __init__(self,in_ch,out_ch,kernel_size,stride,groups) -> None:
        super(ShufflenetV2Block,self).__init__()
        assert stride in [1,2]
        self.stride=stride        
        self.mid_ch=out_ch//2
        pad=kernel_size//2
        out_ch=out_ch-in_ch
        self.groups=groups

        self.branch1=nn.Sequential(
                nn.Conv2d(in_ch if stride==2 else self.mid_ch ,self.mid_ch,kernel_size=1,stride=1,bias=False),
                nn.BatchNorm2d(self.mid_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.mid_ch,self.mid_ch,kernel_size=3,stride=self.stride,padding=pad,groups=self.mid_ch,bias=False),
                nn.BatchNorm2d(self.mid_ch),
                nn.Conv2d(self.mid_ch,self.mid_ch,kernel_size=1,stride=1,bias=False),
                nn.BatchNorm2d(self.mid_ch),
                nn.ReLU(inplace=True)
            )
         
        if stride==2:
            self.branch2=nn.Sequential(
                nn.Conv2d(in_ch,in_ch,kernel_size=3,stride=self.stride,padding=pad,groups=in_ch,bias=False),
                nn.BatchNorm2d(in_ch),
                nn.Conv2d(in_ch,self.mid_ch,kernel_size=1,stride=1,bias=False),
                nn.BatchNorm2d(self.mid_ch),
                nn.ReLU(inplace=True)
            )
        else :
            self.branch2=None   

    def forward(self,x):
        if self.stride==1:
            # x1,x2=self.channelSplit(x)
            x1,x2=torch.chunk(x,2,dim=1)
            print(x1.shape)
            print(x2.shape)
            x2=self.branch1(x2)
            print(x2.shape)
            out=torch.cat((x1,x2),dim=1)

        if self.stride==2:
            x1=self.branch1(x)
            x2=self.branch2(x)
            out=torch.cat((x1,x2),dim=1)
       
        out=self.channelShuffle(out)
        print(out.shape)
        return out
    #no use
    def channelSplit(self,x):
        bs,channelNums,H,W=x.shape
        groups_ch=channelNums//2
        x=torch.reshape(x,(bs*groups_ch,2,H*W))
        x=torch.transpose(x,0,1)
        x=torch.reshape(x,(2,-1,groups_ch,H,W))
        return x[0],x[1]

    def channelShuffle(self,x):
        bs,channelNums,H,W=x.shape
        groups_ch=channelNums//self.groups
        x=torch.reshape(x,(bs,groups_ch,self.groups,H,W))
        x=torch.transpose(x,1,2)
        x=torch.reshape(x,(bs,channelNums,H,W))
        return x

class ShufflenetV2(nn.Module):
    def __init__(self,class_nums,in_ch=3,model_size='1.0x',groups=2) -> None:
        super(ShufflenetV2,self).__init__()
        self.stage_repeats=[4,8,4]
        self.mode_size=model_size
        #there are many other cases,and we ignore them,you can build them by yourself.
        if model_size=='0.5x':
            self.stage_out_ch=[-1,24,48,96,192,1024]
        elif model_size=='1.0x':
            self.stage_out_ch=[-1,24,116,232,464,1024]
        elif model_size=='1.5x':
            self.stage_out_ch=[-1,24,176,352,704,1024]
        elif model_size=='2.0x':
            self.stage_out_ch=[-1,24,244,488,976,2048]
        else:
            raise NotImplementedError
       
        input_ch=self.stage_out_ch[1]
        self.input_layer=nn.Sequential(
            nn.Conv2d(in_ch,input_ch,kernel_size=3,stride=2,padding=3//2, bias=False,groups=3),
            nn.BatchNorm2d(input_ch),
            nn.ReLU(inplace=True)
        )
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=3//2)
        features=[]
        #you can build them one by one
        for inx,repeateNums in enumerate(self.stage_repeats):
            out_ch=self.stage_out_ch[inx+2]
            for i in range(repeateNums):
                if i==0:
                    features.append(ShufflenetV2Block(input_ch,out_ch,kernel_size=3, stride=2,groups=groups))
                else :
                    features.append(ShufflenetV2Block(input_ch,out_ch,kernel_size=3, stride=1,groups=groups))
                input_ch=out_ch
        self.features=nn.Sequential(*features)  
        self.classify=nn.Sequential(
            nn.Conv2d(input_ch,self.stage_out_ch[-1],kernel_size=1, bias=False) ,
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.stage_out_ch[-1],class_nums,kernel_size=1, bias=False) 
        )
        

    def forward(self,x):
        x=self.input_layer(x)
        x=self.maxpool(x)
        x=self.features(x)
        x=self.classify(x)

if __name__=="__main__":
    input=torch.ones([2,3,224,224])
    model=ShufflenetV2(10)
    # model=ShufflenetV2Block(24,116,kernel_size=3, stride=2,groups=2)
    # model=ShufflenetV2Block(116,116,kernel_size=3, stride=1,groups=2)

    # res=model(input)
    # print(res.shape)
    summary(model.to("cuda"),(3,224,224))
    # summary(model.to("cuda"),(24,56,56))
    # summary(model.to("cuda"),(116,28,28))



本文提出了卷积块注意模块（Convolutional Block Attention Module，CBAM）， 一个简单而有效的前馈卷积注意力模块神经网络。给定一个中间特征图，本论文的模块设置为沿两个独立的维度按顺序推断注意力图：通道和空间，然后将注意力图与输入特征相乘用于自适应特征细化的映射。 因为CBAM是一种轻量级且 通用模块，它可以集成到任何CNN架构中较少的开销可以忽略不计，并且可以进行端到端的训练基础CNN。也就是说，加了模块之后依旧可以使用预训练模型进行refine。

CBAM可以通过通道注意力和空间注意力机制对模型的精度进行有效的提高。
在网络结构中有效的使用CBAM可以有效的提高结果精度。

CBAMBlock结果：
  Layer (type)               Output Shape         Param #
================================================================
 AdaptiveMaxPool2d-1            [-1, 512, 1, 1]               0
 AdaptiveAvgPool2d-2            [-1, 512, 1, 1]               0
            Conv2d-3             [-1, 32, 1, 1]          16,384
              ReLU-4             [-1, 32, 1, 1]               0
            Conv2d-5            [-1, 512, 1, 1]          16,384
            Conv2d-6             [-1, 32, 1, 1]          16,384
              ReLU-7             [-1, 32, 1, 1]               0
            Conv2d-8            [-1, 512, 1, 1]          16,384
           Sigmoid-9            [-1, 512, 1, 1]               0
 ChannelAttention-10            [-1, 512, 1, 1]               0
           Conv2d-11              [-1, 1, 7, 7]           4,803
          Sigmoid-12              [-1, 1, 7, 7]               0
 SpatialAttention-13              [-1, 1, 7, 7]               0
================================================================
Total params: 70,339
Trainable params: 70,339
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.10
Forward/backward pass size (MB): 0.03
Params size (MB): 0.27
Estimated Total Size (MB): 0.39
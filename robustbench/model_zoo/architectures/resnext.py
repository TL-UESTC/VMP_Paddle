"""ResNeXt implementation (https://arxiv.org/abs/1611.05431).

MIT License

Copyright (c) 2017 Xuanyi Dong

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

From:
https://github.com/google-research/augmix/blob/master/third_party/WideResNet_pytorch/wideresnet.py

"""

import math

import paddle.nn as nn
import paddle.nn.functional as F
import paddle
from paddle.nn import initializer
# import paddle.fluid.layers as F
# import paddle.fluid.dygraph as nn
# import paddle.fluid as fluid


class ResNeXtBottleneck(nn.Layer):
    """
    ResNeXt Bottleneck Block type C (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua).
    """
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 cardinality,
                 base_width,
                 stride=1,
                 downsample=None):
        super(ResNeXtBottleneck, self).__init__()

        dim = int(math.floor(planes * (base_width / 64.0)))

        self.conv_reduce = nn.Conv2D(
            inplanes,
            dim * cardinality,
            1,
            stride=1,
            padding=0,
            bias_attr=False)
        self.bn_reduce = nn.BatchNorm2D(dim * cardinality)

        self.conv_conv = nn.Conv2D(
            dim * cardinality,
            dim * cardinality,
            3,
            stride=stride,
            padding=1,
            groups=cardinality,
            bias_attr=False)
        self.bn = nn.BatchNorm2D(dim * cardinality)

        self.conv_expand = nn.Conv2D(
            dim * cardinality,
            planes * 4,
            1,
            stride=1,
            padding=0,
            bias_attr=False)
        self.bn_expand = nn.BatchNorm2D(planes * 4)

        self.downsample = downsample

    def forward(self, x):
        residual = x

        bottleneck = self.conv_reduce(x)
        # bottleneck = F.relu(self.bn_reduce(bottleneck), inplace=True)
        bottleneck = F.relu_(self.bn_reduce(bottleneck))

        bottleneck = self.conv_conv(bottleneck)
        # bottleneck = F.relu(self.bn(bottleneck), inplace=True)
        bottleneck = F.relu_(self.bn(bottleneck))

        bottleneck = self.conv_expand(bottleneck)
        bottleneck = self.bn_expand(bottleneck)

        if self.downsample is not None:
            residual = self.downsample(x)

        return F.relu_(residual + bottleneck)


class CifarResNeXt(nn.Layer):
    """ResNext optimized for the Cifar dataset, as specified in
    https://arxiv.org/pdf/1611.05431.pdf."""

    def __init__(self, block, depth, cardinality, base_width, num_classes):
        super(CifarResNeXt, self).__init__()

        # Model type specifies number of layers for CIFAR-10 and CIFAR-100 model
        assert (depth - 2) % 9 == 0, 'depth should be one of 29, 38, 47, 56, 101'
        layer_blocks = (depth - 2) // 9

        self.cardinality = cardinality
        self.base_width = base_width
        self.num_classes = num_classes

        self.conv_1_3x3 = nn.Conv2D(3, 64, 3, 1, 1, bias_attr=False)
        self.bn_1 = nn.BatchNorm2D(64)

        self.inplanes = 64
        self.stage_1 = self._make_layer(block, 64, layer_blocks, 1)
        self.stage_2 = self._make_layer(block, 128, layer_blocks, 2)
        self.stage_3 = self._make_layer(block, 256, layer_blocks, 2)
        self.avgpool = nn.AvgPool2D(8)
        self.classifier = nn.Linear(256 * block.expansion, num_classes)

        self.layers = [
            self.conv_1_3x3,self.bn_1,self.stage_1,self.stage_2,self.stage_3,self.avgpool,self.classifier
        ]
        
        
        initConst1 = initializer.Constant(1.0)
        initConst0 = initializer.Constant(0.0)
        initKM = initializer.KaimingNormal()
        
        for m in self.sublayers():
            t=0
            for name, param in m.named_parameters():
                print(t, name)
                t+=1
            if isinstance(m, nn.Conv2D):
                n = m._kernel_size[0] * m._kernel_size[1] * m._out_channels
                initNorm = initializer.Normal(0, math.sqrt(2. / n))
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                # m._param_attr=initializer.Normal(0, math.sqrt(2. / n))
                attr = paddle.ParamAttr(initializer=initNorm)
                m.weight = paddle.create_parameter(m.weight.shape,"float32",attr=attr)
            if isinstance(m, nn.BatchNorm2D):
                # m.weight.fill_(1)
                # m._param_attr=initializer.Constant(1.0)
                attr_w = paddle.ParamAttr(initializer=initConst1)
                m.weight = paddle.create_parameter(m.weight.shape,"float32",attr=attr_w)
                # m.bias.zero_()
                # m._bias_attr=initializer.Constant(0.0)
                attr_b = paddle.ParamAttr(initializer=initConst0)
                m.bias = paddle.create_parameter(m.bias.shape,"float32",attr=attr_b)
            elif isinstance(m, nn.Linear):
                # initializer.KaimingNormal(m._weight_attr)
                # m._bias_attr=initializer.Constant(0.0)
                attr_b = paddle.ParamAttr(initializer=initKM)
                m.bias = paddle.create_parameter(m.bias.shape,"float32",attr=attr_b)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            n=planes * block.expansion
            downsample = nn.Sequential(
                nn.Conv2D(
                    self.inplanes,
                    planes * block.expansion,
                    1,
                    stride=stride,
                    bias_attr=False),
                    # param_attr =paddle.ParamAttr(initializer=initializer.Normal(mean=0, std=math.sqrt(2. / n)))),
                nn.BatchNorm2D(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, self.cardinality, self.base_width, stride,
                  downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, self.cardinality, self.base_width))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_1_3x3(x)
        # x = F.relu(self.bn_1(x), inplace=True)
        x = F.relu_(self.bn_1(x))
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.avgpool(x)
        x = paddle.reshape(x, [x.shape[0], -1])
        return self.classifier(x)

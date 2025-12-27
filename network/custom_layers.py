import numpy as np
import paddle.nn as nn
from paddle.vision.ops import DeformConv2D

def get_pad(in_, ksize, stride, atrous=1):
    out_ = np.ceil(float(in_) / stride)
    return int(((out_ - 1) * stride + atrous * (ksize - 1) + 1 - in_) / 2)

class CustomConv2D(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 activation=nn.LeakyReLU(0.2)):
        super(CustomConv2D, self).__init__()
        self.conv2d = nn.Conv2D(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                dilation=dilation, groups=groups, bias_attr=bias)
        self.conv2d = nn.utils.spectral_norm(self.conv2d)
        self.activation = activation
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                n = m.weight.shape[0] * m.weight.shape[1] * m.weight.shape[2]
                v = np.random.normal(loc=0., scale=np.sqrt(2. / n), size=m.weight.shape).astype('float32')
                m.weight.set_value(v)

    def forward(self, input):
        x = self.conv2d(input)
        if self.activation is not None:
            return self.activation(x)
        else:
            return x


class CustomDeConv2D(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 output_padding=1, bias=True, activation=nn.LeakyReLU(0.2)):
        super(CustomDeConv2D, self).__init__()
        self.conv2d = nn.Conv2DTranspose(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                         padding=padding, dilation=dilation, groups=groups,
                                         output_padding=output_padding, bias_attr=bias)
        self.conv2d = nn.utils.spectral_norm(self.conv2d)
        self.activation = activation

    def forward(self, input):
        x = self.conv2d(input)
        if self.activation is not None:
            return self.activation(x)
        else:
            return x

class AdvancedConv2D(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 activation=nn.LeakyReLU(0.2), is_dcn=False):
        super(AdvancedConv2D, self).__init__()

        self.is_dcn = is_dcn
        if is_dcn:
            self.offsets = nn.Conv2D(in_channels, 18, kernel_size=3, stride=2,padding=1)
            self.conv2d = DeformConv2D(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.conv2d = nn.utils.spectral_norm(self.conv2d)

        else:
            self.depthwise_conv = nn.Conv2D(in_channels,in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=in_channels,
                bias_attr=bias
            )
            self.depthwise_conv = nn.utils.spectral_norm(self.depthwise_conv)
            self.pointwise_conv = nn.Conv2D(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias_attr=bias
            )
            self.pointwise_conv = nn.utils.spectral_norm(self.pointwise_conv)

        self.activation = activation
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                n = m.weight.shape[0] * m.weight.shape[1] * m.weight.shape[2]
                v = np.random.normal(loc=0., scale=np.sqrt(2. / n), size=m.weight.shape).astype('float32')
                m.weight.set_value(v)

    def forward(self, input):
        if self.is_dcn:
            offsets = self.offsets(input)
            x = self.conv2d(input, offsets)
        else:
            x = self.depthwise_conv(input)
            x = self.pointwise_conv(x)

        if self.activation is not None:
            return self.activation(x)
        else:
            return x

class AdvancedDeConv2D(nn.Layer):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 output_padding=1, bias=True, activation=nn.LeakyReLU(0.2)):
        super(AdvancedDeConv2D, self).__init__()
        self.depthwise_conv = nn.Conv2DTranspose(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            output_padding=output_padding,
            bias_attr=bias
        )
        self.depthwise_conv = nn.utils.spectral_norm(self.depthwise_conv)
        self.pointwise_conv = nn.Conv2DTranspose(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            output_padding=0,
            bias_attr=bias
        )
        self.pointwise_conv = nn.utils.spectral_norm(self.pointwise_conv)
        self.activation = activation

    def forward(self, inputs):
        x = self.depthwise_conv(inputs)
        x = self.pointwise_conv(x)

        if self.activation is not None:
            return self.activation(x)
        else:
            return x
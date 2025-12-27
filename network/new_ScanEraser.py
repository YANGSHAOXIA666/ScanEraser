import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from .custom_layers import get_pad,CustomConv2D, CustomDeConv2D,AdvancedConv2D,AdvancedDeConv2D
from .component import Multi_cloblock,ImprovedConvFFN,SFusionModule
from paddle.nn import Conv2D, BatchNorm2D,AdaptiveAvgPool2D

class Residual(nn.Layer):
    def __init__(self, inp, oup, stride, expand_ratio, keep_3x3=False):
        super(Residual, self).__init__()
        assert stride in [1, 2]
        hidden_dim = inp // expand_ratio
        self.identity = False
        self.identity_div = 1
        self.expand_ratio = expand_ratio

        if expand_ratio == 2:
            self.conv = nn.Sequential(
                Conv2D(
                    inp, inp, 3, 1, 1, groups=inp, bias_attr=False),
                BatchNorm2D(inp),
                nn.ReLU6(),
                Conv2D(
                    inp, hidden_dim, 1, 1, 0, bias_attr=False),
                BatchNorm2D(hidden_dim),
                Conv2D(
                    hidden_dim, oup, 1, 1, 0, bias_attr=False),
                BatchNorm2D(oup),
                nn.ReLU6(),
                Conv2D(
                    oup, oup, 3, stride, 1, groups=oup, bias_attr=False),
                BatchNorm2D(oup))
        else:
            if keep_3x3 == False:
                self.identity = True
            self.conv = nn.Sequential(
                Conv2D(
                    inp, inp, 3, 1, 1, groups=inp, bias_attr=False),
                BatchNorm2D(inp),
                nn.ReLU6(),
                Conv2D(
                    inp, hidden_dim, 1, 1, 0, bias_attr=False),
                BatchNorm2D(hidden_dim),
                Conv2D(
                    hidden_dim, oup, 1, 1, 0, bias_attr=False),
                BatchNorm2D(oup),
                nn.ReLU6(),
                Conv2D(
                    oup, oup, 3, 1, 1, groups=oup, bias_attr=False),
                BatchNorm2D(oup))

    def forward(self, x):
        out = self.conv(x)

        if self.identity:
            if self.identity_div == 1:
                out = out + x
            else:
                shape = x.shape
                id_tensor = x[:, :shape[1] // self.identity_div, :, :]
                out[:, :shape[1] // self.identity_div, :, :] = \
                    out[:, :shape[1] // self.identity_div, :, :] + id_tensor

        return out


class ScanEraserNet(nn.Layer):
    def __init__(self, global_dim, local_dim, heads, depths=[1, 1], attnconv_ks=[7, 9], pool_size=[8, 4], convffn_ks=5,
                 convffn_ratio=4, drop_path_rate=0.0):
        super(ScanEraserNet, self).__init__()
        self.conv1 = CustomConv2D(3, 32, kernel_size=4, stride=2,padding=1)
        self.conva = CustomConv2D(32, 32, kernel_size=3, stride=1,padding=1)
        self.convb = CustomConv2D(32, 64, kernel_size=4, stride=2, padding=1)

        self.res1 = Residual(64, 64, 1, 1)
        self.res2 = Residual(64, 128, 2, 2)

        self.conv2 = AdvancedConv2D(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = AdvancedConv2D(128, 128, kernel_size=3, stride=1, padding=1)

        self.deconv3 = CustomDeConv2D(128, 64, kernel_size=3,padding=1, stride=2)
        self.deconv4 = CustomDeConv2D(64 * 2, 32, kernel_size=3,padding=1, stride=2)
        self.deconv5 = CustomDeConv2D(64, 3, kernel_size=3, padding=1,stride=2)

        self.lateral_connection3 = nn.Sequential(
            nn.Conv2D(64, 128, kernel_size=3, padding=1, stride=1),
            nn.Conv2D(128, 64, kernel_size=1, padding=0, stride=1), )
        self.lateral_connection4 = nn.Sequential(
            nn.Conv2D(32, 64, kernel_size=3, padding=1, stride=1),
            nn.Conv2D(64, 32, kernel_size=1, padding=0, stride=1), )
        self.conv_o1 = nn.Conv2D(64, 3, kernel_size=1)
        self.conv_o2 = nn.Conv2D(32, 3, kernel_size=1)

        self.mask_deconv_c = CustomDeConv2D(128, 64, kernel_size=3,
                                                  padding=1, stride=2)
        self.mask_conv_c = AdvancedConv2D(64, 32, kernel_size=3,
                                               padding=1, stride=1)
        self.mask_deconv_d = CustomDeConv2D(64, 32, kernel_size=3,
                                                  padding=1, stride=2)
        self.mask_conv_d = nn.Conv2D(32, 3, kernel_size=1)
        self.sig = nn.Sigmoid()
        self.pspmodule = SFusionModule(64, [1, 2, 3, 6])

        cnum = 16
        self.astrous_net = nn.Sequential(AdvancedConv2D(4 * cnum, 4 *
                                                             cnum, 3, 1, dilation=2, padding=get_pad(64, 3, 1, 2)),
                                         AdvancedConv2D(4 * cnum, 4 * cnum, 3, 1, dilation=4,
                                                             padding=get_pad(64, 3, 1, 4)),
                                         AdvancedConv2D(4 * cnum, 4 *
                                                             cnum, 3, 1, dilation=8, padding=get_pad(64, 3, 1, 8)),
                                         AdvancedConv2D(4 * cnum, 4 * cnum, 3, 1, dilation=16,
                                                             padding=get_pad(64, 3, 1, 16)))

        cnum = 32
        self.coarse_conva = AdvancedConv2D(6, 32, kernel_size=4, stride=2,
                                        padding=1)
        self.coarse_convb = AdvancedConv2D(cnum, 2 * cnum, kernel_size=4, stride=2, padding=1)
        self.coarse_convc = AdvancedConv2D(2 * cnum, 2 * cnum,
                                               kernel_size=3, stride=1, padding=1)

        self.coarse_res1 = Residual(64, 128, 2, 2)
        self.coarse_res2 = Residual(128, 128, 1, 1)
        self.coarse_res3 = Residual(128, 128, 1, 1)

        self.coarse_convd = AdvancedConv2D(128, 128, kernel_size=3, stride=1, padding=1)

        self.coarse_deconva = CustomDeConv2D(4 * cnum * 2, 2 * cnum,
                                                   kernel_size=3, padding=1, stride=2)
        self.coarse_conve = AdvancedConv2D(2 * cnum, 2 * cnum,
                                               kernel_size=3, stride=1, padding=1)
        self.coarse_deconvb = CustomDeConv2D(2 * cnum * 2, cnum,
                                                   kernel_size=3, padding=1, stride=2)
        self.coarse_convf = AdvancedConv2D(cnum, cnum,
                                               kernel_size=3, stride=1, padding=1)
        self.coarse_deconvc = CustomDeConv2D(2 * cnum, 3,
                                                   kernel_size=3, padding=1, stride=2)

        self.c1 = nn.Conv2D(32, 64, kernel_size=1)
        self.c2 = nn.Conv2D(64, 128, kernel_size=1)
        self.iteration = 2

        dprs = [x.item() for x in paddle.linspace(0, drop_path_rate, sum(depths))]
        for i in range(len(depths)):
            layers = []
            dpr = dprs[sum(depths[:i]):sum(depths[:i + 1])]
            for j in range(depths[i]):
                if j < depths[i] - 1 or i == len(depths) - 1:
                    layers.append(
                        nn.Sequential(
                            Multi_cloblock(global_dim[i], local_dim[i], attnconv_ks[i], pool_size[i], heads[i], dpr[j]),
                            ImprovedConvFFN(global_dim[i] + local_dim[i], global_dim[i] + local_dim[i], convffn_ks, 1,
                                    convffn_ratio, dpr[j])
                        )
                    )
                else:
                    layers.append(
                        nn.Sequential(
                            Multi_cloblock(global_dim[i], local_dim[i], attnconv_ks[i], pool_size[i], heads[i], dpr[j]),
                            ImprovedConvFFN(global_dim[i] + local_dim[i], global_dim[i + 1] + local_dim[i + 1], convffn_ks, 1,
                                    convffn_ratio, dpr[j])
                        )
                    )

            self.__setattr__(f'stage{i}', nn.LayerList(layers))


    def forward(self, x):
        x = self.conv1(x)
        x = self.conva(x)
        con_x1 = x
        x = self.convb(x)

        x = self.res1(x)
        con_x2 = x
        x = self.res2(x)
        net, inp = paddle.split(x, [64, 64], axis=1)
        net = paddle.tanh(net)
        inp = F.relu(inp)
        net = self.astrous_net(net)
        for blk in self.stage0:
            inp = blk(inp)
        for blk in self.stage1:
            inp = blk(inp)
        x = paddle.concat([net, inp], axis=1)
        feature = self.conv2(x)

        x = self.conv3(feature)

        x = self.deconv3(x)
        xo1 = x
        x = paddle.concat([self.lateral_connection3(con_x2), x], axis=1)
        x = self.deconv4(x)
        xo2 = x
        x = paddle.concat([self.lateral_connection4(con_x1), x], axis=1)
        x = self.deconv5(x)

        x_o1 = self.conv_o1(xo1)
        x_o2 = self.conv_o2(xo2)
        x_o_unet = x

        x_mask = self.pspmodule(con_x2, 64)
        mm = self.mask_deconv_c(paddle.concat([x_mask, con_x2], axis=1))
        mm = self.mask_conv_c(mm)
        mm = self.mask_deconv_d(paddle.concat([mm, con_x1], axis=1))
        mm = self.mask_conv_d(mm)
        mm = self.sig(mm)

        input = x
        for i in range(self.iteration):
            x = paddle.concat([input , x], axis=1)
            x = self.coarse_conva(x)
            x_c1 = x
            x = self.coarse_convb(x)
            x_c2 = x
            x = self.coarse_convc(x)
            x = self.coarse_res1(x)
            x_c3 = x
            x = self.coarse_res2(x)
            x = self.coarse_res3(x)
            x = self.coarse_convd(x)
            x = self.coarse_deconva(paddle.concat([x, x_c3], axis=1))
            x = self.coarse_conve(x)
            x = self.coarse_deconvb(paddle.concat([x, x_c2], axis=1))
            x = self.coarse_convf(x)
            x = self.coarse_deconvc(paddle.concat([x, x_c1], axis=1))

        return x_o1, x_o2, x_o_unet, x, mm

def ScanEraser():
    global_feat_dim = [16, 32]
    local_feat_dim = [48, 32]
    heads = [8, 16]
    model = ScanEraserNet(global_feat_dim, local_feat_dim, heads)
    return model

if __name__ == '__main__':
    net = ScanEraser()
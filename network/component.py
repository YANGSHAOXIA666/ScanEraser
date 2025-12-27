import paddle
import paddle.nn as nn
from einops import rearrange

def drop_path(x, drop_prob=0.0, training=False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = paddle.to_tensor(1 - drop_prob)
    shape = (paddle.shape(x)[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + paddle.rand(shape, dtype=x.dtype)
    random_tensor = paddle.floor(random_tensor)
    output = x.divide(keep_prob) * random_tensor
    return output

class DropPath(nn.Layer):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Multi_cloblock(nn.Layer):
    def __init__(self, global_feat_dim, local_feat_dim, kernel_size, pool_size, head, qk_scale=None, drop_path_rate=0.0):
        super().__init__()
        self.global_feat_dim = global_feat_dim
        self.local_feat_dim = local_feat_dim
        self.head = head

        self.dwconv1 = nn.Conv2D(
            global_feat_dim + local_feat_dim, global_feat_dim + local_feat_dim, kernel_size=3, padding=1, dilation=1, stride=1,
            groups=global_feat_dim + local_feat_dim)
        self.dwconv2 = nn.Conv2D(
            global_feat_dim + local_feat_dim, global_feat_dim + local_feat_dim, kernel_size=5, padding=4, dilation=2, stride=1,
            groups=global_feat_dim + local_feat_dim)
        self.dwconv3 = nn.Conv2D(
            global_feat_dim + local_feat_dim, global_feat_dim + local_feat_dim, kernel_size=7, padding=9, dilation=3, stride=1,
            groups=global_feat_dim + local_feat_dim)
        self.conv0 = nn.Conv2D((global_feat_dim + local_feat_dim) * 3, global_feat_dim + local_feat_dim, 1)
        self.norm = nn.LayerNorm(global_feat_dim + local_feat_dim)

        self.global_feat_head = int(self.head * self.global_feat_dim / (self.global_feat_dim + self.local_feat_dim))
        self.fc1 = nn.Linear(global_feat_dim, global_feat_dim * 3)
        self.pool1 = nn.AvgPool2D(pool_size)
        self.pool2 = nn.AvgPool2D(pool_size)
        self.qk_scale = qk_scale or global_feat_dim ** -0.5
        self.softmax = nn.Softmax(axis=-1)

        self.local_feat_head = int(self.head * self.local_feat_dim / (self.global_feat_dim + self.local_feat_dim))
        self.fc2 = nn.Linear(local_feat_dim, local_feat_dim * 3)
        self.qconv = nn.Conv2D(local_feat_dim // self.local_feat_head, local_feat_dim // self.local_feat_head, kernel_size,
                               padding=kernel_size // 2, groups=local_feat_dim // self.local_feat_head)
        self.kconv = nn.Conv2D(local_feat_dim // self.local_feat_head, local_feat_dim // self.local_feat_head, kernel_size,
                               padding=kernel_size // 2, groups=local_feat_dim // self.local_feat_head)
        self.vconv = nn.Conv2D(local_feat_dim // self.local_feat_head, local_feat_dim // self.local_feat_head, kernel_size,
                               padding=kernel_size // 2, groups=local_feat_dim // self.local_feat_head)
        self.fc3 = nn.Conv2D(local_feat_dim // self.local_feat_head, local_feat_dim // self.local_feat_head, 1)
        self.swish = nn.Swish()
        self.fc4 = nn.Conv2D(local_feat_dim // self.local_feat_head, local_feat_dim // self.local_feat_head, 1)
        self.tanh = nn.Tanh()
        self.fc5 = nn.Conv2D(global_feat_dim + local_feat_dim, global_feat_dim + local_feat_dim, 1)
        self.drop_path = DropPath(drop_path_rate)

    def forward(self, x):
        identity = x
        test = x
        x1 = self.dwconv1(test)
        x2 = self.dwconv2(test)
        x3 = self.dwconv3(test)
        x = paddle.concat([x1, x2, x3], axis=1)
        x = self.conv0(x)
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w->b (h w) c')
        x = self.norm(x)

        x_local, x_global = paddle.split(x, [self.local_feat_dim, self.global_feat_dim], axis=-1)
        global_qkv = self.fc1(x_global)
        global_qkv = rearrange(global_qkv, 'b n (m h c)->m b h n c', m=3, h=self.global_feat_head)
        global_q, global_k, global_v = global_qkv[0], global_qkv[1], global_qkv[2]
        global_k = rearrange(global_k, 'b m (h w) c->b (m c) h w', h=H, w=W)
        global_k = self.pool1(global_k)
        global_k = rearrange(global_k, 'b (m c) h w->b m (h w) c', m=self.global_feat_head)
        global_v = rearrange(global_v, 'b m (h w) c->b (m c) h w', h=H, w=W)
        global_v = self.pool1(global_v)
        global_v = rearrange(global_v, 'b (m c) h w->b m (h w) c', m=self.global_feat_head)
        attn = global_q @ global_k.transpose([0, 1, 3, 2]) * self.qk_scale
        attn = self.softmax(attn)
        x_global = attn @ global_v
        x_global = rearrange(x_global, 'b m (h w) c-> b (m c) h w', h=H, w=W)

        local_qkv = self.fc2(x_local)
        local_qkv = rearrange(local_qkv, 'b (h w) (m n c)->m (b n) c h w', m=3, h=H, w=W, n=self.local_feat_head)
        local_q, local_k, local_v = local_qkv[0], local_qkv[1], local_qkv[2]
        local_q = self.qconv(local_q)
        local_k = self.kconv(local_k)
        local_v = self.vconv(local_v)
        attn = local_q * local_k
        attn = self.fc4(self.swish(self.fc3(attn)))
        attn = self.tanh(attn / (self.local_feat_dim ** -0.5))
        x_local = attn * local_v
        x_local = rearrange(x_local, '(b n) c h w->b (n c) h w', b=B)

        x = paddle.concat([x_local, x_global], axis=1)
        x = self.fc5(x)
        out = identity + self.drop_path(x)
        return out

class ImprovedConvFFN(nn.Layer):
    def __init__(self, in_dim, out_dim, kernel_size, stride, exp_ratio=4, drop_path_rate=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.fc1 = nn.Conv2D(in_dim, int(exp_ratio * in_dim), 1)
        self.gelu = nn.GELU()
        self.dwconv1 = nn.Conv2D(int(exp_ratio * in_dim), int(exp_ratio * in_dim), kernel_size,
                                 padding=kernel_size // 2, stride=stride, groups=int(exp_ratio * in_dim))
        self.fc2 = nn.Conv2D(int(exp_ratio * in_dim), out_dim, 1)
        self.drop_path = DropPath(drop_path_rate)
        self.downsample = stride > 1
        if self.downsample:
            self.dwconv2 = nn.Conv2D(in_dim, in_dim, kernel_size, padding=kernel_size // 2, stride=stride,
                                     groups=in_dim)
            self.norm2 = nn.BatchNorm2D(in_dim)
            self.fc3 = nn.Conv2D(in_dim, out_dim, 1)

    def forward(self, x):
        if self.downsample:
            identity = self.fc3(self.norm2(self.dwconv2(x)))
        else:
            identity = x

        x = rearrange(x, 'b c h w->b h w c')
        x = self.norm1(x)
        x = rearrange(x, 'b h w c->b c h w')
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dwconv1(x)
        x = self.fc2(x)
        out = identity + self.drop_path(x)
        return out

class SFusionModule(nn.Layer):
    def __init__(self, num_channels, bin_size_list):
        super(SFusionModule, self).__init__()
        num_filters = num_channels // len(bin_size_list)
        self.features = nn.LayerList()
        for i in range(len(bin_size_list)):
            self.features.append(
                paddle.nn.Sequential(
                    paddle.nn.AdaptiveMaxPool2D(output_size=bin_size_list[i]),
                    paddle.nn.Conv2D(in_channels=num_channels, out_channels=num_filters, kernel_size=1),
                    paddle.nn.BatchNorm2D(num_features=num_filters)
                )
            )

    def forward(self, inputs, out_channels):
        out = []
        for idx, layerlist in enumerate(self.features):
            x = layerlist(inputs)
            x = paddle.nn.functional.interpolate(x=x, size=inputs.shape[2:], mode='bilinear', align_corners=True)
            out.append(x)
        out = paddle.concat(x=out, axis=1)
        return out
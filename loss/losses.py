import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from network.discriminator import Discriminator_STE
from PIL import Image
import numpy as np
import vgg

class pre_network(nn.Layer):
    def __init__(self, pretrained: str = None):
        super(pre_network, self).__init__()
        self.vgg_layers = vgg.vgg19(pretrained=pretrained).features
        self.layer_name_mapping = {
            '3': 'relu1',
            '8': 'relu2',
            '13': 'relu3',
        }

    def forward(self, x):
        output = {}

        for name, module in self.vgg_layers._sub_layers.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return output


def gram_matrix(feat):
    (b, c, h, w) = feat.shape
    feat = feat.reshape([b, c, h * w])
    feat_t = feat.transpose((0, 2, 1))
    gram = paddle.bmm(feat, feat_t) / (c * h * w)
    return gram


def viaual(image):
    im = image.transpose(1, 2).transpose(2, 3).detach().cpu().numpy()
    Image.fromarray(im[0].astype(np.uint8)).show()


def dice_loss(input, target):
    input = F.sigmoid(input)
    input = input.reshape([input.shape[0], -1])
    target = target.reshape([target.shape[0], -1])

    a = paddle.sum(input * target, 1)
    b = paddle.sum(input * input, 1) + 0.001
    c = paddle.sum(target * target, 1) + 0.001
    d = (2 * a) / (b + c)
    dice_loss = paddle.mean(d)
    return 1 - dice_loss


class LossL1(nn.Layer):
    def __init__(self):
        super(LossL1, self).__init__()
        self.l1 = nn.L1Loss()
        self.extractor = pre_network(pretrained='vgg.pdparams')#vgg
        self.discriminator = Discriminator_STE(3)
        self.D_optimizer = paddle.optimizer.Adam(learning_rate=0.00001,
                                                 parameters=self.discriminator.parameters(),
                                                 beta1=0,
                                                 beta2=0.9,
                                                 weight_decay=0.01)
        self.cudaAvailable = paddle.device.is_compiled_with_cuda()
        self.numOfGPUs = paddle.device.cuda.device_count()
        self.lamda = 10.0

    def forward(self, input, mask, x_o1, x_o2, output, mm, gt):

        D_real = self.discriminator(gt, mask)
        D_real = D_real.mean().sum() * -1
        D_fake = self.discriminator(output, mm)
        D_fake = D_fake.mean().sum() * 1
        D_loss = paddle.mean(F.relu(1. + D_real)) + paddle.mean(F.relu(1. + D_fake))

        D_fake = -paddle.mean(D_fake)
        self.D_optimizer.clear_grad()
        D_loss.backward(retain_graph=True)

        self.D_optimizer.step()

        output_comp = (1 - mask) * input + mask * output
        end_output = (1 - mm) * input + mm * output

        holeLoss = 15 * self.l1(mask * end_output, mask * gt)
        validAreaLoss = 3 * self.l1((1 - mask) * end_output, (1 - mask) * gt)
        mask_loss = dice_loss(mm, mask)

        masks_a = F.interpolate(mask, scale_factor=0.25)
        masks_b = F.interpolate(mask, scale_factor=0.5)
        gt1 = F.interpolate(gt, scale_factor=0.25)
        gt2 = F.interpolate(gt, scale_factor=0.5)
        img1 = F.interpolate(input, scale_factor=0.25)
        img2 = F.interpolate(input, scale_factor=0.5)

        mm_a = F.interpolate(mm, scale_factor=0.25)
        mm_b = F.interpolate(mm, scale_factor=0.5)
        end_xo1 = (1 - mm_a) * img1 + mm_a * x_o1
        end_xo2 = (1 - mm_b) * img2 + mm_b * x_o2

        msrloss = 1.5 * self.l1((1 - masks_b) * end_xo2, (1 - masks_b) * gt2) + 7.5 * self.l1(masks_b * end_xo2,
                                                                                              masks_b * gt2) + \
                  1.2 * self.l1((1 - masks_a) * end_xo1, (1 - masks_a) * gt1) + 6 * self.l1(masks_a * end_xo1,
                                                                                            masks_a * gt1)
        feat_output_comp = self.extractor(output_comp)
        feat_output = self.extractor(end_output)
        feat_gt = self.extractor(gt)

        prcLoss = 0.0

        maps = ['relu1', 'relu2', 'relu3']
        for i in range(3):
            prcLoss += 0.01 * self.l1(feat_output[maps[i]], feat_gt[maps[i]])
            prcLoss += 0.01 * self.l1(feat_output_comp[maps[i]], feat_gt[maps[i]])

        styleLoss = 0.0
        for i in range(3):
            styleLoss += 120 * self.l1(gram_matrix(feat_output[maps[i]]), gram_matrix(feat_gt[maps[i]]))
            styleLoss += 120 * self.l1(gram_matrix(feat_output_comp[maps[i]]), gram_matrix(feat_gt[maps[i]]))

        GLoss = msrloss + holeLoss + validAreaLoss + prcLoss + styleLoss + 0.1 * D_fake + mask_loss * 35
        return GLoss.sum()


class LossL2(nn.Layer):
    def __init__(self):
        super(LossL2, self).__init__()
        self.l1 = nn.L1Loss()

    def forward(self, input, mask, x_o1, x_o2, x_o3, output, mm, gt):
        end_output = (1 - mm)*input + mm * output
        holeLoss = self.l1(mask * end_output, mask * gt)
        validAreaLoss = self.l1((1 - mask) * end_output, (1 - mask) * gt)
        image_loss = self.l1(end_output, gt)
        mask_loss = dice_loss(mm, mask)

        masks_a = F.interpolate(mask, scale_factor=0.25)
        masks_b = F.interpolate(mask, scale_factor=0.5)
        gt1 = F.interpolate(gt, scale_factor=0.25)
        gt2 = F.interpolate(gt, scale_factor=0.5)
        img1 = F.interpolate(input , scale_factor =0.25)
        img2 = F.interpolate(input ,scale_factor = 0.5)

        mm_a = F.interpolate(mm, scale_factor=0.25)
        mm_b = F.interpolate(mm, scale_factor=0.5)
        end_xo1 = (1 - mm_a)*img1 + mm_a * x_o1
        end_xo2 = (1 - mm_b)*img2 + mm_b * x_o2

        msrloss =   0.5* self.l1((1-masks_b)*end_xo2,(1-masks_b)*gt2)+2.5*self.l1(masks_b*end_xo2,masks_b*gt2)+\
                    0.4 * self.l1((1-masks_a)*end_xo1,(1-masks_a)*gt1)+2*self.l1(masks_a*end_xo1,masks_a*gt1)

        GLoss =  6 * (holeLoss)  + 1.2 * (validAreaLoss) + mask_loss * 0.8 + msrloss + 12 * image_loss
        return GLoss.sum()




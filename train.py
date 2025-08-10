# 可视化
from visualdl import LogWriter
import os
# paddle包
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import DataLoader
from dehw_train_dataset.data_loader import TrainDataSet, ValidDataSet
from loss.losses import LossWithGAN_STE
# 使用SwinT增强的Erasenet
from network.new_ScanEraser import ScanEraser_xxs1

# 其他工具
import utils
import random
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math


# 计算psnr
log = LogWriter('log')
def psnr(img1, img2):
   mse = np.mean((img1/1.0 - img2/1.0) ** 2 )
   if mse < 1.0e-10:
      return 100
   return 10 * math.log10(255.0**2/mse)


# 训练配置字典
CONFIG = {
    'modelsSavePath': 'swin_ScanEraser',
    'batchSize': 16,
    'traindataRoot': 'dehw_train_dataset',
    'validdataRoot': 'work',
    # 'pretrained': '',
    'num_epochs': 100,
    'seed': 3971
}


# 设置随机种子
random.seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])
paddle.seed(CONFIG['seed'])
# noinspection PyProtectedMember
paddle.framework.random._manual_program_seed(CONFIG['seed'])


batchSize = CONFIG['batchSize']
if not os.path.exists(CONFIG['modelsSavePath']):
    os.makedirs(CONFIG['modelsSavePath'])

traindataRoot = CONFIG['traindataRoot']
validdataRoot = CONFIG['validdataRoot']

# 创建数据集容器
TrainData = TrainDataSet(training=True, file_path=traindataRoot)
TrainDataLoader = DataLoader(TrainData, batch_size=batchSize, shuffle=True,
                             num_workers=0, drop_last=True)
ValidData = ValidDataSet(file_path=validdataRoot)
ValidDataLoader = DataLoader(ValidData, batch_size=1, shuffle=True, num_workers=0, drop_last=True)

netG = ScanEraser_xxs1()

if CONFIG['pretrained'] is not None:
    print('loaded ')
    weights = paddle.load(CONFIG['pretrained'])
    netG.load_dict(weights)

lr = 1e-3
decay_factor = 0.8

G_optimizer = paddle.optimizer.Adam(learning_rate=lr, parameters=netG.parameters())
criterion = LossWithGAN_STE()
# loss_function = LossWithGAN_STE()

print('OK!')
num_epochs = CONFIG['num_epochs']
best_psnr = 42.75
iters = 0

for epoch_id in range(1, num_epochs + 1):

    netG.train()

    if epoch_id % 10 == 0:
        lr = lr * (decay_factor ** (epoch_id // 5))
        G_optimizer = paddle.optimizer.Adam(learning_rate=lr, parameters=netG.parameters())

    for k, (imgs, gts, masks) in enumerate(TrainDataLoader):
        iters += 1

        x_o1, x_o2, x_o_unet, x, mm = netG(imgs)
        # G_loss = loss_function(x_o1, x_o2, x_o_unet, x, mm ,gts , masks)
        G_loss = criterion(imgs, masks, x_o1, x_o2, x_o_unet,x,mm, gts)
        G_loss = G_loss.sum()

        G_loss.backward()
        G_optimizer.step()
        G_optimizer.clear_grad()
        if iters % 100 == 0:
            # print('epoch{}, iters{}, loss:{:.5f}, lr:{}'.format(
            #     epoch_id, iters, G_loss.item(), G_optimizer.get_lr()
            # ))
            log.add_scalar(tag="train_loss", step=iters, value=G_loss.item())

    netG.eval()
    val_psnr = 0
    val_psnr1 = 0

    # noinspection PyAssignmentToLoopOrWithParameter
    for index, (imgs, gt) in enumerate(ValidDataLoader):
        _, _, h, w = imgs.shape
        rh, rw = h, w
        step = 512
        pad_h = step - h if h < step else 0
        pad_w = step - w if w < step else 0
        m = nn.Pad2D((0, pad_w, 0, pad_h))
        imgs = m(imgs)
        _, _, h, w = imgs.shape
        res = paddle.zeros_like(imgs)
        res1 = paddle.zeros_like(imgs)
        mm_out = paddle.zeros_like(imgs)
        mm_in = paddle.zeros_like(imgs)

        for i in range(0, h, step):
            for j in range(0, w, step):
                if h - i < step:
                    i = h - step
                if w - j < step:
                    j = w - step
                clip = imgs[:, :, i:i + step, j:j + step]
                clip = clip.cuda()
                with paddle.no_grad():
                    x_o1, x_o2, x_o_unet, x, mm = netG(clip)
                x= x.cpu()
                mm = mm.cpu()
                x_o_unet = x_o_unet.cpu()
                mm_in[:, :, i:i + step, j:j + step] = mm

                g_image_clip_with_mask = clip * (1 - mm) + x_o_unet * mm
                res[:, :, i:i + step, j:j + step] = g_image_clip_with_mask

                test1 = clip * (1 - mm) + x * mm
                res1[:, :, i:i + step, j:j + step] = test1

        res = res[:, :, :rh, :rw]
        res1 = res1[:, :, :rh, :rw]

        # 改变通道
        output = utils.pd_tensor2img(res)
        output1 = utils.pd_tensor2img(res1)
        target = utils.pd_tensor2img(gt)
        mm_in = utils.pd_tensor2img(mm_in)

        psnr_value = psnr(output, target)
        psnr_value1 = psnr(output1, target)
        # print('psnr: ', psnr_value)

        # if index in [2, 3]:
        #     fig = plt.figure(figsize=(20, 10),dpi=100)
        #     # 图一
        #     ax1 = fig.add_subplot(2, 2, 1)  # 1行 2列 索引为1
        #     ax1.imshow(output)
        #     # 图二
        #     ax2 = fig.add_subplot(2, 2, 2)
        #     ax2.imshow(mm_in)

        #     plt.show()

        del res
        del gt
        del target
        del output

        val_psnr += psnr_value
        val_psnr1 +=psnr_value1
    ave_psnr = val_psnr / (index + 1)
    ave_psnr1 = val_psnr1 / (index + 1)
    # print('epoch:{}, psnr:{}'.format(epoch_id, ave_psnr))
    print('epoch:{}, psnr:{}, psnr1:{}'.format(epoch_id, ave_psnr,ave_psnr1))
    log.add_scalar(tag="valid_psnr", step=epoch_id, value=ave_psnr)
    # paddle.save(netG.state_dict(), CONFIG['modelsSavePath'] +
    #             '/STE_{}_{:.4f}.pdparams'.format(epoch_id, ave_psnr
    #             ))
    if ave_psnr > best_psnr:
        best_psnr = ave_psnr
        paddle.save(netG.state_dict(), CONFIG['modelsSavePath'] + '/STE_best.pdparams')
import paddle
import numpy as np
import cv2
import os
import random
from PIL import Image

from paddle.vision.transforms import Compose, RandomCrop, ToTensor
from paddle.vision.transforms import functional as F


def random_horizontal_flip(imgs):
    if random.random() < 0.3:
        for i in range(len(imgs)):
            imgs[i] = imgs[i].transpose(Image.FLIP_LEFT_RIGHT)
    return imgs


def random_rotate(imgs):
    if random.random() < 0.3:
        max_angle = 10
        angle = random.random() * 2 * max_angle - max_angle
        for i in range(len(imgs)):
            img = np.array(imgs[i])
            w, h = img.shape[:2]
            rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
            img_rotation = cv2.warpAffine(img, rotation_matrix, (h, w))
            imgs[i] = Image.fromarray(img_rotation)
    return imgs


def ImageTransform():
    return Compose([ToTensor(), ])


class TrainDataSet(paddle.io.Dataset):

    def __init__(self, training=True, file_path=None):
        super().__init__()
        self.training = training
        self.path = file_path
        self.image_list = os.listdir(self.path + '/images')
        self.image_path = self.path + '/images'

        self.gt_path = self.path + '/gts'

        self.mask_path = self.path + '/mask'

        self.ImgTrans = ImageTransform()
        self.RandomCropparam = RandomCrop(512, pad_if_needed=True)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):

        while True:
            image_path = self.image_list[index]

            img = Image.open(self.image_path + '/' + image_path).convert('RGB')
            try:
                gt = Image.open(self.gt_path + '/' + image_path[:-4] + '.jpg').convert('RGB')
            except:
                gt = Image.open(self.gt_path + '/' + image_path[:-4] + '.png').convert('RGB')
            try:
                mask = Image.open(self.mask_path + '/' + image_path[:-4] + '.jpg').convert('RGB')
            except:
                mask = Image.open(self.mask_path + '/' + image_path[:-4] + '.png').convert('RGB')

            if np.array(img).shape[0] <= 512 or np.array(img).shape[1] <= 512:
                index += 1
            else:
                break

        param = self.RandomCropparam._get_param(img.convert('RGB'), (512, 512))
        inputImage = F.crop(img.convert('RGB'), *param)
        maskIn = F.crop(255 - np.array(mask.convert('RGB')), *param)
        groundTruth = F.crop(gt.convert('RGB'), *param)
        del img
        del gt
        del mask

        inputImage = self.ImgTrans(inputImage)
        maskIn = self.ImgTrans(maskIn)
        groundTruth = self.ImgTrans(groundTruth)

        return inputImage, groundTruth, maskIn


# 验证数据集
class ValidDataSet(paddle.io.Dataset):
    def __init__(self, file_path=None):
        super().__init__()
        self.path = file_path
        self.image_list = os.listdir(self.path + '/image')
        self.image_path = self.path + '/image'

        self.gt_path = self.path + '/gts'
        self.gt_list = os.listdir(self.gt_path)

        # self.mask_path = self.path + '/mask'
        # self.mask_list = os.listdir(self.mask_path)

        self.ImgTrans = ImageTransform()

    def __getitem__(self, index):
        image_path = self.image_list[index]

        img = Image.open(self.image_path + '/' + image_path).convert('RGB')
        try:
            gt = Image.open(self.gt_path + '/' + image_path[:-4] + '.jpg').convert('RGB')
        except:
            gt = Image.open(self.gt_path + '/' + image_path[:-4] + '.png').convert('RGB')

        inputImage = self.ImgTrans(img)
        groundTruth = self.ImgTrans(gt)

        return inputImage, groundTruth

    def __len__(self):
        return 200
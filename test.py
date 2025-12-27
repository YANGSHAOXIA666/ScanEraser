import os
import sys
import glob
import time
import json
import cv2
import paddle
import paddle.nn as nn
from network.new_ScanEraser import ScanEraser
import utils
from paddle.vision.transforms import Compose, ToTensor
from PIL import Image

netG = ScanEraser()
weights = paddle.load('')
netG.load_dict(weights)
netG.eval()
print("ok")


def ImageTransform():
    return Compose([ToTensor(), ])


ImgTrans = ImageTransform()


def process(src_image_dir, save_dir):
    image_paths = glob.glob(os.path.join(src_image_dir, "*.jpg"))
    for image_path in image_paths:

        # do something
        img = Image.open(image_path)
        inputImage = paddle.to_tensor([ImgTrans(img)])

        _, _, h, w = inputImage.shape
        rh, rw = h, w
        step = 512
        pad_h = step - h if h < step else 0
        pad_w = step - w if w < step else 0
        m = nn.Pad2D((0, pad_w, 0, pad_h))
        imgs = m(inputImage)
        _, _, h, w = imgs.shape
        res = paddle.zeros_like(imgs)

        for i in range(0, h, step):
            for j in range(0, w, step):
                if h - i < step:
                    i = h - step
                if w - j < step:
                    j = w - step
                clip = imgs[:, :, i:i + step, j:j + step]
                # clip = clip.cuda()
                with paddle.no_grad():
                    x_o1, x_o2, x_o_unet, x, mm = netG(clip)
                x = x.cpu()
                mm = mm.cpu()
                clip = clip.cpu()
                g_image_clip_with_mask = x * mm + clip * (1 - mm)
                res[:, :, i:i + step, j:j + step] = g_image_clip_with_mask
        res = res[:, :, :rh, :rw]
        output = utils.pd_tensor2img(res)
        save_path = os.path.join(save_dir, os.path.basename(image_path).replace(".jpg", ".png"))
        cv2.imwrite(save_path, output)
        del output, res


if __name__ == "__main__":
    sys.argv = ["script.py", "test/test_image", "test/test_output"]
    assert len(sys.argv) == 3

    src_image_dir = sys.argv[1]
    save_dir = sys.argv[2]

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    process(src_image_dir, save_dir)

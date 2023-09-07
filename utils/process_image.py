import torch
import numpy as np
import cv2


def max_hw(imgs: list):
    """
    Takes in list of image names from the dataloader and return the max height and max width present in all of those images.
    """
    max_h = 0
    max_w = 0
    ims = []
    for img in imgs:
        img = cv2.imread(img)
        h, w = img.shape[:2]
        max_h, max_w = max(h, max_h), max(w, max_w)
        ims.append(img)
    return ims, max_h, max_w


def make_divisible(maxhw, by: int = 32):  # all images have height/width divisible by 32 to avoid encoder issues
    max_h, max_w = maxhw[0], maxhw[1]
    max_h = max_h - (max_h % by)
    max_w = max_w - (max_w % by)
    return max_h, max_w


def process_img_batch(imgs: list):
    """
    Takes in list of img names from the dataloader and converts them to Tensors.
    """
    y = []
    imgs, max_h, max_w = max_hw(imgs)
    max_h, max_w = make_divisible((max_h, max_w), 32)  # max_h and max_w should be a multiple of 32

    for img in imgs:
        img = letterbox(img, (max_h, max_w))
        img = img.transpose(2, 0, 1)  # channel first format
        y.append(img)
    return torch.from_numpy(np.array(y))


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    """
    Resize images to the new_shape and adds padding if required.
    """
    shape = im.shape[:2]  # current shape [height, width]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1]) # Scale ratio (new / old)

    # computer unpadded size
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border
    return im

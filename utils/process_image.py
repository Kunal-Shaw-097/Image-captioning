import torch
import numpy as np
import cv2


def read(ims: list):
    """
    Takes in list of image names from the dataloader and reads them.
    """
    imgs = []
    for im in ims:
        img = cv2.imread(im)
        imgs.append(img)
    return imgs

def process_img_batch(ims: list, img_size : int = 640):
    """
    Takes in list of img names from the dataloader and converts them to Tensors.
    """
    y = []
    imgs = read(ims)

    for img in imgs:
        img = letterbox(img, (img_size, img_size))
        y.append(img)
    return y

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

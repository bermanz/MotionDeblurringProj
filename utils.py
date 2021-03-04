import PIL
from PIL import Image
import numpy as np
import os
import torch
import torchvision.utils as utils
import matplotlib as plt
from enum import Enum, auto


class SetType(Enum):
    """LoopConf are the different possible configurations for a control loops"""

    train = auto()
    val = auto()
    test = auto()

class ObjType(Enum):
    """The objective type of the network"""

    SingleImgDblr = auto()
    VidFromMotion = auto()


def tensor2RGB(tensor):
    return (tensor.squeeze() * 255).clamp(0, 255).numpy().transpose(1, 2, 0).astype("uint8")


def rmRunningFields(checkpoint):
    runKeys = [key for key in checkpoint.keys() if "running_" in key]
    for key in runKeys:
        del checkpoint[key]


def load_image(filename, size=None, scale=None):
    img = Image.open(filename)
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    return img

def save_image(data, filename):

    if not isinstance(data, torch.Tensor):
        if isinstance(data, PIL.PngImagePlugin.PngImageFile) or isinstance(data, PIL.Image.Image):
            img = data
        else:
            img = Image.fromarray(data)
        img.save(filename)
    else:
        utils.save_image(data, filename)


def saveImages(data, path):
    import os
    if not os.path.isdir(path):
        os.mkdir(path)

    for i, img in enumerate(data):
        # save_image(targetDir + '/' + f'sharp{i}.png', tensor2RGB(img))
        save_image(tensor2RGB(img), path + '/' + f'sharp{i}.png')


def makeGif(imgPath):
    """Generates a gif based on images path"""
    import imageio
    filenames = os.listdir(imgPath)
    filenames.sort()
    images = []
    for filename in filenames:
        images.append(imageio.imread(os.path.join(imgPath, filename)))
    imageio.mimsave(os.path.join(imgPath, "sharpVid.gif"), images, duration=0.2)


def visualiseSet(dataSet, batchSz=4, objType=ObjType.SingleImgDblr):
    """Loads and shows a small subset of input-target pairs from the dataset."""
    setLoader = torch.utils.data.DataLoader(dataSet, batch_size=batchSz, shuffle=True, num_workers=0)
    inputs, targets = next(iter(setLoader))
    if objType == ObjType.SingleImgDblr:
        fig, ax = plt.subplots(2, batchSz)
    else:
        fig, ax = plt.subplots(2, (targets.shape[1] + 1) // 2, figsize=(19.2, 9.77))
    rows = ax.shape[0]
    cols = ax.shape[1]
    for i in range(rows):
        for j in range(cols):
            curAx = ax[i, j]
            if j == i == 0:
                showTensorImg(curAx, inputs.squeeze())
                curAx.set_title('Blurry')
            else:
                idx = i * cols + j - 1
                showTensorImg(curAx, targets.squeeze()[idx])
                curAx.set_title(f'Sharp {idx + 1}')
    # plt.tight_layout()


def resizeImg(img, maxXSize=720):
    """Resize input image to avoid large images inferencing issues"""
    if img.size[1] > maxXSize:
        downRat = maxXSize / img.size[1]
        outImg = img.resize((int(img.size[0] * downRat),maxXSize))
    else:
        outImg = img
    return outImg


def showTensorImg(axes, tensIn):
    """shows a tensor by converting it to a valid numpy array and plotting it."""
    import matplotlib
    img = tensIn.squeeze().numpy()
    if isinstance(axes, np.ndarray):
        for i, ax in enumerate(axes):
            ax.imshow(img[i].transpose(1, 2, 0))
            ax.axis('off')
    else:
        axes.imshow(img.transpose(1, 2, 0))
        axes.axis('off')


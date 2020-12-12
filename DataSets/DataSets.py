import torch
from torch.utils.data import Dataset, DataLoader
import h5py
from zipfile import ZipFile
import numpy as np
from PIL import Image
from enum import Enum, auto
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms

dataBaseDir = os.path.join(os.path.dirname(__file__), 'REDS')

class SetType(Enum):
    """LoopConf are the different possible configurations for a control loops"""

    train = auto()
    val = auto()
    test = auto()


def getSetParams(setPath):
    """Create and initialize an H5 file for the desired dataset.

        Params:
            zipPath (str): the path of the source zip file
            setSpatDim (tuple of ints): the dimension of the images
            dType (numpy.generic): the data type of the images

        Returns:
             numEl: the number of elements in the set
             spatDims: the spatial dimensions of the elements in the set
    """
    with ZipFile(setPath, 'r') as zip:
        srcFilesList = [file for file in zip.namelist() if '.png' in file]
        numEl = len(srcFilesList)
        setSpatDim = np.array(Image.open(zip.open(srcFilesList[0]))).shape

    return numEl, setSpatDim


def initH5File(setType, dType=np.uint8):
    """Create and initialize an H5 file for the desired dataset.

        Params:
             SetType (SetType): the set type
             dType (numpy.generic): the data type of the images
    """

    # create paths:
    xSrcPath = os.path.join(dataBaseDir, 'Raw', setType.name, setType.name + '_' + 'blur' + '.zip')
    ySrcPath = os.path.join(dataBaseDir, 'Raw', setType.name, setType.name + '_' + 'sharp' + '.zip')
    numEl, setSpatDim = getSetParams(xSrcPath)
    targetPath = os.path.join(dataBaseDir, 'H5', setType.name + '.h5')

    # Create the H5 file:
    with h5py.File(targetPath, "w") as h5File:
        h5File.create_dataset('_'.join(['x', setType.name]), (numEl, *setSpatDim), dtype=dType)
        h5File.create_dataset('_'.join(['y', setType.name]), (numEl, *setSpatDim), dtype=dType)

    # Initialize the H5 File:
    with h5py.File(targetPath, "a") as h5File:
        with tqdm(total=2*numEl, desc='Transferring Set to H5') as pbar:
            for label, path in zip(['x', 'y'], [xSrcPath, ySrcPath]):
                with ZipFile(path, 'r') as zipFile:
                    srcFilesList = [file for file in zipFile.namelist() if '.png' in file]
                    srcFilesList.sort()
                    for i, file in enumerate(srcFilesList):
                        data = zipFile.open(file)
                        img = Image.open(data)
                        imgNumpy = np.array(img)
                        h5File['_'.join([label, setType.name])][i] = imgNumpy
                        pbar.update(1)

def visualiseSet(dataSet, batchSz = 4):
    """Loads and shows a small subset of input-target pairs from the dataset."""
    setLoader = torch.utils.data.DataLoader(dataSet, batch_size=batchSz, shuffle=True, num_workers=0)
    fig, ax = plt.subplots(2, batchSz)
    inputs, targets = next(iter(setLoader))
    for idx in range(len(inputs)):
        showTensorImg(ax[0, idx], inputs[idx])
        ax[0, idx].set_title('input')
        showTensorImg(ax[1, idx], targets[idx])
        ax[1, idx].set_title('target')

    plt.suptitle('Random Collection of input-target pairs')
    plt.tight_layout()


def showTensorImg(axes, tensIn):
    """shows a tensor by converting it to a valid numpy array and plotting it."""
    img = tensIn.numpy().transpose((1, 2, 0))
    axes.imshow(img)
    axes.axis('off')


class DatasetH5(Dataset):
    def __init__(self, setType=SetType.train, transform=None):
        super(DatasetH5, self).__init__()

        self.file = h5py.File(os.path.join(dataBaseDir, 'H5', setType.name + '.h5'), 'r')
        self.transform = transform
        self.setType = setType

    def __getitem__(self, index):
        x = self.file['x_' + self.setType.name][index, ...]
        y = self.file['y_' + self.setType.name][index, ...]

        if self.transform is not None:
            x = self.transform(x)
        # y = torch.from_numpy(y)
        yTransform = transforms.ToTensor()
        y = yTransform(y)

        return x, y

    def __len__(self):
        return self.file['x_' + self.setType.name].shape[0]


if __name__ == '__main__':

    if not os.path.exists(os.path.join(dataBaseDir, 'H5', SetType.train.name + '.h5')):
        initH5File(SetType.train)

    if not os.path.exists(os.path.join(dataBaseDir, 'H5', SetType.val.name + '.h5')):
        initH5File(SetType.val)

    transform = transforms.Compose([transforms.ToTensor()])
    trainSet = DatasetH5(SetType.train, transform)
    visualiseSet(trainSet)

    valSet = DatasetH5(SetType.val, transform)
    visualiseSet(valSet)

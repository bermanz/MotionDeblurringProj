import h5py
import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import h5py
import sys
from zipfile import ZipFile
import numpy as np
from PIL import Image
from enum import Enum, auto
import os

class SetType(Enum):
    """LoopConf are the different possible configurations for a control loops"""

    Training = auto()
    Validation = auto()
    Test = auto()

def readFromZip(zipName):
    """a generator for reading images from a zip file"""
    with ZipFile(zipName, 'r') as zip:
        # printing all the contents of the zip file
        for file in zip.namelist():
            data = zip.open(file)
            img = Image.open(data)
            imgArr = np.array(img)
            yield imgArr


def initH5File(zipPath, setSpatDim = (720, 1280), dType=np.uint8):
    """Create and initialize an H5 file for the desired dataset.

        Params:
             zipPath (str): the path of the source zip file
             setSpatDim (tuple of ints): the dimension of the images
             dType (numpy.generic): the data type of the images

    """
    zipName = zipPath.split('/')[-1]
    targetFileName = zipName.split('.')[0] + '.h5'
    targetPath = os.path.join(os.path.dirname(__file__), 'H5', targetFileName)
    with ZipFile(zipPath, 'r') as zip:
        setSz = len(zip.namelist())

        # Create the H5 file:
        with h5py.File(targetPath, "w") as h5File:
          h5File.create_dataset('data', (setSz, *setSpatDim), dtype=dType)

        # Initialize the H5 File:
        with h5py.File(targetPath, "a") as h5File:
            for i, file in enumerate(zip.namelist()):
                data = zip.open(file)
                img = Image.open(data)
                imgArr = np.array(img)
                h5File['data'][i] = imgArr


class DatasetH5(Dataset):
    def __init__(self, setType=SetType.Training, transform=None):
        super(DatasetH5, self).__init__()

        self.file = h5py.File(setType+'.h5', 'r')
        self.transform = transform
        self.setType = setType

    def __getitem__(self, index):
        x = self.file['X_' + self.setType.name][index, ...]
        y = self.file['Y_' + self.setType.name][index, ...]


        if self.transform is not None:
            x = self.transform(x)
            y = torch.from_numpy(np.asarray(y).astype(int))

        return x, y

    def __len__(self):
        return self.file['X_' + self.setType].shape[0]



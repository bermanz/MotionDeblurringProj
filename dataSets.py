import torch
from torch.utils.data import Dataset, DataLoader
import h5py
from zipfile import ZipFile
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms
import torchvision.transforms.functional as tf
import matplotlib.gridspec as gridspec
import cv2
from scipy.ndimage import convolve
from scipy.io import loadmat
import sys
from utils import SetType, ObjType, visualiseSet

baseDir = os.path.join(os.path.dirname(__file__), "DataSets")

def getSubSetParams(setPath):
    """get the parameters of a subset of the data.

        Params:
            zipPath (str): the path of the source zip file
            setSpatDim (tuple of ints): the dimension of the images
            dType (numpy.generic): the data type of the images

        Returns:
             numEl (int): the number of elements in the subset
             spatDims (tuple): the spatial dimensions of the elements in the subset
             scenesNum (int): the number of different scenes available in the subset

    """
    with ZipFile(setPath, 'r') as zip:
        srcFilesList = [file for file in zip.namelist() if '.png' in file]
        numEl = len(srcFilesList)
        setSpatDim = np.array(Image.open(zip.open(srcFilesList[0]))).shape
        sceneList = np.unique([file.split('/')[-2] for file in srcFilesList])
        scenesNum = len(sceneList)
    return numEl, setSpatDim, scenesNum


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
    numEl = 0
    scenesNum = 0

    dirCont = [name for name in os.listdir(setPath) if ".zip" in name]
    for file in dirCont:
        curNumEl,  setSpatDim, curScenesNum = getSubSetParams(os.path.join(setPath, file))
        numEl += curNumEl
        scenesNum += curScenesNum

    return numEl, setSpatDim, scenesNum

def getBlurredImg(imgSeq):
    """Returns the blurry image based on the input sequence.

        Params:
            imgSeq (numpy.ndarray): a 4D array containing a sequence of N adjacent frames, concatenated along the
            1st dimension.

        Returns:
            blurredImg(numpy.ndarray): the blurred version of the input sequence (a 3D array)

    """
    blurredImg = imgSeq.mean(axis=0)
    return (blurredImg * 255).astype(np.uint8)


def simPhaseMaskAcq(rawFrames, psfs):
    """ Simulates the acquisition process when using a phase mask, described by the provided PSFs

        Params:
            rawFrames (numpy.ndarray): a 4D array containing a sequence of N adjacent frames, concatenated along the
            1st dimension.

            psfs (numpy.ndarray): a 4D array holding N 3D point-spread functions, each corresponding to a specific
            raw frame

        Returns:
            acqFrames (numpy.ndarray): a 4D array containing a the N acquired frames, after (simulated as) being
            acquired with the phase mask
    """
    acqFrames = np.zeros_like(rawFrames)
    for nFrame in range(rawFrames.shape[0]):
        curFrame = rawFrames[nFrame]
        curPsf = psfs[nFrame]
        for channel in range(curFrame.shape[-1]):
            curFilt = curPsf[:, :, channel]
            acqFrames[nFrame][:, :, channel] = convolve(curFrame[:, :, channel], curFilt, mode='nearest')

    return acqFrames


def initH5File(setType, rawPath,  dType=np.uint8, nFrames=7, downSamp=False, psfs=None):
    """Create and initialize an H5 file for the desired dataset.

        Params:
             SetType (SetType): the set type
             rawPath (str): the full path to the source dir of the raw data path
             dType (numpy.generic): the data type of the images
             objType (ObjType): the objective type of the network
             nFrames (int): the number of frames to reconstruct and to generate the blurry source frame from
             nFrames (bool): indicator for down-sampling the input images of 45% in both axes.
             psfs (numpy.ndarray): the PSFs of the phase-mask at the acquisition times of the input frames.
    """
    if not os.path.isdir(os.path.join(baseDir, "REDS")):
        os.mkdir(os.path.join(baseDir, "REDS"))
    if not os.path.isdir(os.path.join(baseDir, "REDS", "H5")):
        os.mkdir(os.path.join(baseDir, "REDS", "H5"))

    ## Generate Dataset:
    numEl, setSpatDim, scenesNum = getSetParams(rawPath)
    if downSamp:
        setSpatDim = (int(setSpatDim[0]*0.45), int(setSpatDim[1]*0.45), 3)
    elPerScene = numEl//scenesNum
    bluredPerScene = elPerScene // nFrames
    targetPath = os.path.join(baseDir, "REDS", 'H5')
    targetPathX = os.path.join(targetPath, setType.name + ObjType.VidFromMotion.name + '_x.h5')
    targetPathY = os.path.join(targetPath, setType.name + ObjType.VidFromMotion.name + '_y.h5')
    with h5py.File(targetPathY, "w") as h5File:
        h5File.create_dataset('_'.join(['y', setType.name]), (bluredPerScene * scenesNum, *setSpatDim), dtype=dType)
    with h5py.File(targetPathX, "w") as h5File:
        h5File.create_dataset('_'.join(['x', setType.name]), (bluredPerScene * scenesNum, nFrames, *setSpatDim), dtype=dType)

    # Initialize the H5 Files:
    h5FileX = h5py.File(targetPathX, "a")
    h5FileY = h5py.File(targetPathY, "a")

    dirCont = [name for name in os.listdir(rawPath) if ".zip" in name]
    with tqdm(total=numEl, desc=f'Generating a H5 {setType.name} set') as pbar:
        for file in dirCont:
            with ZipFile(os.path.join(rawPath, file), 'r') as zipFile:
                srcFilesList = [file for file in zipFile.namelist() if '.png' in file]
                srcFilesList.sort()
                sceneList = np.unique([file.split('/')[-2] for file in srcFilesList])
                for scene in sceneList:
                    sceneFiles = [file for file in srcFilesList if '/' + scene + '/' in file]
                    framesSeq = np.array([])
                    baseIdx = int(scene) * bluredPerScene
                    for i, file in enumerate(sceneFiles):
                        data = zipFile.open(file)
                        img = Image.open(data)
                        imgNumpy = np.array(img)
                        if downSamp:
                            imgDims = imgNumpy.shape
                            imgResized = cv2.resize(imgNumpy, (int(imgDims[1] * 0.45), int(imgDims[0] * 0.45)))
                        else:
                            imgResized = imgNumpy
                        if not framesSeq.any():
                            framesSeq = np.expand_dims(imgResized, axis=0)
                        else:
                            framesSeq = np.concatenate([framesSeq, np.expand_dims(imgResized, axis=0)])

                        if not ((i+1) % nFrames):  ## dump the nFrames into the h5 file as a sequence
                            for label, path in zip(['x', 'y'], [targetPathX, targetPathY]):
                                if label == 'x':
                                    h5FileX['_'.join([label, setType.name])][baseIdx + i//nFrames] = framesSeq.copy()
                                else:
                                    seqFloat = framesSeq.astype(float) / 255
                                    if psfs is not None:  # Simulate the phase-mask acquisition
                                        inSeq = simPhaseMaskAcq(seqFloat, psfs)
                                    else:
                                        inSeq = seqFloat
                                    h5FileY['_'.join([label, setType.name])][baseIdx + i//nFrames] = \
                                        getBlurredImg(inSeq)
                            framesSeq = np.array([])
                        pbar.update(1)
    h5FileX.close()
    h5FileY.close()


def getSetLoader(setType, batchSz, setIdx=None):
    dataSet = RedsH5(setType, objType=ObjType.VidFromMotion)
    if setIdx is not None:
        dataSubSet = torch.utils.data.Subset(dataSet, setIdx)
        setLoader = torch.utils.data.DataLoader(dataSubSet, batch_size=batchSz, num_workers=0)
    else:
        setLoader = torch.utils.data.DataLoader(dataSet, batch_size=batchSz, shuffle=True, num_workers=0)
    return setLoader

class RedsH5(Dataset):
    def __init__(self, setType=SetType.train, objType=ObjType.SingleImgDblr):
        super(RedsH5, self).__init__()
        self.objType = objType
        self.setType = setType

        if setType == SetType.test:
            setType = SetType.val

        if self.objType == ObjType.SingleImgDblr:
            self.file = h5py.File(os.path.join(baseDir, "REDS", 'H5', setType.name + '.h5'), 'r')
        else:
            self.fileX = h5py.File(os.path.join(baseDir,  "REDS", 'H5', setType.name + objType.name + '_x.h5'), 'r')
            self.fileY = h5py.File(os.path.join(baseDir,  "REDS", 'H5', setType.name + objType.name + '_y.h5'), 'r')


    def __getitem__(self, index):
        setType = self.setType if self.setType != SetType.test else SetType.val
        if self.objType == ObjType.SingleImgDblr:
            x = self.file['x_' + setType.name][index, ...]
            y = self.file['y_' + setType.name][index, ...]
        else:
            x = self.fileX['x_' + setType.name][index, ...]
            y = self.fileY['y_' + setType.name][index, ...]

        ## Combine all data to a single tensor
        toTensor = transforms.ToTensor()
        input = toTensor(y)
        target = torch.zeros([x.shape[0], *input.shape])
        for frame in range(x.shape[0]):
            target[frame] = toTensor(x[frame])
        combData = torch.cat((input.unsqueeze(0), target))

        ## Apply transformations for all data:
        if not self.setType == SetType.test:
            combTrans = transforms.Compose([transforms.RandomCrop(320),
                                            transforms.RandomVerticalFlip(),
                                            transforms.RandomHorizontalFlip()])
            combData = combTrans(combData)

        ## Re-divide data to input and target:
        input = combData[0]
        if self.setType == SetType.train:
            # Add 1% noise as done in the paper
            noise = torch.randn_like(input) * 1/100
            input += noise
            input = torch.clamp(input, 0, 1)

        target = combData[1:]
        return input, target

    def __len__(self):
        setType = self.setType if self.setType != SetType.test else SetType.val
        if self.objType == ObjType.SingleImgDblr:
            return self.file['y_' + setType.name].shape[0]
        else:
            return self.fileY['y_' + setType.name].shape[0]


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Generate datasets for video-from-image network training")
    parser.add_argument("-t", "--training", help="The full-path to the raw-training data directory")
    parser.add_argument("-v", "--validation", help="The full-path to the raw-validation data directory")
    parser.add_argument("-m", "--masked", type=int, default=1, choices=[0, 1],
                        help="Simulate phase-masking during dataset generation. default: 1 (do simulate)")
    parser.add_argument("--debug", action="store_true",
                        help="Display an exemplary set of input-targets upon completion")
    args = parser.parse_args()

    trainPath = args.training
    valPath = args.validation
    isSpatTempDist = args.masked
    debug = args.debug

    if isSpatTempDist:
        ## Load spatioTemporal static PSFs:
        spatTempPsfs = loadmat("DataSets/spatioTempSim/psfs.mat")['psfAr'].squeeze()
    else:
        spatTempPsfs = None

    if trainPath:
        initH5File(SetType.train, trainPath, psfs=spatTempPsfs)
        if debug:
            trainSet = RedsH5(SetType.train, objType=ObjType.VidFromMotion)
            visualiseSet(trainSet, batchSz=1, objType=ObjType.VidFromMotion)

    if valPath:
        initH5File(SetType.val, valPath, psfs=spatTempPsfs)
        if debug:
            valSet = RedsH5(SetType.val, objType=ObjType.VidFromMotion)
            visualiseSet(valSet, batchSz=1, objType=ObjType.VidFromMotion)

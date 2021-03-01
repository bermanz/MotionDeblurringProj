import torch
from Ref.model import *
device = "cuda" if torch.cuda.is_available() else "cpu"
from utils import *
from torchvision import transforms
import numpy as np
import os
from lossFuncs import *
import pandas as pd

def prepImgForInf(img):
    """Prepare image for inference"""

    ## Prepare image for inference:
    input = img
    width, height = input.size
    input = input.crop((0, 0, width - width % 20, height - height % 20))
    input_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    input = input_transform(input)
    input = input.unsqueeze(0)
    return input.to(device)

def estMidFrame(blurryImg, midNetWeights):

    ## Load trained model:
    model = centerEsti()
    model.load_state_dict(midNetWeights)

    model.to(device)
    model.eval()

    input = prepImgForInf(blurryImg)
    with torch.no_grad():
        xHat = model(input)

    return xHat

def estAllFrames(blurryImg, midFrameEst):
    """Estimates the 7 sharp frames based on the input estimated middle frame"""

    ## Load trained models:
    model2 = F35_N8()
    model3 = F26_N9()
    model4 = F17_N9()

    checkpoint = torch.load('Ref/models/F35_N8.pth', map_location=torch.device('cpu'))
    rmRunningFields(checkpoint)
    model2.load_state_dict(checkpoint['state_dict_G'])

    checkpoint = torch.load('Ref/models/F26_N9_from_F35_N8.pth', map_location=torch.device('cpu'))
    rmRunningFields(checkpoint)
    model3.load_state_dict(checkpoint['state_dict_G'])

    checkpoint = torch.load('Ref/models/F17_N9_from_F26_N9_from_F35_N8.pth', map_location=torch.device('cpu'))
    rmRunningFields(checkpoint)
    model4.load_state_dict(checkpoint['state_dict_G'])

    model2.to(device)
    model3.to(device)
    model4.to(device)

    model2.eval()
    model3.eval()
    model4.eval()

    if not isinstance(blurryImg, torch.Tensor):
        input = prepImgForInf(blurryImg)
    else:
        input = blurryImg

    ## Forward Pass:
    with torch.no_grad():
        output3_5 = model2(input, midFrameEst)
        output2_6 = model3(input, output3_5[0], midFrameEst, output3_5[1])
        output1_7 = model4(input, output2_6[0], output3_5[0], output3_5[1], output2_6[1])

    ## Parse Output frames:
    outputFrames = []
    if input.shape[0] > 1:
        outSeq = []
        for i in range(input.shape[0]):
            outSeq.append(output1_7[0][i].to('cpu'))
            outSeq.append(output2_6[0][i].to('cpu'))
            outSeq.append(output3_5[0][i].to('cpu'))
            outSeq.append(midFrameEst[i].to('cpu'))
            outSeq.append(output3_5[1][i].to('cpu'))
            outSeq.append(output2_6[1][i].to('cpu'))
            outSeq.append(output1_7[1][i].to('cpu'))
            outputFrames.append(outSeq.copy())
            outSeq.clear()
    else:
        outputFrames.append(output1_7[0].to('cpu'))
        outputFrames.append(output2_6[0].to('cpu'))
        outputFrames.append(output3_5[0].to('cpu'))
        outputFrames.append(midFrameEst.to('cpu'))
        outputFrames.append(output3_5[1].to('cpu'))
        outputFrames.append(output2_6[1].to('cpu'))
        outputFrames.append(output1_7[1].to('cpu'))

    return outputFrames


def getResult(inImgPath, solWeights, targetPath):
    """Get the output video for the input image using the input trained weights"""
    inImg = load_image(inImgPath)
    if not os.path.isdir(targetPath):
        os.mkdir(targetPath)

    save_image(inImg, os.path.join(targetPath, "blurry.png"))
    for weights in solWeights:
        midFrameEst = estMidFrame(inImg, weights)
        estFrames = estAllFrames(inImg, midFrameEst)
        saveImages(estFrames, targetPath)
        makeGif(targetPath)

def infBatch(netWeights, isPM=False, batchSz=4, setIdx=None):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = centerEsti()
    if netWeights is not None:
        model.load_state_dict(netWeights)
    model.to(device)

    valSet = getSetLoader(SetType.test, batchSz, setIdx=setIdx)

    model.eval()
    with torch.no_grad():
        y, x = next(iter(valSet))
        if not isPM:
            y = x.mean(axis=1)  # TODO: Verify if correct!
        nFrames = x.shape[1]
        midFrame = int(np.floor(nFrames / 2))

        if netWeights is not None:
            inputs, targets = y.to(device), x[:, midFrame, :, :].to(device)
            outputs = model(inputs)
            allOutFrames = estAllFrames(inputs, outputs)
        else:
            inputs = y
            allOutFrames = x

        return inputs, allOutFrames


def compSolutions(isQuant=True, isQual=True):
    """Compare all inspected solutions both quantitatively and qualitatively"""

    refCP = torch.load('Ref/models/center_v3.pth')
    rmRunningFields(refCP)
    refWeights = refCP["state_dict_G"]
    regWeights = torch.load('Models/RegCam/trainData_e44.pth')["weights"]
    pmWeights = torch.load('Models/PhaseMasked/trainData_e42.pth')["weights"]

    labels = ["Ref", "Reg", "PM"]
    quantRes = pd.DataFrame(index=["PSNR", "SSIM"], columns=labels)

    dataSet = RedsH5(SetType.test, objType=ObjType.VidFromMotion)
    batchSz = 4
    # setIdx = torch.randint(high=len(dataSet)-1, size=(batchSz,)).tolist()
    setIdx = [515,1253, 1710, 2100]
    labels.append("Target")
    for netWeights, label in zip([refWeights, regWeights, pmWeights, None], labels):
        if isQuant:
            if label is None:
                break
            quantRes.loc["PSNR", label], quantRes.loc["SSIM", label] = evalNetQuant(netWeights, batchSz=batchSz,
                                                                                    isPM=label == "PM")
        if isQual:
            blurry, outFrames = infBatch(netWeights, label == "PM", setIdx=setIdx)
            for i in range(blurry.shape[0]):
                baseDir = os.path.join("Results", str(setIdx[i]))
                if not os.path.isdir(baseDir):
                    os.mkdir(baseDir)

                if label=="Ref":
                    save_image(blurry[i], os.path.join(baseDir, "blurry.png"))
                saveImages(outFrames[i], os.path.join(baseDir, label))
                makeGif(os.path.join(baseDir, label))
    return quantRes


def vidFromBlur(blurImgPath, checkpoint, outpath):
    """Inferences a sharp video (7 sharp frames) from an input blurry frame, and saves the results under the Results
    directory."""
    checkDict = torch.load(checkpoint)
    if "state_dict_G" in checkDict:  # it's the pre-trained network from the paper
        rmRunningFields(checkDict)
        netWeights = checkDict['state_dict_G']
    else:  # my re-trained networks
        netWeights = checkDict['weights']
    getResult(blurImgPath, [netWeights], outpath)


if __name__ == "__main__":
    import os
    import argparse
    parser = argparse.ArgumentParser(description="Inference a sharp video using the trained video-from-image network")
    parser.add_argument("-b", "--blurry", help="The full-path to the blurry input image")
    parser.add_argument("-c", "--checkpoint", help="The full-path to a checkpoint of the trained network with which "
                                                   "to inference")
    parser.add_argument("-o", "--output", help="The full-path to the output directory")
    args = parser.parse_args()

    blurryImgPath = args.blurry
    checkpoint = args.checkpoint
    output = args.output

    vidFromBlur(blurryImgPath, checkpoint, output)

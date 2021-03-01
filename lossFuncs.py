from torchvision import models
from Ref.model import centerEsti
from DataSets import *

device = "cuda" if torch.cuda.is_available() else "cpu"
## Load VGG for evaluating the Perceptual Loss:
vgg16 = models.vgg16(pretrained=True).features
for param in vgg16.parameters():
    param.requires_grad_(False)
vgg16.to(device)

def l2Loss(x_hat, x):
    return ((x_hat - x) ** 2).mean()

def percLoss(x_hat, x):
    featuresX = getVggFeatures(x, vgg16)
    featuresX_hat = getVggFeatures(x_hat, vgg16)
    totLoss = 0
    for feature in featuresX.keys():
        totLoss += ((featuresX[feature] - featuresX_hat[feature]) ** 2).mean()
    return totLoss / len(featuresX)

def getVggFeatures(x, vgg16):
    """ Run an image forward through the pre-trained Vgg16 network and access it's feature map's required for
    evaluating the perceptual loss, as used by Jin et al. in the paper
    """

    ## The layers proposed by the authors for the perceptual loss:
    layers = {'8': 'relu2_2',
              '15': 'relu3_3'}
    features = {}
    # model._modules is a dictionary holding each module in the model
    for name, layer in vgg16._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

def pSNR(x, xhat):
    mse = ((x - xhat) ** 2).mean()
    return 20 * torch.log10(255.0 / torch.sqrt(mse))

def evalImgQuant(x, xhat):
    """Evaluate the quality of the reconstructed image xhat by calculating it's PSNR and SSIM"""
    from pytorch_msssim import ssim
    x = x.clip(0, 1) * 255
    xhat = xhat.clip(0, 1) * 255
    return pSNR(x, xhat), ssim(x, xhat)

def evalNetQuant(netWeights, batchSz=4, isPM=False):
    """Evaluate the network quantitatively over the entire validation set"""

    ## Load Model for training:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = centerEsti()
    model.load_state_dict(netWeights)
    model.to(device)

    valSet = getSetLoader(SetType.test, batchSz)

    totPsnr = 0
    totSsim = 0

    model.eval()
    valProg = tqdm(valSet, desc='Test', leave=False, ncols=100)
    with torch.no_grad():
        for y, x in valProg:
            if not isPM:
                y = x.mean(axis=1)  # TODO: Verify if correct!

            nFrames = x.shape[1]
            midFrame = int(np.floor(nFrames / 2))

            inputs, targets = y.to(device), x[:, midFrame, :, :].to(device)
            outputs = model(inputs)

            ## Loss Calculation:
            curPsnr, curSsim = evalImgQuant(targets, outputs)

            totPsnr += curPsnr
            totSsim += curSsim

    return totPsnr.to("cpu").numpy() / len(valSet), totSsim.to("cpu").numpy() / len(valSet)

from Ref.model import centerEsti
import pandas as pd

from DataSets import *
import lossFuncs


def train(nEpochs, optParams, batchSz=8, valEpochFact = 1, percWeight=3/4, checkpoint=None, outpath=None):
    """The training routine for the video-from-image network

        Params:
         nEpochs (float): the number of desired epochs to run the training
         optParams (dict): a dictionary containing the required optimizer arguments
         batchSz (int): the size of the batch
         valEpochFact (int): the number of training iterations to run before every validation iteration
         percWeight (float): the weight of the perceptual loss from the total-loss
         checkpoint (dict): a checkpoint of the network for initialization.
    """

    ## Load Model for training:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = centerEsti()
    if checkpoint is not None:
        checkDict = torch.load(checkpoint)
        model.load_state_dict(checkDict['weights'])
        e0 = checkDict['epoch']+1
    else:
        e0 = 0

    torch.cuda.empty_cache()
    model.to(device)

    ## Set Training:
    cols = pd.MultiIndex.from_product([['Training', 'Validation'], ['L2', 'Perceptual', 'Total']])
    idx = np.arange(start=e0, stop=e0+nEpochs, dtype=int)
    lossTable = pd.DataFrame(np.zeros((nEpochs, 6)), columns=cols, index=idx)

    optimizer = torch.optim.Adam(model.parameters(), **optParams)
    trainSet = getSetLoader(SetType.train, batchSz)
    valSet = getSetLoader(SetType.val, batchSz)

    valMinLoss = np.inf
    for epoch in tqdm(idx, desc='epochs', ncols=100):
        totLoss = 0
        totL2Loss = 0
        totPercLoss = 0
        model.train()
        trainProg = tqdm(trainSet, desc='training', leave=False, ncols=100)
        for y, x in trainProg:
            nFrames = x.shape[1]
            midFrame = int(np.floor(nFrames / 2))

            ## Forward Pass:
            optimizer.zero_grad()
            inputs, targets = y.to(device), x[:, midFrame, :, :].to(device)
            outputs = model(inputs)

            ## Loss Calculation:
            l2Loss = lossFuncs.l2Loss(outputs, targets)
            percLoss = lossFuncs.percLoss(outputs, targets)
            loss = l2Loss + percWeight * percLoss

            ## Back Propagation
            loss.backward()
            optimizer.step()

            ## Accumulate Loss:
            totL2Loss += l2Loss.item()
            totPercLoss += percLoss.item()
            totLoss += loss.item()
            trainProg.set_description(f'train loss {loss.item():.2}')

        lossTable.loc[epoch, ('Training', 'L2')] = totL2Loss / len(trainSet)
        lossTable.loc[epoch, ('Training', 'Perceptual')] = totPercLoss / len(trainSet)
        lossTable.loc[epoch, ('Training', 'Total')] = totLoss / len(trainSet)

        if epoch % valEpochFact == 0:
            torch.cuda.empty_cache()

            totLoss = 0
            totL2Loss = 0
            totPercLoss = 0
            model.eval()
            valProg = tqdm(valSet, desc='validation', leave=False, ncols=100)
            with torch.no_grad():
                for y, x in valProg:

                    nFrames = x.shape[1]
                    midFrame = int(np.floor(nFrames / 2))

                    inputs, targets = y.to(device), x[:, midFrame, :, :].to(device)
                    outputs = model(inputs)

                    ## Loss Calculation:
                    l2Loss = lossFuncs.l2Loss(outputs, targets)
                    percLoss = lossFuncs.percLoss(outputs, targets)
                    loss = l2Loss + percWeight * percLoss

                    totL2Loss += l2Loss.item()
                    totPercLoss += percLoss.item()
                    totLoss += loss.item()
                    valProg.set_description(f'validation loss {loss.item():.2}')

            lossTable.loc[epoch, ('Validation', 'L2')] = totL2Loss / len(valSet)
            lossTable.loc[epoch, ('Validation', 'Perceptual')] = totPercLoss / len(valSet)
            lossTable.loc[epoch, ('Validation', 'Total')] = totLoss / len(valSet)

            if totLoss < valMinLoss:  # Save Network's weights:
                checkpoint = {
                    "epoch": epoch,
                    "lr": optimizer.param_groups[0]['lr'],
                    "weights": model.state_dict()
                }
                if outpath:
                    torch.save(checkpoint, os.path.join(outpath, f'trainData_e{epoch}.pth'))
                else:
                    torch.save(checkpoint, f'trainData_e{epoch}.pth')
                valMinLoss = totLoss

    return lossTable, checkpoint


def testLoss(netWeights, percWeight=3/4, batchSz=8):
    """return the loss of the provided network over the test-set"""

    ## Load Model for training:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = centerEsti()
    model.load_state_dict(netWeights)
    model.to(device)

    valSet = getSetLoader(SetType.val, batchSz)

    totLoss = 0
    totL2Loss = 0
    totPercLoss = 0

    model.eval()
    valProg = tqdm(valSet, desc='validation', leave=False, ncols=100)
    with torch.no_grad():
        for y, x in valProg:
            nFrames = x.shape[1]
            midFrame = int(np.floor(nFrames / 2))

            inputs, targets = y.to(device), x[:, midFrame, :, :].to(device)
            outputs = model(inputs)

            ## Loss Calculation:
            l2Loss = lossFuncs.l2Loss(outputs, targets)
            percLoss = lossFuncs.percLoss(outputs, targets)
            loss = l2Loss + percWeight * percLoss

            totL2Loss += l2Loss.item()
            totPercLoss += percLoss.item()
            totLoss += loss.item()
            valProg.set_description(f'validation loss {loss.item():.2}')

    return totL2Loss / len(valSet), totPercLoss / len(valSet), totLoss / len(valSet)


if __name__ == "__main__":
    import os
    import argparse
    parser = argparse.ArgumentParser(description="Train the video-from-image network")
    parser.add_argument("-e", "--epochs", type=int, default=44,
                        help="The number of epochs to train for")
    parser.add_argument("-c", "--checkpoint", help="The full-path to a checkpoint for the training to initialize the "
                                                   "network with")
    parser.add_argument("-o", "--output", help="The full-path to the output directory")
    args = parser.parse_args()

    nEpochs = args.epochs
    inCheckpoint = args.checkpoint
    outPath = args.output

    # default parameters:
    optParams = {
        'lr': 1e-4,
        'betas': (0.5, 0.999),
        'eps': 1e-8,
        'weight_decay': 0
    }

    lossTable, outCheckpoint = train(nEpochs, optParams, checkpoint=inCheckpoint)
    if outPath:
        torch.save(outCheckpoint, os.path.join(outPath, f'trainData_e{nEpochs}.pth'))
        lossTable.to_csv(os.path.join(outPath, f'convData_e{nEpochs}.pth'))
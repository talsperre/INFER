import os
import sys
import math
import random
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import args
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from torch.autograd import Variable
from KittiDataset import KittiDataset

def obstacleLossFun(outputBatch, obstacleBatch):
    outFlat = outputBatch.view(-1)
    inpFlat = obstacleBatch.view(-1)
    intersection = (outFlat * inpFlat).abs().sum()
    return intersection / len(outFlat)

def heatmapAccuracy(outputMap, labelMap, thr=1.5):
    pred = np.unravel_index(outputMap.argmax(), outputMap.shape)
    gt = np.unravel_index(labelMap.argmax(), labelMap.shape)

    dist = math.sqrt((pred[0] - gt[0]) ** 2 + (pred[1] - gt[1]) ** 2)
    if dist <= thr:
        return 1, dist, (pred[0], pred[1])
    return 0, dist, (pred[0], pred[1])

def weightedMSE(outputMap, labelMap, weightMap):
    out = (outputMap - labelMap) ** 2
    out = out * weightMap
    loss = out.sum(0)
    return loss

def weightMatrix(labelMap):
    labelClone = labelMap.clone()
    weightMat = labelMap.clone()
    num_nonzeros = torch.nonzero(labelClone).size(0)
    num_zeros = cmd.imageHeight * cmd.imageWidth - num_nonzeros
    weightMat[labelClone == 0] = float(1) / num_zeros
    weightMat[labelClone != 0] = float(1) / num_nonzeros
    return weightMat

# Returns True if the model is LSTM based
def loadModel(modelType, imageWidth, imageHeight, activation, initType, numChannels, batchnorm, dilation,
              hiddenUnits=512, fcSize=4096, softmax=False):
    # Encoder Decoder CNN without LSTM / RNN units
    if modelType == "edCNN_wp":
        from Model import EnDeWithPooling
        model = EnDeWithPooling(activation, initType, numChannels, batchnorm, softmax)
        model.init_weights()
        return model, False

    if modelType == "convLSTM":
        from Model import EnDeConvLSTM
        model = EnDeConvLSTM(activation, initType, numChannels, imageHeight, imageWidth, batchnorm=batchnorm,
                             softmax=softmax)
        model.init_weights()
        return model, True

    if modelType == "convLSTM_ws":
        from Model import EnDeConvLSTM_ws
        model = EnDeConvLSTM_ws(activation, initType, numChannels, imageHeight, imageWidth, batchnorm=batchnorm,
                                softmax=softmax)
        model.init_weights()
        return model, True

    if modelType == "skipLSTM":
        from Model import SkipLSTMEnDe
        model = SkipLSTMEnDe(activation, initType, numChannels, imageHeight, imageWidth, batchnorm=batchnorm,
                             softmax=softmax)
        model.init_weights()
        return model, True

    if modelType == "enDeLayerNorm":
        from Model import EnDeLayerNorm_ws
        model = EnDeLayerNorm_ws(activation, initType, numChannels, imageHeight, imageWidth, softmax=softmax)
        model.init_weights()
        return model, True

    if modelType == "enDeLayerNorm1D":
        from Model import EnDeLayerNorm1D_ws
        model = EnDeLayerNorm1D_ws(activation, initType, numChannels, imageHeight, imageWidth, softmax=softmax)
        model.init_weights()
        return model, True

    if modelType == "skipLayerNorm":
        from Model import SkipLSTMLayerNorm
        model = SkipLSTMLayerNorm(activation, initType, numChannels, imageHeight, imageWidth, softmax=softmax)
        model.init_weights()
        return model, True

    if modelType == "skipLayerNorm1D":
        from Model import SkipLSTMLayerNorm1D
        model = SkipLSTMLayerNorm1D(activation, initType, numChannels, imageHeight, imageWidth, softmax=softmax)
        model.init_weights()
        return model, True

###########################################################################
#####                            MAIN CODE                            #####
###########################################################################

cmd = args.arguments
cmd.channels = []
if cmd.lane != "False":
    cmd.channels.append("lane")
if cmd.obstacles != "False":
    cmd.channels.append("obstacles")
if cmd.road != "False":
    cmd.channels.append("road")
if cmd.vehicles != "False":
    cmd.channels.append("vehicles")

print("Channels Used: ", cmd.channels)
model, isLSTM = loadModel(cmd.modelType, cmd.imageWidth, cmd.imageHeight, cmd.activation, cmd.initType,
                          len(cmd.channels) + 1, cmd.batchnorm, cmd.dilation)

model = model.cuda()

# Make Directory Structure to Save the Models:
baseDir = os.path.dirname(os.path.realpath(__file__))
expDir = os.path.join(baseDir, 'ablation_cache', cmd.modelType, time.strftime("%d_%m_%Y_%H_%M"), cmd.expID)
lossDir = os.path.join(expDir, 'loss')
os.makedirs(expDir, exist_ok=True)
os.makedirs(lossDir, exist_ok=True)

# Save the command line arguments
with open(os.path.join(expDir, 'args.txt'), 'w') as cmdFile:
    for arg in vars(cmd):
        cmdFile.write(arg + ' ' + str(getattr(cmd, arg)) + '\n')

# Loss Function
criterion = nn.MSELoss()

# Optimizer
optimizer = None
if cmd.optMethod == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=cmd.lr, betas=(cmd.beta1, cmd.beta2), weight_decay=cmd.weightDecay)
elif cmd.optMethod == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=cmd.lr, momentum=cmd.momentum, weight_decay=cmd.weightDecay,
                          nesterov=False)
elif cmd.optMethod == 'amsgrad':
    optimizer = optim.Adam(model.parameters(), lr=cmd.lr, betas=(cmd.beta1, cmd.beta2), weight_decay=cmd.weightDecay,
                           amsgrad=True)

# Default CUDA tensor
torch.set_default_tensor_type(torch.cuda.FloatTensor)

# Scale factor to scale the label channel
scf = 1
if cmd.scaleFactor:
    scf = cmd.imageHeight * cmd.imageWidth

if cmd.csvDir is None:
    cmd.csvDir = cmd.dataDir

print("-"*100)
print("Loss: ", cmd.lossFun)
print("Data Dir: ", cmd.dataDir)
print("CSV Dir: ", cmd.csvDir)

trainInfoPath = os.path.join(cmd.csvDir, cmd.trainPath)
trainDataset = KittiDataset(cmd.dataDir, height=cmd.imageHeight, width=cmd.imageWidth, train=True,
                            infoPath=trainInfoPath, augmentation=cmd.augmentation,
                            augmentationProb=cmd.augmentationProb, channels=cmd.channels,
                            groundTruth=cmd.groundTruth)

valInfoPath = os.path.join(cmd.csvDir, cmd.valPath)
valDataset = KittiDataset(cmd.dataDir, height=cmd.imageHeight, width=cmd.imageWidth, train=False,
                          infoPath=valInfoPath, channels=cmd.channels, groundTruth=cmd.groundTruth)

epochTrainLoss = []
epochValidLoss = []

# Saving Model Weights
best_model_weights = copy.deepcopy(model.state_dict())
best_loss = 100000000

# Saving Future Model Weights
best_model_weights_future = copy.deepcopy(model.state_dict())
best_loss_future = 100000000

# Loss History
lossHistory = []

# Train Loss History
trainHistory = []

# Validation History
validationHistory = []

for epoch in range(cmd.nepochs):
    print("-"*100)
    print("Epoch No: {}".format(epoch))
    startTime = time.time()
    model.train()
    # Total Loss of one trajectory
    loss = None
    # Hidden states of the LSTM
    state = None
    # Network Prediction
    out = None
    prevOut = None
    # Number of samples forwarded
    count = 0
    # Apply Loss after every batch only
    labelBatch = None
    outputBatch = None
    weightBatch = None

    # Updated train and val loss after each epoch
    trainLossPerEpoch = []
    validLossPerEpoch = []

    seqLoss = []
    curSeqNum = 0

    # Training Loop
    for i in range(len(trainDataset)):
        if loss is None:
            # First pair to be forwarded, hence zero grad
            model.zero_grad()

        grid, kittiSeqNum, vehicleId, frame1, frame2, endOfSequence, offset, numFrames, augmentation = trainDataset[i]

        # The Last Channel is the target frame and first n - 1 are source frames
        inp = grid[:-1, :].unsqueeze(0).cuda()
        currLabel = grid[-1:, :].unsqueeze(0).cuda()
        # weightMat = weightMatrix(currLabel)
        currOutput = None
        obstacle = None

        if labelBatch is None:
            labelBatch = scf * grid[-1:, :].unsqueeze(0).cuda()
        else:
            labelBatch = torch.cat((labelBatch, (scf * (grid[-1:, :])).unsqueeze(0).cuda()), 0)

        # Pass the future predictions after pre-conditioning the LSTM
        if offset >= int(cmd.futureFrames) and epoch > int(cmd.futureEpochs):
            new_inp = inp.clone().squeeze(0)
            if cmd.minMaxNorm:
                mn, mx = torch.min(prevOut), torch.max(prevOut)
                prevOut = (prevOut - mn) / (mx - mn)
            new_inp[0] = prevOut
            inp = new_inp.unsqueeze(0).cuda()

        if isLSTM:
            if cmd.modelType in ["skipLSTM", "skipLayerNorm1D", "skipLayerNorm"]:
                # 3 LSTMs => 3 hidden states
                out = model.forward(inp, state)
                currOutputMap = out.clone()
                state = (model.h, model.c, model.h1, model.c1, model.h2, model.c2)
            else:
                # Simple LSTM => Only 1 hidden state
                out = model.forward(inp, state)
                currOutputMap = out.clone()
                state = (model.h, model.c)
        else:
            # No LSTM => No hidden state
            # Forward the input and obtain the result
            out = model.forward(inp)
            currOutputMap = out.clone()

        if outputBatch is None:
            outputBatch = out
        else:
            outputBatch = torch.cat((outputBatch, out), 0)

        count += 1
        prevOut = currOutputMap.detach().cpu().squeeze(0).squeeze(0)
        currOutputMap = currOutputMap.detach().cpu().numpy().squeeze(0).squeeze(0)
        currLabel = currLabel.detach().cpu().numpy().squeeze(0).squeeze(0)
        _, dist, predCoordinates = heatmapAccuracy(currOutputMap, currLabel)

        if offset >= int(cmd.futureFrames):
            seqLoss.append(dist)

        if count == cmd.seqLen or endOfSequence is True:
            # Regularization
            l2_reg = None
            for W in model.parameters():
                if l2_reg is None:
                    l2_reg = W.norm(2)
                else:
                    l2_reg = l2_reg + W.norm(2)

            l2_reg = cmd.gamma * l2_reg

            if cmd.lossFun == "default":
                loss = criterion(outputBatch, labelBatch)
                # loss = sum([criterion(outputBatch, labelBatch), l2_reg, obstacleLoss])
            elif cmd.lossFun == "weightedMSE":
                loss = weightedMSE(outputBatch, labelBatch, weightBatch)
                # loss = sum([weightedMSE(outputBatch, labelBatch, weightBatch), l2_reg, obstacleLoss])

            if isLSTM:
                if endOfSequence is True:
                    loss.backward()
                else:
                    loss.backward(retain_graph=True)
            else:
                loss.backward()

            if cmd.gradClip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cmd.gradClip)

            optimizer.step()

            # Reset
            loss = None
            labelBatch = None
            outputBatch = None
            count = 0

            if endOfSequence is True:
                if numFrames >= 60:
                    trainLossPerEpoch.append(np.mean(seqLoss))
                    print("kittiSeq: {}, vehicleId: {}, trainSeqNo: {}, numFrames: {}, Augmentation: {}, Seq Loss: {}".format(
                        kittiSeqNum, vehicleId, curSeqNum, numFrames, augmentation, np.mean(seqLoss)))
                    lossHistory.append(["Training", kittiSeqNum, vehicleId, curSeqNum, augmentation, np.mean(seqLoss)])

                curSeqNum += 1
                if isLSTM:
                    state = None
                seqLoss = []

    print("Average train loss: ", np.mean(trainLossPerEpoch))
    print("For training : --- %s seconds ---" % (time.time() - startTime))

    if np.mean(trainLossPerEpoch) >= 15:
        print("Params Value")
        for name, param in model.named_parameters():
            print("Name: ", name)
            print("Grad: ", param.grad.data.norm(2.))

    epochTrainLoss.append(np.mean(trainLossPerEpoch))
    trainHistory.append([epoch, epochTrainLoss[-1]])

    # Validation
    startTime = time.time()
    state = None
    model.eval()
    seqLoss = []
    curSeqNum = 0
    for i in range(len(valDataset)):
        grid, kittiSeqNum, vehicleId, frame1, frame2, endOfSequence, offset, numFrames, augmentation = valDataset[i]

        # The Last Channel is the target frame and first n - 1 are source frames
        inp = grid[:-1, :].unsqueeze(0).cuda()
        label = grid[-1:, :].unsqueeze(0).cuda()

        if offset >= int(cmd.futureFrames) and epoch > int(cmd.futureEpochs):
            new_inp = inp.clone()
            new_inp = new_inp.squeeze(0)
            if cmd.minMaxNorm:
                mn, mx = torch.min(prevOut), torch.max(prevOut)
                prevOut = (prevOut - mn) / (mx - mn)
            new_inp[0] = prevOut
            inp = new_inp.unsqueeze(0).cuda()

        if isLSTM:
            if cmd.modelType in ["skipLSTM", "skipLayerNorm1D", "skipLayerNorm"]:
                out = model.forward(inp, state)
                currOutputMap = out.clone()
                state = (model.h, model.c, model.h1, model.c1, model.h2, model.c2)
            else:
                out = model.forward(inp, state)
                currOutputMap = out.clone()
                state = (model.h, model.c)
        else:
            out = model.forward(inp)
            currOutputMap = out.clone()

        prevOut = currOutputMap.detach().cpu().squeeze(0).squeeze(0)
        outputMap = out.detach().cpu().numpy().squeeze(0).squeeze(0)
        labelMap = label.detach().cpu().numpy().squeeze(0).squeeze(0)
        _, dist, predCoordinates = heatmapAccuracy(outputMap, labelMap)

        if offset >= int(cmd.futureFrames):
            seqLoss.append(dist)

        if endOfSequence:
            state = None
            if offset >= int(cmd.futureFrames):
                if numFrames >= 60:
                    validLossPerEpoch.append(np.mean(seqLoss))
                    print("kittiSeq: {}, vehicleId: {}, valSeqNo: {}, numFrames: {}, Augmentation: {}, Seq Loss: {}".format(
                        kittiSeqNum, vehicleId, curSeqNum, numFrames, augmentation, np.mean(seqLoss)))
                    lossHistory.append(["Validation", kittiSeqNum, vehicleId, curSeqNum, augmentation, np.mean(seqLoss)])
            seqLoss = []
            curSeqNum += 1

    avgValidLoss = np.mean(validLossPerEpoch)
    print("Average valid loss: ", avgValidLoss)
    epochValidLoss.append(avgValidLoss)
    validationHistory.append([epoch, epochValidLoss[-1]])

    if avgValidLoss < best_loss:
        best_loss = avgValidLoss
        best_model_weights = copy.deepcopy(model.state_dict())
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": epochTrainLoss[-1],
            "valid_loss": best_loss
        }
        torch.save(checkpoint, os.path.join(expDir, 'checkpoint.tar'))
        torch.save(model, os.path.join(expDir, 'model.pth'))

    if epoch > int(cmd.futureEpochs):
        if avgValidLoss < best_loss_future:
            best_loss_future = avgValidLoss
            best_model_weights_future = copy.deepcopy(model.state_dict())
            checkpoint_future = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": epochTrainLoss[-1],
                "valid_loss": best_loss_future
            }
            torch.save(checkpoint_future, os.path.join(expDir, 'checkpoint_future.tar'))
            torch.save(model, os.path.join(expDir, 'model_future.pth'))

    print("For Validation: --- %s seconds ---" % (time.time() - startTime))

    if epoch % 5 == 0:
        fig, ax = plt.subplots(1)
        ax.plot(range(len(epochTrainLoss)), epochTrainLoss, 'r', label='Train Loss')
        ax.plot(range(len(epochValidLoss)), epochValidLoss, 'g', label='Valid Loss')
        ax.legend()
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        fig.savefig(os.path.join(expDir, 'loss', 'loss_epoch'))
        plt.close()

print("Best Validation Loss: ", best_loss)

# Plotting
fig, ax = plt.subplots(1)
ax.plot(range(len(epochTrainLoss)), epochTrainLoss, 'r', label='Train Loss')
ax.plot(range(len(epochValidLoss)), epochValidLoss, 'g', label='Valid Loss')
ax.legend()
plt.ylabel('Loss')
plt.xlabel('Epochs')
fig.savefig(os.path.join(expDir, 'loss', 'loss_epoch'))
plt.close()

lossHistoryPath = os.path.join(expDir, 'loss', 'history.csv')
lossHistory.insert(0, ["Type", "kittiSeq", "vehicleId", "valSeqNo", "Augmentation", "Seq Loss"])

with open(lossHistoryPath, "w") as f:
    wr = csv.writer(f)
    wr.writerows(lossHistory)

trainHistoryPath = os.path.join(expDir, 'loss', 'epoch_train.csv')
validationHistoryPath = os.path.join(expDir, 'loss', 'epoch_validation.csv')
trainHistory.insert(0, ["Epoch", "Train Loss"])
validationHistory.insert(0, ["Epoch", "Validation Loss"])

with open(trainHistoryPath, "w") as f:
    wr = csv.writer(f)
    wr.writerows(trainHistory)

with open(validationHistoryPath, "w") as f:
    wr = csv.writer(f)
    wr.writerows(validationHistory)

def writeCSV(filePath, dataList):
    with open(filePath, "w") as f:
        wr = csv.writer(f)
        wr.writerows(dataList)

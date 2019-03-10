import os
import pandas as pd
import numpy as np

import torch
import torchvision
import torchvision.transforms.functional as TF

from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class BaselineDataset(Dataset):
    def __init__(self, KITTIBaseDir, height=256, width=256, train=True, infoPath=None, augmentation=False,
                 augmentationProb=0.3, channels=None, groundTruth=False):
        self.baseDir = KITTIBaseDir

        # Path to disparity directory
        self.obstacleDir = os.path.join(self.baseDir, 'obstacles')
        # Path to lane directory
        self.laneDir = os.path.join(self.baseDir, 'lane')
        # Path to road directory
        self.roadDir = os.path.join(self.baseDir, 'road')
        # Path to target directory
        self.targetDir = os.path.join(self.baseDir, 'target')
        # Path to vehicles directory
        self.vehiclesDir = os.path.join(self.baseDir, 'vehicles')
        # Target GT refers to the occupancy grid of the target vehicle id computed using ground truth
        self.targetGTDir = os.path.join(self.baseDir, 'targetGT')
        # Target GT but not gaussian data
        self.targetGTNonGaussianDir = os.path.join(self.baseDir, 'non-gaussian')
        # The rgb occupany map directory
        self.rgbDir = os.path.join(self.baseDir, 'rgbGrid')

        self.height, self.width = height, width

        # train = True if train dataset, else train = False
        self.train = train

        self.transform = transforms.Compose([
            transforms.Resize(self.height),
            transforms.ToTensor()
        ])

        # Affine Transformation Parameters
        self.horizontalShift = 0
        self.verticalShift = 0

        # augmentation = True if there is dataset augmentation
        self.augmentation = augmentation

        # Augmentation Probability
        self.augmentationProb = augmentationProb

        # True if we are using ground truth data
        self.groundTruth = groundTruth

        # Channels to Use
        self.channels = channels

        # Path to the dataset info / csv file
        self.infoPath = infoPath

        self.train_df = pd.read_csv(self.infoPath, sep=' ', names=['kittiSequence', 'vehicleId',
                                                                   'startFrame', 'endFrame', 'numFrames'])

        # Length of the Pandas Data Frame
        self.dataFrameLen = len(self.train_df)

        # Length of dataset
        self.len = int(self.train_df['numFrames'].sum()) - self.dataFrameLen

        # Add starting and ending indexes column to the data frame
        self.train_df['startIndex'] = np.zeros((self.dataFrameLen, 1))
        self.train_df['endIndex'] = np.zeros((self.dataFrameLen, 1))

        # List to Map indexes to the corresponding vehicle
        self.indexToVehicle = np.ones((self.len, 1))

        # Updating indexToVehicle and the dataFrame
        curIdx = 0
        for row in range(self.dataFrameLen):
            cur_frame = self.train_df.loc[row]
            startFrame = cur_frame['startFrame']
            endFrame = cur_frame['endFrame']
            seqLength = endFrame - startFrame

            startIdx = int(curIdx)
            endIdx = int(curIdx + seqLength - 1)

            self.train_df.loc[row, 'startIndex'] = startIdx
            self.train_df.loc[row, 'endIndex'] = endIdx
            curIdx = endIdx + 1

            self.indexToVehicle[startIdx:curIdx, 0] = row

    def __len__(self):
        return self.len

    def affineTransformParams(self):
        # Return default params if value of prob greater than augmentationProb
        prob = np.random.random()
        horizontal_shift = 0
        vertical_shift = 0
        if prob < self.augmentationProb and self.augmentation:
            # horizontal_shift = np.random.randint(- int(self.width * 0.2), int(self.width * 0.2))
            vertical_shift = np.random.randint(- int(self.height * 0.2), int(self.height * 0.2))

        return horizontal_shift, vertical_shift

    def __getitem__(self, idx):
        row = int(self.indexToVehicle[idx, 0])
        # Get vehicle Id
        vehicleId = self.train_df.loc[row, 'vehicleId']
        # Get the kitti sequence no
        kittiSeqNum = self.train_df.loc[row, 'kittiSequence']
        # Get the num of kitti frames
        numFrames = self.train_df.loc[row, 'numFrames']
        # Get the Current frame
        offset = idx - self.train_df.loc[row, 'startIndex']
        frame1 = int(self.train_df.loc[row, 'startFrame'] + offset)
        frame2 = int(frame1 + 1)

        # Load image for current frame
        curLaneImg = Image.open(os.path.join(self.laneDir, str(kittiSeqNum).zfill(4),
                                            str(frame1).zfill(6)+'.png'))
        curRoadImg = Image.open(os.path.join(self.roadDir, str(kittiSeqNum).zfill(4),
                                            str(frame1).zfill(6)+'.png'))
        curObstacleImg = Image.open(os.path.join(self.obstacleDir, str(kittiSeqNum).zfill(4),
                                                str(frame1).zfill(6)+'.png'))
        curTargetImg = Image.open(os.path.join(self.targetGTDir, str(kittiSeqNum).zfill(4),
                                              str(frame1).zfill(6), str(vehicleId).zfill(6)+'.png'))
        curVehiclesImg = Image.open(os.path.join(self.vehiclesDir, str(kittiSeqNum).zfill(4),
                                                str(frame1).zfill(6), str(vehicleId).zfill(6)+'.png'))
        rgbImage = Image.open(os.path.join(self.rgbDir, str(kittiSeqNum).zfill(4),
                                                str(frame1).zfill(6)+'.png'))

        # Load image for next frame
        nextTargetImg = Image.open(os.path.join(self.targetDir, str(kittiSeqNum).zfill(4),
                                                str(frame2).zfill(6), str(vehicleId).zfill(6) + '.png'))
        if self.groundTruth:
            nextTargetImg = Image.open(os.path.join(self.targetGTDir, str(kittiSeqNum).zfill(4),
                                                    str(frame2).zfill(6), str(vehicleId).zfill(6) + '.png'))

        # Apply Affine Transforms
        if self.train:
            degree = 0
            curLaneImg = TF.affine(curLaneImg, degree, (self.horizontalShift, self.verticalShift),
                                   1, 0, fillcolor=0)
            curRoadImg = TF.affine(curRoadImg, degree, (self.horizontalShift, self.verticalShift),
                                   1, 0, fillcolor=0)
            curObstacleImg = TF.affine(curObstacleImg, degree, (self.horizontalShift, self.verticalShift),
                                   1, 0, fillcolor=0)
            curTargetImg = TF.affine(curTargetImg, degree, (self.horizontalShift, self.verticalShift),
                                   1, 0, fillcolor=0)
            curVehiclesImg = TF.affine(curVehiclesImg, degree, (self.horizontalShift, self.verticalShift),
                                   1, 0, fillcolor=0)
            nextTargetImg = TF.affine(nextTargetImg, degree, (self.horizontalShift, self.verticalShift),
                                   1, 0, fillcolor=0)

        # Apply simple torchvision transforms
        curLaneTensor = self.transform(curLaneImg)
        curRoadTensor = self.transform(curRoadImg)
        curObstacleTensor = self.transform(curObstacleImg)
        curVehiclesTensor = self.transform(curVehiclesImg)
        curTargetTensor = self.transform(curTargetImg)
        nextTargetTensor = self.transform(nextTargetImg)
        rgbTensor = self.transform(rgbImage)

        inpTensor = curTargetTensor

        # Concatenating the channels:
        inpTensor = torch.cat((inpTensor, rgbTensor), dim=0)
        inpTensor = torch.cat((inpTensor, nextTargetTensor), dim=0)

        endOfSequence = False
        if frame2 == self.train_df.loc[row, 'endFrame']:
            self.horizontalShift, self.verticalShift = self.affineTransformParams()
            endOfSequence = True

        augmentation = True # Default Value if self.augmentation is True
        if self.horizontalShift == 0 and self.verticalShift == 0:
            augmentation = False

        return inpTensor, kittiSeqNum, vehicleId, frame1, frame2, endOfSequence, offset, numFrames, augmentation
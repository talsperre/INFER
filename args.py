import argparse
parser = argparse.ArgumentParser()


# Custom FloatRange class, to check for float argument ranges
class FloatRange(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __eq__(self, other):
        return self.start <= other <= self.end

################ Model Options ################################
parser.add_argument('-initType', help='Weight initialization for the linear layers', type=str.lower,
                    choices=['xavier','default'], default='default')
parser.add_argument('-activation', help='Activation function to be used', type=str.lower,
                    choices=['relu', 'selu'], default='relu')
parser.add_argument('-imageWidth', help='Width of the input image', type=int, default=64)
parser.add_argument('-imageHeight', help='Height of the input image', type=int, default=64)
parser.add_argument('-modelType', help='Model definition', choices=['edCNN_wp', 'convLSTM', 'convLSTM_ws', 'skipLSTM',
                                                                    'enDeLayerNorm', 'enDeLayerNorm1D',
                                                                    'skipLayerNorm', 'skipLayerNorm1D'])
parser.add_argument('-dilation', help='Dilation for convolutions', default=True)
parser.add_argument('-lossOT', help='Penalize obstacle and trajectory', default=False)
parser.add_argument('-usePrev', help='Use previous prediction to predict next, in LSTM', default=False)
parser.add_argument('-lane', help='Use lane channel', default=True)
parser.add_argument('-obstacles', help='Use obstacles channel', default=True)
parser.add_argument('-road', help='Use road channel', default=True)
parser.add_argument('-vehicles', help='Use vehicles channel', default=True)
parser.add_argument('-numChannels', '--list', action='append', help='Num Channels used',
                    type=str.lower, default=4)
parser.add_argument('-resume_path', help='Resume from a checkpoint', default=None)

################### Hyperparameters ###########################
parser.add_argument('-lr', help='Learning rate', type=float, default=1e-4)
parser.add_argument('-momentum', help='Momentum', type=float, default=0.9)
parser.add_argument('-weightDecay', help='Weight decay', type=float, default=0.)
parser.add_argument('-lrDecay', help='Learning rate decay factor', type=float, default=0.)
parser.add_argument('-nepochs', help='Number of iterations after loss is to be computed',
                    type=int, default=1)
parser.add_argument('-beta1', help='beta1 for ADAM optimizer', type=float, default=0.9)
parser.add_argument('-beta2', help='beta2 for ADAM optimizer', type=float, default=0.999)
parser.add_argument('-gradClip', help='Max allowed magnitude for the gradient norm, \
    if gradient clipping is to be performed. (Recommended: 1.0)', type=float, default=None)
parser.add_argument('-optMethod', help='Optimization method : adam | sgd |amsgrad ',
                    type=str.lower, choices=['adam', 'sgd', 'amsgrad'], default='adam')
parser.add_argument('-batchnorm', help='Use batchnorm', default=False)
parser.add_argument('-seqLen', help='backprop after this min seq len', type=int, default=10)
parser.add_argument('-lossFun', help='Loss Function: default | dice | weightedmse | all',
                    type=str.lower, choices=['dice', 'default', 'weightedmse', 'all'], default='default')
parser.add_argument('-scaleFactor', help='Use Scaling Factor', default=True)
parser.add_argument('-softmax', help='Use Softmax', default=False)
parser.add_argument('-gamma', help = 'For L2 regularization', type=float, default=0.0)
parser.add_argument('-futureEpochs', help = 'Begin Future Prediction after these many epochs', type=float, default=10.0)
parser.add_argument('-futureFrames', help = 'Begin Future Prediction after these many Frames', type=float, default=20.0)
parser.add_argument('-scheduledSampling', help = 'Use Scheduled Sampling', default=False)
parser.add_argument('-minMaxNorm', help = 'Use Min-Max Normalization', default=True)

################### Dataset ######################################
parser.add_argument('-dataDir', help='Dataset directory')
parser.add_argument('-augmentation', help='Dataset Augmentation', default=False)
parser.add_argument('-augmentationProb', help='Augmentation Probability', type=float, default=0.3)
parser.add_argument('-groundTruth', help='Use Ground Truth for prediction', default=False)
parser.add_argument('-csvDir', help='The train and validation files dir', default=None)
parser.add_argument('-trainPath', help='The Train csv name', default='train3.csv')
parser.add_argument('-valPath', help='The Val csv name', default='val3.csv')


###### Experiments, Snapshots, and Visualization #############
parser.add_argument('-expID', help='experiment ID', default='tmp')

arguments = parser.parse_args()

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ConvLSTM import ConvLSTMCell
from LayerNorm2D import LayerNormConv2d

class EnDeWithPooling(nn.Module):
    def __init__(self, activation, initType, numChannels, batchnorm=False, softmax=False):
        super(EnDeWithPooling, self).__init__()

        self.batchnorm = batchnorm
        self.bias = not batchnorm
        self.initType = initType
        self.activation = None
        self.numChannels = numChannels
        self.softmax = softmax

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = nn.SELU(inplace=True)

        self.conv1 = nn.Conv2d(self.numChannels, 16, 3, 1, 1, bias=self.bias)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1, bias=self.bias)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1, bias=self.bias)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 3, 2, 1, 1)
        self.deconv2 = nn.ConvTranspose2d(32, 16, 3, 2, 1, 1)
        self.deconv1 = nn.ConvTranspose2d(16, 8, 3, 2, 1, 1)
        self.classifier = nn.Conv2d(8, 1, 1)

        self.pool = nn.MaxPool2d(2, 2)
        self.intermediate = nn.Conv2d(64, 64, 1, 1, 0, bias=self.bias)
        self.skip1 = nn.Conv2d(16, 16, 1, 1, 0, bias=self.bias)
        self.skip2 = nn.Conv2d(32, 32, 1, 1, 0, bias=self.bias)

        if self.batchnorm:
            self.bn1 = nn.BatchNorm2d(16)
            self.bn2 = nn.BatchNorm2d(32)
            self.bn3 = nn.BatchNorm2d(64)
            self.bn4 = nn.BatchNorm2d(32)
            self.bn5 = nn.BatchNorm2d(16)
            self.bn6 = nn.BatchNorm2d(8)

    def forward(self, x):
        if self.batchnorm:
            conv1_ = self.pool(self.bn1(self.activation(self.conv1(x))))
            conv2_ = self.pool(self.bn2(self.activation(self.conv2(conv1_))))
            conv3_ = self.pool(self.bn3(self.activation(self.conv3(conv2_))))
            intermediate_ = self.activation(self.intermediate(conv3_))
            skip_deconv3_ = self.deconv3(intermediate_) + self.activation(self.skip2(conv2_))
            deconv3_ = self.bn4(self.activation(skip_deconv3_))
            skip_deconv2_ = self.deconv2(deconv3_) + self.activation(self.skip1(conv1_))
            deconv2_ = self.bn5(self.activation(skip_deconv2_))
            deconv1_ = self.bn6(self.activation(self.deconv1(deconv2_)))
            score = self.classifier(deconv1_)
        else:
            conv1_ = self.pool(self.activation(self.conv1(x)))
            conv2_ = self.pool(self.activation(self.conv2(conv1_)))
            conv3_ = self.pool(self.activation(self.conv3(conv2_)))
            intermediate_ = self.activation(self.intermediate(conv3_))
            skip_deconv3_ = self.deconv3(intermediate_) + self.activation(self.skip2(conv2_))
            deconv3_ = self.activation(skip_deconv3_)
            skip_deconv2_ = self.deconv2(deconv3_) + self.activation(self.skip1(conv1_))
            deconv2_ = self.activation(skip_deconv2_)
            deconv1_ = self.activation(self.deconv1(deconv2_))
            if self.softmax:
                score = F.softmax(self.classifier(deconv1_), dim=1)
            else:
                score = self.classifier(deconv1_)

        return score

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if self.initType == 'default':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, np.sqrt(2. / n))
                elif self.initType == 'xavier':
                    nn.init.xavier_normal_(m.weight.data)

                if m.bias is not None:
                    m.bias.data.zero_()

            if isinstance(m, nn.ConvTranspose2d):
                if self.initType == 'default':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, np.sqrt(2. / n))
                elif self.initType == 'xavier':
                    nn.init.xavier_normal_(m.weight.data)

                if m.bias is not None:
                    m.bias.data.zero_()

class EnDeConvLSTM(nn.Module):
    def __init__(self, activation, initType, numChannels, imageHeight, imageWidth, batchnorm=False, softmax=False):
        super(EnDeConvLSTM, self).__init__()

        self.batchnorm = batchnorm
        self.bias = not self.batchnorm
        self.initType = initType
        self.activation = None
        self.batchsize = 1
        self.numChannels = numChannels
        self.softmax = softmax

        # Encoder
        self.conv1 = nn.Conv2d(self.numChannels, 16, 3, 1, 1, bias=self.bias)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1, bias=self.bias)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1, bias=self.bias)

        # Decoder
        self.deconv3 = nn.ConvTranspose2d(64, 32, 3, 2, 1, 1)
        self.deconv2 = nn.ConvTranspose2d(32, 16, 3, 2, 1, 1)
        self.deconv1 = nn.ConvTranspose2d(16, 8, 3, 2, 1, 1)

        # LSTM
        self.convlstm = ConvLSTMCell((int(imageWidth / 8), int(imageHeight / 8)), 64, 64, (3, 3))
        self.h, self.c = None, None

        # Skip Connections
        self.skip1 = nn.Conv2d(16, 16, 1, 1, 0, bias=self.bias)
        self.skip2 = nn.Conv2d(32, 32, 1, 1, 0, bias=self.bias)

        self.pool = nn.MaxPool2d(2, 2)
        self.classifier = nn.Conv2d(8, 1, 1)

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = nn.SELU(inplace=True)

        if self.batchnorm:
            self.bn1 = nn.BatchNorm2d(16)
            self.bn2 = nn.BatchNorm2d(32)
            self.bn3 = nn.BatchNorm2d(64)
            self.bn4 = nn.BatchNorm2d(32)
            self.bn5 = nn.BatchNorm2d(16)
            self.bn6 = nn.BatchNorm2d(8)

    def forward(self, x, s=None):
        if s is None:
            self.h, self.c = self.convlstm.init_hidden(self.batchsize)
        else:
            self.h, self.c = s

        if self.batchnorm is True:
            # Encoder
            conv1_ = self.pool(self.bn1(self.activation(self.conv1(x))))
            conv2_ = self.pool(self.bn2(self.activation(self.conv2(conv1_))))
            conv3_ = self.pool(self.bn3(self.activation(self.conv3(conv2_))))

            # LSTM
            self.h, self.c = self.convlstm(conv3_, (self.h, self.c))

            # Decoder
            deconv3_ = self.bn4(self.activation(self.deconv3(self.h)) + self.activation(self.skip2(conv2_)))
            deconv2_ = self.bn5(self.activation(self.deconv2(deconv3_)) + self.activation(self.skip1(conv1_)))
            deconv1_ = self.bn6(self.activation(self.deconv1(deconv2_)))

            if self.softmax:
                score = F.softmax(self.classifier(deconv1_), dim=1)
            else:
                score = self.classifier(deconv1_)
        else:
            # Encoder
            conv1_ = self.pool(self.activation(self.conv1(x)))
            conv2_ = self.pool(self.activation(self.conv2(conv1_)))
            conv3_ = self.pool(self.activation(self.conv3(conv2_)))

            # LSTM
            self.h, self.c = self.convlstm(conv3_, (self.h, self.c))

            # Decoder
            deconv3_ = self.activation(self.deconv3(self.h)) + self.activation(self.skip2(conv2_))
            deconv2_ = self.activation(self.deconv2(deconv3_)) + self.activation(self.skip1(conv1_))
            deconv1_ = self.activation(self.deconv1(deconv2_))
            if self.softmax:
                score = F.softmax(self.classifier(deconv1_), dim=1)
            else:
                score = self.classifier(deconv1_)

        return score

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if self.initType == 'default':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, np.sqrt(2. / n))
                elif self.initType == 'xavier':
                    nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                if self.initType == 'default':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, np.sqrt(2. / n))
                elif self.initType == 'xavier':
                    nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, ConvLSTMCell):
                n = m.conv.kernel_size[0] * m.conv.kernel_size[1] * m.conv.out_channels
                m.conv.weight.data.normal_(0, np.sqrt(2. / n))
                if m.conv.bias is not None:
                    m.conv.bias.data.zero_()


class EnDeConvLSTM_ws(nn.Module):
    def __init__(self, activation, initType, numChannels, imageHeight, imageWidth, batchnorm=False, softmax=False):
        super(EnDeConvLSTM_ws, self).__init__()

        self.batchnorm = batchnorm
        self.bias = not self.batchnorm
        self.initType = initType
        self.activation = None
        self.batchsize = 1
        self.numChannels = numChannels
        self.softmax = softmax

        # Encoder
        self.conv1 = nn.Conv2d(self.numChannels, 16, 3, 1, 1, bias=self.bias)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1, bias=self.bias)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1, bias=self.bias)

        # Decoder
        self.deconv3 = nn.ConvTranspose2d(64, 32, 3, 2, 1, 1)
        self.deconv2 = nn.ConvTranspose2d(32, 16, 3, 2, 1, 1)
        self.deconv1 = nn.ConvTranspose2d(16, 8, 3, 2, 1, 1)

        # Conv LSTM
        self.convlstm = ConvLSTMCell((int(imageWidth / 8), int(imageHeight / 8)), 64, 64, (3, 3))
        self.h, self.c = None, None

        self.pool = nn.MaxPool2d(2, 2)
        self.classifier = nn.Conv2d(8, 1, 1)

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = nn.SELU(inplace=True)

        if self.batchnorm:
            self.bn1 = nn.BatchNorm2d(16)
            self.bn2 = nn.BatchNorm2d(32)
            self.bn3 = nn.BatchNorm2d(64)
            self.bn4 = nn.BatchNorm2d(32)
            self.bn5 = nn.BatchNorm2d(16)
            self.bn6 = nn.BatchNorm2d(8)

    def forward(self, x, s=None):
        if s is None:
            self.h, self.c = self.convlstm.init_hidden(self.batchsize)
        else:
            self.h, self.c = s

        if self.batchnorm is True:
            # Encoder
            conv1_ = self.pool(self.bn1(self.activation(self.conv1(x))))
            conv2_ = self.pool(self.bn2(self.activation(self.conv2(conv1_))))
            conv3_ = self.pool(self.bn3(self.activation(self.conv3(conv2_))))

            # LSTM
            self.h, self.c = self.convlstm(conv3_, (self.h, self.c))

            # Decoder
            deconv3_ = self.bn4(self.activation(self.deconv3(self.h)))
            deconv2_ = self.bn5(self.activation(self.deconv2(deconv3_)))
            deconv1_ = self.bn6(self.activation(self.deconv1(deconv2_)))

            if self.softmax:
                score = F.softmax(self.classifier(deconv1_), dim=1)
            else:
                score = self.classifier(deconv1_)
        else:
            # Encoder
            conv1_ = self.pool(self.activation(self.conv1(x)))
            conv2_ = self.pool(self.activation(self.conv2(conv1_)))
            conv3_ = self.pool(self.activation(self.conv3(conv2_)))

            # LSTM
            self.h, self.c = self.convlstm(conv3_, (self.h, self.c))

            # Decoder
            deconv3_ = self.activation(self.deconv3(self.h))
            deconv2_ = self.activation(self.deconv2(deconv3_))
            deconv1_ = self.activation(self.deconv1(deconv2_))

            if self.softmax:
                score = F.softmax(self.classifier(deconv1_), dim=1)
            else:
                score = self.classifier(deconv1_)

        return score

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if self.initType == 'default':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, np.sqrt(2. / n))
                elif self.initType == 'xavier':
                    nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                if self.initType == 'default':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, np.sqrt(2. / n))
                elif self.initType == 'xavier':
                    nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, ConvLSTMCell):
                n = m.conv.kernel_size[0] * m.conv.kernel_size[1] * m.conv.out_channels
                m.conv.weight.data.normal_(0, np.sqrt(2. / n))
                if m.conv.bias is not None:
                    m.conv.bias.data.zero_()


class SkipLSTMEnDe(nn.Module):
    def __init__(self, activation, initType, numChannels, imageHeight, imageWidth, batchnorm=False, softmax=False):
        super(SkipLSTMEnDe, self).__init__()

        self.batchnorm = batchnorm
        self.bias = not self.batchnorm
        self.initType = initType
        self.activation = None
        self.batchsize = 1
        self.numChannels = numChannels
        self.softmax = softmax

        # Encoder
        self.conv1 = nn.Conv2d(self.numChannels, 16, 3, 1, 1, bias=self.bias)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1, bias=self.bias)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1, bias=self.bias)

        # Decoder
        self.deconv3 = nn.ConvTranspose2d(64, 32, 3, 2, 1, 1)
        self.deconv2 = nn.ConvTranspose2d(32, 16, 3, 2, 1, 1)
        self.deconv1 = nn.ConvTranspose2d(16, 8, 3, 2, 1, 1)

        # LSTM
        self.convlstm = ConvLSTMCell((int(imageWidth / 8), int(imageHeight / 8)), 64, 64, (3, 3))
        self.h, self.c = None, None

        # Skip Connections LSTM
        self.skip1 = ConvLSTMCell((int(imageWidth / 2), int(imageHeight / 2)), 16, 16, (3, 3))
        self.h1, self.c1 = None, None
        self.skip2 = ConvLSTMCell((int(imageWidth / 4), int(imageHeight / 4)), 32, 32, (3, 3))
        self.h2, self.c2 = None, None

        self.pool = nn.MaxPool2d(2, 2)
        self.classifier = nn.Conv2d(8, 1, 1)

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = nn.SELU(inplace=True)

        if self.batchnorm:
            self.bn1 = nn.BatchNorm2d(16)
            self.bn2 = nn.BatchNorm2d(32)
            self.bn3 = nn.BatchNorm2d(64)
            self.bn4 = nn.BatchNorm2d(32)
            self.bn5 = nn.BatchNorm2d(16)
            self.bn6 = nn.BatchNorm2d(8)

    def forward(self, x, s=None):
        if s is None:
            self.h, self.c = self.convlstm.init_hidden(self.batchsize)
            self.h1, self.c1 = self.skip1.init_hidden(self.batchsize)
            self.h2, self.c2 = self.skip2.init_hidden(self.batchsize)
        else:
            self.h, self.c, self.h1, self.c1, self.h2, self.c2 = s

        if self.batchnorm is True:
            # Encoder
            conv1_ = self.pool(self.bn1(self.activation(self.conv1(x))))
            conv2_ = self.pool(self.bn2(self.activation(self.conv2(conv1_))))
            conv3_ = self.pool(self.bn3(self.activation(self.conv3(conv2_))))

            # LSTM
            self.h, self.c = self.convlstm(conv3_, (self.h, self.c))
            self.h2, self.c2 = self.skip2(conv2_, (self.h2, self.c2))
            self.h1, self.c1 = self.skip1(conv1_, (self.h1, self.c1))

            # Decoder
            deconv3_ = self.bn4(self.activation(self.deconv3(self.h)) + self.activation(self.h2))
            deconv2_ = self.bn5(self.activation(self.deconv2(deconv3_)) + self.activation(self.h1))
            deconv1_ = self.bn6(self.activation(self.deconv1(deconv2_)))

            if self.softmax:
                score = F.softmax(self.classifier(deconv1_), dim=1)
            else:
                score = self.classifier(deconv1_)
        else:
            # Encoder
            conv1_ = self.pool(self.activation(self.conv1(x)))
            conv2_ = self.pool(self.activation(self.conv2(conv1_)))
            conv3_ = self.pool(self.activation(self.conv3(conv2_)))

            # LSTM
            self.h, self.c = self.convlstm(conv3_, (self.h, self.c))
            self.h2, self.c2 = self.skip2(conv2_, (self.h2, self.c2))
            self.h1, self.c1 = self.skip1(conv1_, (self.h1, self.c1))

            # Decoder
            deconv3_ = self.activation(self.deconv3(self.h)) + self.activation(self.h2)
            deconv2_ = self.activation(self.deconv2(deconv3_)) + self.activation(self.h1)
            deconv1_ = self.activation(self.deconv1(deconv2_))
            if self.softmax:
                score = F.softmax(self.classifier(deconv1_), dim=1)
            else:
                score = self.classifier(deconv1_)

        return score

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if self.initType == 'default':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, np.sqrt(2. / n))
                elif self.initType == 'xavier':
                    nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                if self.initType == 'default':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, np.sqrt(2. / n))
                elif self.initType == 'xavier':
                    nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, ConvLSTMCell):
                n = m.conv.kernel_size[0] * m.conv.kernel_size[1] * m.conv.out_channels
                m.conv.weight.data.normal_(0, np.sqrt(2. / n))
                if m.conv.bias is not None:
                    m.conv.bias.data.zero_()


class EnDeLayerNorm_ws(nn.Module):
    def __init__(self, activation, initType, numChannels, imageHeight, imageWidth, softmax=False):
        super(EnDeLayerNorm_ws, self).__init__()

        self.initType = initType
        self.activation = None
        self.batchsize = 1
        self.numChannels = numChannels
        self.softmax = softmax

        # Encoder
        self.conv1 = nn.Conv2d(self.numChannels, 16, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1, bias=True)

        # Decoder
        self.deconv3 = nn.ConvTranspose2d(64, 32, 3, 2, 1, 1)
        self.deconv2 = nn.ConvTranspose2d(32, 16, 3, 2, 1, 1)
        self.deconv1 = nn.ConvTranspose2d(16, 8, 3, 2, 1, 1)

        # Conv LSTM
        self.convlstm = ConvLSTMCell((int(imageWidth / 8), int(imageHeight / 8)), 64, 64, (3, 3))
        self.h, self.c = None, None

        self.pool = nn.MaxPool2d(2, 2)
        self.classifier = nn.Conv2d(8, 1, 1)

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = nn.SELU(inplace=True)

        self.layerNorm1 = LayerNormConv2d(16)
        self.layerNorm2 = LayerNormConv2d(32)
        self.layerNorm3 = LayerNormConv2d(64)

        self.layerNorm4 = LayerNormConv2d(32)
        self.layerNorm5 = LayerNormConv2d(16)
        self.layerNorm6 = LayerNormConv2d(8)


    def forward(self, x, s=None):
        if s is None:
            self.h, self.c = self.convlstm.init_hidden(self.batchsize)
        else:
            self.h, self.c = s

        # Encoder
        conv1_ = self.pool(self.layerNorm1(self.activation(self.conv1(x))))
        conv2_ = self.pool(self.layerNorm2(self.activation(self.conv2(conv1_))))
        conv3_ = self.pool(self.layerNorm3(self.activation(self.conv3(conv2_))))

        # LSTM
        self.h, self.c = self.convlstm(conv3_, (self.h, self.c))

        # Decoder
        deconv3_ = self.layerNorm4(self.activation(self.deconv3(self.h)))
        deconv2_ = self.layerNorm5(self.activation(self.deconv2(deconv3_)))
        deconv1_ = self.layerNorm6(self.activation(self.deconv1(deconv2_)))

        if self.softmax:
            score = F.softmax(self.classifier(deconv1_), dim=1)
        else:
            score = self.classifier(deconv1_)
        return score

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if self.initType == 'default':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, np.sqrt(2. / n))
                elif self.initType == 'xavier':
                    nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                if self.initType == 'default':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, np.sqrt(2. / n))
                elif self.initType == 'xavier':
                    nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, ConvLSTMCell):
                n = m.conv.kernel_size[0] * m.conv.kernel_size[1] * m.conv.out_channels
                m.conv.weight.data.normal_(0, np.sqrt(2. / n))
                if m.conv.bias is not None:
                    m.conv.bias.data.zero_()

class EnDeLayerNorm1D_ws(nn.Module):
    def __init__(self, activation, initType, numChannels, imageHeight, imageWidth, softmax=False):
        super(EnDeLayerNorm1D_ws, self).__init__()

        self.initType = initType
        self.activation = None
        self.batchsize = 1
        self.numChannels = numChannels
        self.softmax = softmax

        # Encoder
        self.conv1 = nn.Conv2d(self.numChannels, 16, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1, bias=True)

        # Decoder
        self.deconv3 = nn.ConvTranspose2d(64, 32, 3, 2, 1, 1)
        self.deconv2 = nn.ConvTranspose2d(32, 16, 3, 2, 1, 1)
        self.deconv1 = nn.ConvTranspose2d(16, 8, 3, 2, 1, 1)

        # Conv LSTM
        self.convlstm = ConvLSTMCell((int(imageWidth / 8), int(imageHeight / 8)), 64, 64, (3, 3))
        self.h, self.c = None, None

        self.pool = nn.MaxPool2d(2, 2)
        self.classifier = nn.Conv2d(8, 1, 1)

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = nn.SELU(inplace=True)

        self.layerNorm1 = nn.LayerNorm([256, 256])
        self.layerNorm2 = nn.LayerNorm([128, 128])
        self.layerNorm3 = nn.LayerNorm([64, 64])

        self.layerNorm4 = nn.LayerNorm([64, 64])
        self.layerNorm5 = nn.LayerNorm([128, 128])
        self.layerNorm6 = nn.LayerNorm([256, 256])

    def forward(self, x, s=None):
        if s is None:
            self.h, self.c = self.convlstm.init_hidden(self.batchsize)
        else:
            self.h, self.c = s

        # Encoder
        conv1_ = self.pool(self.layerNorm1(self.activation(self.conv1(x))))
        conv2_ = self.pool(self.layerNorm2(self.activation(self.conv2(conv1_))))
        conv3_ = self.pool(self.layerNorm3(self.activation(self.conv3(conv2_))))

        # LSTM
        self.h, self.c = self.convlstm(conv3_, (self.h, self.c))

        # Decoder
        deconv3_ = self.layerNorm4(self.activation(self.deconv3(self.h)))
        deconv2_ = self.layerNorm5(self.activation(self.deconv2(deconv3_)))
        deconv1_ = self.layerNorm6(self.activation(self.deconv1(deconv2_)))

        if self.softmax:
            score = F.softmax(self.classifier(deconv1_), dim=1)
        else:
            score = self.classifier(deconv1_)
        return score

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if self.initType == 'default':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, np.sqrt(2. / n))
                elif self.initType == 'xavier':
                    nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                if self.initType == 'default':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, np.sqrt(2. / n))
                elif self.initType == 'xavier':
                    nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, ConvLSTMCell):
                n = m.conv.kernel_size[0] * m.conv.kernel_size[1] * m.conv.out_channels
                m.conv.weight.data.normal_(0, np.sqrt(2. / n))
                if m.conv.bias is not None:
                    m.conv.bias.data.zero_()


class SkipLSTMLayerNorm(nn.Module):
    def __init__(self, activation, initType, numChannels, imageHeight, imageWidth, softmax=False):
        super(SkipLSTMLayerNorm, self).__init__()

        self.initType = initType
        self.activation = None
        self.batchsize = 1
        self.numChannels = numChannels
        self.softmax = softmax

        # Encoder
        self.conv1 = nn.Conv2d(self.numChannels, 16, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1, bias=True)

        # Decoder
        self.deconv3 = nn.ConvTranspose2d(64, 32, 3, 2, 1, 1)
        self.deconv2 = nn.ConvTranspose2d(32, 16, 3, 2, 1, 1)
        self.deconv1 = nn.ConvTranspose2d(16, 8, 3, 2, 1, 1)

        # LSTM
        self.convlstm = ConvLSTMCell((int(imageWidth / 8), int(imageHeight / 8)), 64, 64, (3, 3))
        self.h, self.c = None, None

        # Skip Connections LSTM
        self.skip1 = ConvLSTMCell((int(imageWidth / 2), int(imageHeight / 2)), 16, 16, (3, 3))
        self.h1, self.c1 = None, None
        self.skip2 = ConvLSTMCell((int(imageWidth / 4), int(imageHeight / 4)), 32, 32, (3, 3))
        self.h2, self.c2 = None, None

        self.pool = nn.MaxPool2d(2, 2)
        self.classifier = nn.Conv2d(8, 1, 1)

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = nn.SELU(inplace=True)

        self.layerNorm1 = LayerNormConv2d(16)
        self.layerNorm2 = LayerNormConv2d(32)
        self.layerNorm3 = LayerNormConv2d(64)

        self.layerNorm4 = LayerNormConv2d(32)
        self.layerNorm5 = LayerNormConv2d(16)
        self.layerNorm6 = LayerNormConv2d(8)


    def forward(self, x, s=None):
        if s is None:
            self.h, self.c = self.convlstm.init_hidden(self.batchsize)
            self.h1, self.c1 = self.skip1.init_hidden(self.batchsize)
            self.h2, self.c2 = self.skip2.init_hidden(self.batchsize)
        else:
            self.h, self.c, self.h1, self.c1, self.h2, self.c2 = s

        # Encoder
        conv1_ = self.pool(self.layerNorm1(self.activation(self.conv1(x))))
        conv2_ = self.pool(self.layerNorm2(self.activation(self.conv2(conv1_))))
        conv3_ = self.pool(self.layerNorm3(self.activation(self.conv3(conv2_))))

        # LSTM
        self.h, self.c = self.convlstm(conv3_, (self.h, self.c))
        self.h2, self.c2 = self.skip2(conv2_, (self.h2, self.c2))
        self.h1, self.c1 = self.skip1(conv1_, (self.h1, self.c1))

        # Decoder
        deconv3_ = self.layerNorm4(self.activation(self.deconv3(self.h)) + self.activation(self.h2))
        deconv2_ = self.layerNorm5(self.activation(self.deconv2(deconv3_)) + self.activation(self.h1))
        deconv1_ = self.layerNorm6(self.activation(self.deconv1(deconv2_)))
        if self.softmax:
            score = F.softmax(self.classifier(deconv1_), dim=1)
        else:
            score = self.classifier(deconv1_)

        return score

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if self.initType == 'default':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, np.sqrt(2. / n))
                elif self.initType == 'xavier':
                    nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                if self.initType == 'default':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, np.sqrt(2. / n))
                elif self.initType == 'xavier':
                    nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, ConvLSTMCell):
                n = m.conv.kernel_size[0] * m.conv.kernel_size[1] * m.conv.out_channels
                m.conv.weight.data.normal_(0, np.sqrt(2. / n))
                if m.conv.bias is not None:
                    m.conv.bias.data.zero_()


class SkipLSTMLayerNorm1D(nn.Module):
    def __init__(self, activation, initType, numChannels, imageHeight, imageWidth, softmax=False):
        super(SkipLSTMLayerNorm1D, self).__init__()

        self.initType = initType
        self.activation = None
        self.batchsize = 1
        self.numChannels = numChannels
        self.softmax = softmax

        # Encoder
        self.conv1 = nn.Conv2d(self.numChannels, 16, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1, bias=True)

        # Decoder
        self.deconv3 = nn.ConvTranspose2d(64, 32, 3, 2, 1, 1)
        self.deconv2 = nn.ConvTranspose2d(32, 16, 3, 2, 1, 1)
        self.deconv1 = nn.ConvTranspose2d(16, 8, 3, 2, 1, 1)

        # LSTM
        self.convlstm = ConvLSTMCell((int(imageWidth / 8), int(imageHeight / 8)), 64, 64, (3, 3))
        self.h, self.c = None, None

        # Skip Connections LSTM
        self.skip1 = ConvLSTMCell((int(imageWidth / 2), int(imageHeight / 2)), 16, 16, (3, 3))
        self.h1, self.c1 = None, None
        self.skip2 = ConvLSTMCell((int(imageWidth / 4), int(imageHeight / 4)), 32, 32, (3, 3))
        self.h2, self.c2 = None, None

        self.pool = nn.MaxPool2d(2, 2)
        self.classifier = nn.Conv2d(8, 1, 1)

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = nn.SELU(inplace=True)

        self.layerNorm1 = nn.LayerNorm([256, 256])
        self.layerNorm2 = nn.LayerNorm([128, 128])
        self.layerNorm3 = nn.LayerNorm([64, 64])

        self.layerNorm4 = nn.LayerNorm([64, 64])
        self.layerNorm5 = nn.LayerNorm([128, 128])
        self.layerNorm6 = nn.LayerNorm([256, 256])


    def forward(self, x, s=None):
        if s is None:
            self.h, self.c = self.convlstm.init_hidden(self.batchsize)
            self.h1, self.c1 = self.skip1.init_hidden(self.batchsize)
            self.h2, self.c2 = self.skip2.init_hidden(self.batchsize)
        else:
            self.h, self.c, self.h1, self.c1, self.h2, self.c2 = s

        # Encoder
        conv1_ = self.pool(self.layerNorm1(self.activation(self.conv1(x))))
        conv2_ = self.pool(self.layerNorm2(self.activation(self.conv2(conv1_))))
        conv3_ = self.pool(self.layerNorm3(self.activation(self.conv3(conv2_))))

        # LSTM
        self.h, self.c = self.convlstm(conv3_, (self.h, self.c))
        self.h2, self.c2 = self.skip2(conv2_, (self.h2, self.c2))
        self.h1, self.c1 = self.skip1(conv1_, (self.h1, self.c1))

        # Decoder
        deconv3_ = self.layerNorm4(self.activation(self.deconv3(self.h)) + self.activation(self.h2))
        deconv2_ = self.layerNorm5(self.activation(self.deconv2(deconv3_)) + self.activation(self.h1))
        deconv1_ = self.layerNorm6(self.activation(self.deconv1(deconv2_)))
        if self.softmax:
            score = F.softmax(self.classifier(deconv1_), dim=1)
        else:
            score = self.classifier(deconv1_)

        return score

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if self.initType == 'default':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, np.sqrt(2. / n))
                elif self.initType == 'xavier':
                    nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                if self.initType == 'default':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, np.sqrt(2. / n))
                elif self.initType == 'xavier':
                    nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, ConvLSTMCell):
                n = m.conv.kernel_size[0] * m.conv.kernel_size[1] * m.conv.out_channels
                m.conv.weight.data.normal_(0, np.sqrt(2. / n))
                if m.conv.bias is not None:
                    m.conv.bias.data.zero_()
"""
Implements a Convolutional LSTM
Inspired from the elegant implementation available here.
https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py
"""

import torch
from torch.autograd import Variable
import torch.nn as nn


class ConvLSTMCell(nn.Module):

    def __init__(self, input_shape, c_in, hidden_size, kernel_size):

        """
        Initialize a ConvLSTMCell object

        input_shape: (Width, Height)
        c_in: number of channels in input
        hidden_size: number of channels in hidden layer
        kernel_size: conv kernel dimensions (F1, F2)
        """

        super(ConvLSTMCell, self).__init__()

        self.width, self.height = input_shape
        self.c_in = c_in
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.batch_size = 1

        self.conv = nn.Conv2d(in_channels = self.c_in + self.hidden_size, \
            out_channels = 4 * self.hidden_size, kernel_size = self.kernel_size, \
            padding = self.padding, bias = True)

        self.h_cur, self.c_cur = self.init_hidden(self.batch_size)

    def forward(self, x_cur, s_cur = None):

        """
        Does a forward pass
        x_cur: input at the current step
        s_cur: state (from the previous step), i.e., (the current state)
                s_cur = (h_cur, c_cur) (h_cur -> output, c_cur -> cellstate)
        """

        if s_cur is not None:
            self.h_cur, self.c_cur = s_cur
        else:
            # Initialize
            self.h_cur, self.c_cur = self.init_hidden(self.batch_size)

        combined = torch.cat([x_cur, self.h_cur], dim=1)

        # Perform conv
        combined_conv_ = self.conv(combined)

        # Split into input, forget, output, and activation gates. Apply non linearities
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv_, self.hidden_size, dim = 1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        # Update state
        c_next = f * self.c_cur + i * g
        h_next = o * torch.tanh(c_next)

        self.h_cur = h_next
        self.c_cur = c_next

        return self.h_cur, self.c_cur

    def init_hidden(self, batch_size):
        return(Variable(torch.zeros(batch_size, self.hidden_size, self.height, self.width)).cuda(), \
            Variable(torch.zeros(batch_size, self.hidden_size, self.height, self.width)).cuda())
#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2022
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as imagemodels
import torch.utils.model_zoo as model_zoo
from torch import Tensor
import numpy as np
import math

class mutlimodal(nn.Module):
    def __init__(self, args):
        super(mutlimodal, self).__init__()
        num_channels = args["acoustic_model"]["out_channels"]
        z_dim = args["audio_model"]["z_dim"]
        c_dim = args["audio_model"]["c_dim"]

        self.conv = nn.Conv1d(
            args["acoustic_model"]["in_channels"], num_channels, 
            args["acoustic_model"]["kernel_size"], 
            args["acoustic_model"]["stride"], 
            args["acoustic_model"]["padding"], bias=False)

        self.encoder = nn.Sequential(
            nn.LayerNorm(num_channels),
            nn.ReLU(True),
            nn.Linear(num_channels, num_channels, bias=False),
            # nn.InstanceNorm1d(num_channels),

            nn.LayerNorm(num_channels),
            nn.ReLU(True),
            nn.Linear(num_channels, num_channels, bias=False),
            # nn.InstanceNorm1d(num_channels),

            nn.LayerNorm(num_channels),
            nn.ReLU(True),
            nn.Linear(num_channels, num_channels, bias=False),
            # nn.InstanceNorm1d(num_channels),

            nn.LayerNorm(num_channels),
            nn.ReLU(True),
            nn.Linear(num_channels, num_channels, bias=False),
            # nn.InstanceNorm1d(num_channels),

            nn.LayerNorm(num_channels),
            nn.ReLU(True),
            nn.Linear(num_channels, z_dim),
            # nn.InstanceNorm1d(z_dim),
        )
        self.rnn1 = nn.LSTM(z_dim, c_dim, batch_first=True)
        self.rnn2 = nn.LSTM(c_dim, c_dim, batch_first=True)
        self.rnn3 = nn.LSTM(c_dim, c_dim, batch_first=True)
        self.rnn4 = nn.LSTM(c_dim, c_dim, batch_first=True)

        self.english_rnn1 = nn.LSTM(512, 512, batch_first=True, bidirectional=True)
        self.english_rnn2 = nn.LSTM(1024, 1024, batch_first=True, bidirectional=True)
        # self.bn1 = nn.BatchNorm1d(512)
        # self.bn2 = nn.BatchNorm1d(512)
        self.relu = nn.ReLU()
        # self.ln = nn.LayerNorm(512)

    def forward(self, mels):
        z = self.conv(mels)
        z = self.relu(z)
        z = self.encoder(z.transpose(1, 2))

        c, _ = self.rnn1(z)
        c, _ = self.rnn2(c)
        c, _ = self.rnn3(c)
        c, _ = self.rnn4(c)

        s, _ = self.english_rnn1(c)
        # s = self.ln(s)
        # s = self.relu(s)
        s, _ = self.english_rnn2(s)
        # s = self.ln(s)
        # s = self.relu(s)

        return z, z, s.transpose(1, 2)

    def encode(self, mels, feat):
        if feat == 'mels': return mels
        z = self.conv(mels)
        z = self.relu(z)
        if feat == 'pre_z': return z
        z = self.encoder(z.transpose(1, 2))
        if feat == 'z': return z

        c, _ = self.rnn1(z)
        if feat == 'c1': return z
        c, _ = self.rnn2(c)
        if feat == 'c2': return z
        c, _ = self.rnn3(c)
        if feat == 'c3': return z
        c, _ = self.rnn4(c)
        if feat == 'c4': return z

        s, _ = self.english_rnn1(c)
        # s = self.ln(s)
        # s = self.relu(s)
        if feat == 's1': return s
        s, _ = self.english_rnn2(s)
        # s = self.ln(s)
        # s = self.relu(s)
        if feat == 's2': return s


# class mutlimodal(nn.Module):
#     def __init__(self, args):
#         super(mutlimodal, self).__init__()
#         z_dim = args["audio_model"]["z_dim"]
#         c_dim = args["audio_model"]["c_dim"]
#         embedding_dim = args["audio_model"]["embedding_dim"]

#         self.conv = nn.Conv1d(
#             args["acoustic_model"]["in_channels"], 256, 
#             args["acoustic_model"]["kernel_size"], 
#             args["acoustic_model"]["stride"], 
#             args["acoustic_model"]["padding"], bias=False)

#         # self.conv2 = nn.Conv2d(128, 256, kernel_size=(1,11), stride=(1,1), padding=(0,5))
#         self.conv3 = nn.Conv2d(256, 512, kernel_size=(1,17), stride=(1,1), padding=(0,8))
#         # self.conv4 = nn.Conv2d(512, 512, kernel_size=(1,17), stride=(1,1), padding=(0,8))
#         self.conv5 = nn.Conv2d(512, embedding_dim, kernel_size=(1,17), stride=(1,1), padding=(0,8))
#         # self.relu = nn.ReLU()
#         # self.batchnorm1 = nn.BatchNorm2d(1)

#     def forward(self, x):

#         # x = self.batchnorm1(x.unsqueeze(1))
#         x = self.conv(x)
#         x = F.relu(self.conv3(x.unsqueeze(2)))
#         x = F.relu(self.conv5(x))
#         x = x.squeeze(2)

#         return x, x, x

#     def encode(self, x, feat):
#         if feat == 'mels': return x
#         # x = self.batchnorm1(x.unsqueeze(1))
#         x = self.conv(x)
#         if feat == 'pre_z': return x
#         if feat == 'z': return x

#         if feat == 'c1': return x
#         if feat == 'c2': return x
#         if feat == 'c3': return x
#         if feat == 'c4': return x

#         x = F.relu(self.conv3(x.unsqueeze(2)))
#         if feat == 's1': return x
#         x = F.relu(self.conv5(x))
#         x = x.squeeze(2)
#         if feat == 's2': return x
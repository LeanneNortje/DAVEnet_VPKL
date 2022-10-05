#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2020
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import math


class AcousticEncoder(nn.Module):
    def __init__(self, args):
        super(AcousticEncoder, self).__init__()
        num_channels = args["acoustic_model"]["out_channels"]
        z_dim = args["audio_model"]["z_dim"]
        c_dim = args["audio_model"]["c_dim"]
        # self.first_norm_layer = nn.InstanceNorm1d(args["acoustic_model"]["in_channels"])
        # self.last_norm_layer = nn.InstanceNorm1d(z_dim)

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

    def encode(self, mels, feat):

        z = self.conv(mels)
        if feat == 'pre_z': return z
        z = self.encoder(z.transpose(1, 2))
        if feat == 'z': return z

        c, _ = self.rnn1(z)
        if feat == 'c1': return c
        c, _ = self.rnn2(c)
        if feat == 'c2': return c
        c, _ = self.rnn3(c)
        if feat == 'c3': return c
        c, _ = self.rnn4(c)
        if feat == 'c4': return c

    def forward(self, mels):
        z = self.conv(mels)
        # z = self.first_norm_layer(z)
        z = self.encoder(z.transpose(1, 2))
        # z = self.last_norm_layer(z)
        c, _ = self.rnn1(z)
        c, _ = self.rnn2(c)
        c, _ = self.rnn3(c)
        c, _ = self.rnn4(c)
        return z, c.transpose(1, 2)

# class AcousticEncoder(nn.Module):
#     def __init__(self, args):
#         super(AcousticEncoder, self).__init__()
#         num_channels = args["acoustic_model"]["out_channels"]
#         z_dim = args["audio_model"]["z_dim"]
#         c_dim = args["audio_model"]["c_dim"]

#         self.conv = nn.Conv1d(
#             args["acoustic_model"]["in_channels"], 1, 
#             args["acoustic_model"]["kernel_size"], 
#             args["acoustic_model"]["stride"], 
#             args["acoustic_model"]["padding"], bias=False)
#         self.encoder = 
#         self.rnn1 = nn.LSTM(z_dim, c_dim, batch_first=True)
#         self.rnn2 = nn.LSTM(c_dim, c_dim, batch_first=True)
#         self.rnn3 = nn.LSTM(c_dim, c_dim, batch_first=True)
#         self.rnn4 = nn.LSTM(c_dim, c_dim, batch_first=True)

#     def encode(self, mels, feat):

#         z = self.conv(mels)
#         if feat == 'pre_z': return z
#         z = self.encoder(z.transpose(1, 2))
#         if feat == 'z': return z

#         c, _ = self.rnn1(z)
#         if feat == 'c1': return c
#         c, _ = self.rnn2(c)
#         if feat == 'c2': return c
#         c, _ = self.rnn3(c)
#         if feat == 'c3': return c
#         c, _ = self.rnn4(c)
#         if feat == 'c4': return c

#     def forward(self, mels):
#         z = self.conv(mels)
#         # z = self.first_norm_layer(z)
#         z = self.encoder(z.transpose(1, 2))
#         # z = self.last_norm_layer(z)
#         c, _ = self.rnn1(z)
#         c, _ = self.rnn2(c)
#         c, _ = self.rnn3(c)
#         c, _ = self.rnn4(c)
#         return z, c.transpose(1, 2)

# class AudioCNN(nn.Module):
#     # function adapted from https://github.com/dharwath

#     def __init__(self, args):
#         super(AudioCNN, self).__init__()
#         self.embedding_dim = args["audio_model"]["embedding_dim"]
#         self.batchnorm1 = nn.BatchNorm2d(1)

#         self.pool = nn.MaxPool2d(
#             kernel_size=tuple(args["audio_model"]["max_pool"]["kernel_size"]), 
#             stride=tuple(args["audio_model"]["max_pool"]["stride"]),
#             padding=tuple(args["audio_model"]["max_pool"]["padding"])
#             )

#         self.layers = torch.nn.Sequential()

#         for i_layer, layer in enumerate(args["audio_model"]["conv_layers"]): 

#             if i_layer == len(args["audio_model"]["conv_layers"]) - 1:
#                 layer["out_channels"] = self.embedding_dim
                
#             self.layers.add_module(
#                 f'conv_{i_layer+1}',
#                 nn.Conv2d(
#                     layer["in_channels"], 
#                     layer["out_channels"], 
#                     kernel_size=(args["audio_model"]["c_dim"], 1) if i_layer == 0 else tuple(layer["kernel_size"]),
#                     stride=tuple(layer["stride"]),
#                     padding=tuple(layer["padding"])
#                     )
#                 )
#             self.layers.add_module(
#                 f'relu_{i_layer+1}', 
#                 nn.ReLU()
#                 )

#             if i_layer + 1 in args["audio_model"]["max_pool"]["after_layers"]:
#                 self.layers.add_module(
#                     f'pool_{i_layer+1}', 
#                     self.pool
#                     )

#     def forward(self, x):
#         if x.dim() == 3:
#             x = x.unsqueeze(1)
#         x = self.batchnorm1(x)
#         x = self.layers(x)
#         # x = x.squeeze(2)
#         return x

#     def encode(self, x, layer_num):
#         layer_map = {
#             1: 1,
#             2: 3,
#             3: 6,
#             4: 9,
#             5: 12,
#             6: 13
#         }
#         if x.dim() == 3:
#             x = x.unsqueeze(1)
#         x = self.batchnorm1(x)
#         for i, layer in enumerate(self.layers):
#             x = layer(x)
#             if i == layer_map[layer_num]:
#                 return x

class AudioCNN(nn.Module):
    # function adapted from https://github.com/dharwath

    def __init__(self, args):
        super(AudioCNN, self).__init__()
        self.embedding_dim = args["audio_model"]["embedding_dim"]
        self.batchnorm1 = nn.BatchNorm2d(1)

        self.pool = nn.MaxPool2d(
            kernel_size=tuple(args["audio_model"]["max_pool"]["kernel_size"]), 
            stride=tuple(args["audio_model"]["max_pool"]["stride"]),
            padding=tuple(args["audio_model"]["max_pool"]["padding"])
            )

        self.layers = torch.nn.Sequential()

        for i_layer, layer in enumerate(args["audio_model"]["conv_layers"]): 

            if i_layer == len(args["audio_model"]["conv_layers"]) - 1:
                layer["out_channels"] = self.embedding_dim
                
            self.layers.add_module(
                f'conv_{i_layer+1}',
                nn.Conv2d(
                    layer["in_channels"], 
                    layer["out_channels"], 
                    kernel_size=tuple(layer["kernel_size"]),
                    stride=tuple(layer["stride"]),
                    padding=tuple(layer["padding"])
                    )
                )
            self.layers.add_module(
                f'relu_{i_layer+1}', 
                nn.ReLU()
                )

            if i_layer + 1 in args["audio_model"]["max_pool"]["after_layers"]:
                self.layers.add_module(
                    f'pool_{i_layer+1}', 
                    self.pool
                    )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.batchnorm1(x)
        x = self.layers(x)
        x = x.squeeze(2)
        return x

    def encode(self, x, layer_num):
        layer_map = {
            1: 1,
            2: 3,
            3: 6,
            4: 9,
            5: 12,
            6: 13
        }
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.batchnorm1(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == layer_map[layer_num]:
                return x

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(0)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)).unsqueeze(1)
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term).transpose(0, 1)
        pe[0, :, 1::2] = torch.cos(position * div_term).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return x

class TransformerModel(nn.Module):
    def __init__(self, args):
        super().__init__()

        c_dim = args["audio_model"]["c_dim"]
        embedding_dim = args["audio_model"]["embedding_dim"]
        self.rnn = nn.LSTM(c_dim, embedding_dim, batch_first=True)
        self.pool = nn.Conv2d(1, 1, (6, 1), (4, 1), (1, 0), bias=False)
        self.relu = nn.ReLU()
        sample_encoder_layer = nn.TransformerEncoderLayer(
            c_dim, args["audio_model"]["num_heads"], dropout=0.0
            )
        self.trm = nn.TransformerEncoder(sample_encoder_layer, 4)
        self.positional_encoder = PositionalEncoding(c_dim, 128)

    def forward(self, c):

        c = c.transpose(1, 2)

        c = c.unsqueeze(1)
        c = self.pool(c)
        c = c.squeeze(1)

        c = self.positional_encoder(c)
        c = self.trm(c)
        c = self.relu(c)

        c, _ = self.rnn(c)
        c = self.relu(c)

        return c.transpose(1, 2)

# class BidrectionalAudioLSTM(nn.Module):
#     def __init__(self, args):
#         super(BidrectionalAudioLSTM, self).__init__()

#         c_dim = args["audio_model"]["c_dim"]
#         num_channels = args["acoustic_model"]["out_channels"]
#         self.conv = nn.Conv1d(
#             args["acoustic_model"]["in_channels"], num_channels, 
#             args["acoustic_model"]["kernel_size"], 
#             args["acoustic_model"]["stride"], 
#             args["acoustic_model"]["padding"], bias=False)

#         self.rnn1 = nn.LSTM(512, 512, batch_first=True, bidirectional=True)
#         self.rnn3 = nn.LSTM(1024, 1024, batch_first=True, bidirectional=True)
#         self.rnn4 = nn.LSTM(2048, 2048, batch_first=True)

#         self.pool = nn.Conv2d(1, 1, (6, 1), (4, 1), (1, 0), bias=False)
#         self.relu = nn.ReLU()


#     def forward(self, c):
#         c = self.conv(c)
#         c = c.transpose(1, 2)

#         c, _ = self.rnn1(c)
#         c = self.relu(c)

#         c, _ = self.rnn3(c)
#         c = self.relu(c)

#         c = c.unsqueeze(1)
#         c = self.pool(c)
#         c = c.squeeze(1)

#         c, _ = self.rnn4(c)
#         c = self.relu(c)

#         return c.transpose(1, 2)

#     def encode(self, c, feat):
#         c = c.transpose(1, 2)

#         c, _ = self.rnn1(c)
#         c = self.relu(c)
#         if feat == 's1': return c

#         c, _ = self.rnn3(c)
#         c = self.relu(c)
#         if feat == 's2': return c

#         c = c.unsqueeze(1)
#         c = self.pool(c)
#         c = c.squeeze(1)
#         if feat == 's3': return c

#         c, _ = self.rnn4(c)
#         c = self.relu(c)
#         if feat == 's4': return c

# class BidrectionalAudioLSTM(nn.Module):
#     def __init__(self, args):
#         super(BidrectionalAudioLSTM, self).__init__()

#         # # num_channels = args["acoustic_model"]["out_channels"]
#         # # self.conv = nn.Conv1d(
#         # #     args["acoustic_model"]["in_channels"], num_channels, 
#         # #     args["acoustic_model"]["kernel_size"], 
#         # #     args["acoustic_model"]["stride"], 
#         # #     args["acoustic_model"]["padding"], bias=False)

#         # c_dim = args["audio_model"]["c_dim"]
#         # embedding_dim = args["audio_model"]["embedding_dim"]
#         # # self.trans = nn.TransformerEncoderLayer(c_dim, 8, batch_first=True)
#         # # self.rnn1 = nn.LSTM(c_dim, embedding_dim//2, batch_first=True, bidirectional=True)
#         # self.linear = nn.Linear(c_dim, embedding_dim)
#         # # self.relu = nn.ReLU()


#     def forward(self, c):
#         # # c = self.conv(c)
#         # # c = self.trans(c)
#         # # c = self.relu(c)
#         # # c, _ = self.rnn1(c)
#         # # c = self.relu(c)
#         # # c, _ = self.rnn2(c)
#         # c = self.linear(c)
#         return c.transpose(1, 2)

#     def encode(self, c, feat):
#         # # c = self.conv(c)
#         # # c = self.trans(c)
#         # # c = self.relu(c)
#         # # c, _ = self.rnn1(c)
#         # # c = self.relu(c)
#         # # c, _ = self.rnn2(c)
#         # c = self.linear(c)
#         return c.transpose(1, 2)

class BidrectionalAudioLSTM(nn.Module):
    def __init__(self, args):
        super(BidrectionalAudioLSTM, self).__init__()

        num_channels = args["acoustic_model"]["out_channels"]
        z_dim = args["audio_model"]["z_dim"]
        c_dim = args["audio_model"]["c_dim"]

        # self.conv = nn.Conv1d(
        #     args["acoustic_model"]["in_channels"], num_channels, 
        #     args["acoustic_model"]["kernel_size"], 
        #     args["acoustic_model"]["stride"], 
        #     args["acoustic_model"]["padding"], bias=False)

        self.rnn1 = nn.LSTM(512, 512, batch_first=True, bidirectional=True)
        self.rnn3 = nn.LSTM(1024, 1024, batch_first=True, bidirectional=True)
        # self.rnn4 = nn.LSTM(2048, 2048, batch_first=True)
        self.relu = nn.ReLU()


    def forward(self, c):
        c = c.transpose(1, 2)
        # c = self.conv(c)

        c, _ = self.rnn1(c)
        # c = self.relu(c)

        c, _ = self.rnn3(c)
        # c = self.relu(c)

        # c, _ = self.rnn4(c)
        # c = self.relu(c)

        return c.transpose(1, 2)

    def encode(self, c, feat):
        c = c.transpose(1, 2)
        # c = self.conv(c)

        c, _ = self.rnn1(c)
        # c = self.relu(c)
        if feat == 's1': return c

        c, _ = self.rnn3(c)
        # c = self.relu(c)
        if feat == 's2': return c


class ResidualCNNLayer(nn.Module):
    # function adapted from https://github.com/wnhsu/ResDAVEnet-VQ

    def __init__(self, in_channels, out_channels, kernel_sizes, strides, padding, downsample_layer=None):
        super(ResidualCNNLayer, self).__init__()

        self.resLayers = torch.nn.Sequential()
        self.in_channels = [in_channels]
        self.in_channels.extend(out_channels[0:-1])
        self.in_channels = in_channels

        for i_layer, kernel_size in enumerate(kernel_sizes):

            self.resLayers.add_module(
                f'conv_{i_layer+1}',
                nn.Conv2d(
                    self.in_channels, out_channels[i_layer], tuple(kernel_size), 
                    tuple(strides[i_layer]), tuple(padding[i_layer])
                    )
                )
            self.resLayers.add_module(f'batch_norm_{i_layer+1}', nn.BatchNorm2d(out_channels[i_layer]))

            if i_layer != len(kernel_sizes) - 1:
                self.resLayers.add_module(f'relu_{i_layer+1}', nn.ReLU(inplace=True))
            
            self.in_channels = out_channels[i_layer]

        self.last_relu = nn.ReLU(inplace=True)

        if downsample_layer is not None:
            self.downsample_layer = torch.nn.Sequential()
            self.downsample_layer.add_module(
                f'downsample_conv',
                nn.Conv2d(
                    downsample_layer["in_channels"], downsample_layer["out_channels"], tuple(downsample_layer["kernel_size"]),
                    tuple(downsample_layer["stride"]), tuple(downsample_layer["padding"]), bias=False
                    )
                )
            self.downsample_layer.add_module(f'downsample_batch_norm', nn.BatchNorm2d(downsample_layer["out_channels"]))
        else: self.downsample_layer = downsample_layer

    def forward(self, x):
        residual = x
        x = self.resLayers(x)
        if self.downsample_layer is not None: residual = self.downsample_layer(residual)
        x += residual
        x = self.last_relu(x)
        return x

class ResidualAudioCNN(nn.Module):
    # function adapted from https://github.com/wnhsu/ResDAVEnet-VQ
    
    def __init__(self, args):
        super(ResidualAudioCNN, self).__init__()
        self.embedding_dim = args["audio_model"]["embedding_dim"]
        self.layers = torch.nn.Sequential()

        conv_layer = 1
        residual_block = 1
        for i_layer, layer in enumerate(args["audio_model"]["conv_layers"]): 

            if "residual" in layer:

                self.layers.add_module(
                    f'residual_block_{residual_block}',
                    ResidualCNNLayer(
                        layer["in_channels"], 
                        layer["out_channels"], 
                        layer["kernel_size"], 
                        layer["stride"], 
                        layer["padding"],
                        layer["downsample_layer"] if "downsample_layer" in layer else None
                        )
                    )
                residual_block += 1
                
            else:
                self.layers.add_module(
                    f'conv_{conv_layer}',
                    nn.Conv2d(
                        layer["in_channels"], 
                        layer["out_channels"], 
                        kernel_size=tuple(layer["kernel_size"]),
                        stride=tuple(layer["stride"]),
                        padding=tuple(layer["padding"])
                        )
                    )
                self.layers.add_module(f'batch_norm_{conv_layer}', nn.BatchNorm2d(layer["out_channels"]))
                self.layers.add_module(f'relu_{conv_layer}', nn.ReLU())
                conv_layer += 1 

    def forward(self, x):

        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.layers(x)
        # x = x.squeeze(2)
        return x

class ResidualBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, num_channels, kernel, stride, padding):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, num_channels, kernel_size=kernel, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=kernel, stride=(1, 1), padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.downsample = None
        if stride[0] != 1 or stride[1] != 1 or in_channels != num_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, num_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_channels),
                )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResDavenet(nn.Module):
    def __init__(self, args):
        
        block=ResidualBlock
        super(ResDavenet, self).__init__()

        self.batchnorm1 = nn.BatchNorm2d(1)
        self.layers = nn.Sequential()

        for i_layer, layer in enumerate(args["audio_model"]["conv_layers"]):
            if layer["type"] == "conv":
                self.layers.add_module(
                    f'layer_{i_layer+1}',
                    nn.Conv2d(
                        layer["in_channels"], layer["out_channels"], kernel_size=tuple(layer["kernel_size"]),
                        stride=tuple(layer["stride"]), padding=tuple(layer["padding"]), bias=False
                        )
                    )
                if layer["batchnorm"]:
                    self.layers.add_module(
                        f'batchnorm_{i_layer+1}',
                        nn.BatchNorm2d(layer["out_channels"])
                        )

                self.layers.add_module(f'relu_{i_layer+1}', nn.ReLU())

            if layer["type"] == "residual":
                for j, sub_layer in enumerate(layer["layers"]):
                    self.layers.add_module(
                        f'residual_layer_{i_layer+1}_{j+1}',
                        block(
                            sub_layer["in_channels"], sub_layer["out_channels"], 
                            tuple(sub_layer["kernel_size"]), tuple(sub_layer["stride"]), tuple(sub_layer["padding"]))
                        )
        

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.layers(x)
        return x

class ChannelNorm(nn.Module):
    def __init__(self, channels, epsilon=1e-05):
        super().__init__()
        self.weight = nn.parameter.Parameter(torch.Tensor(1, channels, 1))
        self.bias = nn.parameter.Parameter(torch.Tensor(1, channels, 1))
        self.epsilon = epsilon

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True)
        x = (x - mean) * torch.rsqrt(var + self.epsilon)
        x = x * self.weight + self.bias
        return x


class Encoder(nn.Module):
    def __init__(self, channels=512):
        super().__init__()

        self.conv0 = nn.Conv1d(40, channels, 11, stride=1, padding=5)
        self.norm0 = ChannelNorm(channels)
        self.conv1 = nn.Conv1d(channels, channels, 4, stride=2, padding=1)
        self.norm1 = ChannelNorm(channels)
        self.conv2 = nn.Conv1d(channels, channels, 4, stride=2, padding=1)
        self.norm2 = ChannelNorm(channels)
        self.conv3 = nn.Conv1d(channels, channels, 4, stride=2, padding=1)
        self.norm3 = ChannelNorm(channels)
        self.conv4 = nn.Conv1d(channels, channels, 4, stride=2, padding=1)
        self.norm4 = ChannelNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.norm0(self.conv0(x)))
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.conv2(x)))
        x = F.relu(self.norm3(self.conv3(x)))
        x = F.relu(self.norm4(self.conv4(x)))
        return x


class Context(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(512, 512, batch_first=True)
        self.lstm2 = nn.LSTM(512, 512, batch_first=True)
        self.lstm3 = nn.LSTM(512, 512, batch_first=True)
        self.lstm4 = nn.LSTM(512, 512, batch_first=True)
        self.h = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x, _ = self.lstm4(x)
        return x

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x, h = self.lstm(x, self.h)
        self.h = h
        return x


class CPC(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.context = Context()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        c = self.context(x.transpose(1, 2))
        return x, c.transpose(1, 2)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x).transpose(1, 2)
        c = self.context.encode(x)
        return c
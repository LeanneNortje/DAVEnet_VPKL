#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2020
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as imagemodels
import torch.utils.model_zoo as model_zoo

class Resnet50(imagemodels.ResNet):
    # function adapted from https://github.com/dharwath
    
    def __init__(self, args, pretrained=True):
        super(Resnet50, self).__init__(imagemodels.resnet.Bottleneck, [3, 4, 6, 3])
        if pretrained:
            self.load_state_dict(model_zoo.load_url(imagemodels.resnet.model_urls['resnet50']))
        self.avgpool = None
        self.fc = None
        embedding_dim = args["audio_model"]["embedding_dim"]
        self.embedder = nn.Conv2d(2048, embedding_dim, kernel_size=1, stride=1, padding=0)
        # self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.embedder(x)
        # x = self.relu(x)
        return x

class Resnet101(imagemodels.ResNet):
    # function adapted from https://github.com/dharwath
    
    def __init__(self, args, pretrained=True):
        super(Resnet101, self).__init__(imagemodels.resnet.Bottleneck, [3, 4, 23, 3])
        if pretrained:
            self.load_state_dict(model_zoo.load_url(imagemodels.resnet.model_urls['resnet101']))
        self.avgpool = None
        self.fc = None
        embedding_dim = args["audio_model"]["embedding_dim"]
        self.embedder = nn.Conv2d(2048, embedding_dim, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.embedder(x)
        # x = self.relu(x)
        return x

class VGG16(nn.Module):
    # function adapted from https://github.com/dharwath

    def __init__(self, args, pretrained=False):
        super(VGG16, self).__init__()
        
        embedding_dim = args["audio_model"]["embedding_dim"]
        seed_model = imagemodels.__dict__['vgg16'](pretrained=pretrained).features
        seed_model = nn.Sequential(*list(seed_model.children())[:-1]) # remove final maxpool
        last_layer_index = len(list(seed_model.children()))
        seed_model.add_module(str(last_layer_index),
            nn.Conv2d(512, embedding_dim, kernel_size=(3,3), stride=(1,1), padding=(1,1)))
        self.image_model = seed_model

    def forward(self, x):
        x = self.image_model(x)
        return x

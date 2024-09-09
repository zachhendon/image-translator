import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import FeaturePyramidNetwork
from torchvision.models.resnet import resnet18
from torchvision.models.feature_extraction import create_feature_extractor
import timm


class ASF(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(256, 256, 3, padding=1)

        self.spatial = nn.Sequential(
            nn.Conv2d(1, 1, 3, bias=False, padding=1),
            nn.ReLU(),
            nn.Conv2d(1, 1, 1, bias=False),
            nn.Sigmoid(),
        )
        self.attention = nn.Sequential(
            nn.Conv2d(256, 1, 1, bias=False), nn.Sigmoid())

    def forward(self, x):
        attention_inp = self.conv(x)
        attention_weights = torch.mean(attention_inp, dim=1, keepdim=True)
        attention_weights = self.spatial(attention_weights) + x
        attention_weights = self.attention(attention_weights)

        fused = x * attention_weights
        return fused


class DBNet(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = resnet18()
        # backbone = resnet18(weights='DEFAULT')
        # backbone = timm.create_model('resnet18', pretrained=True)
        return_nodes = {
            "layer1": "layer1",
            "layer2": "layer2",
            "layer3": "layer3",
            "layer4": "layer4",
        }
        self.body = create_feature_extractor(backbone, return_nodes)

        self.fpn = FeaturePyramidNetwork([64, 128, 256, 512], 64)
        self.asf = ASF()

        self.prob = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 2, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 2, 2),
            nn.Sigmoid(),
        )
        # self.thresh = nn.Sequential(
        #     nn.Conv2d(256, 64, 3, padding=1, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(64, 64, 2, 2),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(64, 1, 2, 2),
        #     nn.Sigmoid(),
        self.thresh = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            self._init_upsample(64, 64, smooth=False, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            self._init_upsample(64, 1, smooth=False, bias=False),
            nn.Sigmoid())

    def _init_upsample(self,
                       in_channels, out_channels,
                       smooth=False, bias=False):
        if smooth:
            inter_out_channels = out_channels
            if out_channels == 1:
                inter_out_channels = in_channels
            module_list = [
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_channels, inter_out_channels, 3, 1, 1, bias=bias)]
            if out_channels == 1:
                module_list.append(
                    nn.Conv2d(in_channels, out_channels,
                              kernel_size=1, stride=1, padding=1, bias=True))

            return nn.Sequential(module_list)
        else:
            return nn.ConvTranspose2d(in_channels, out_channels, 2, 2)

    def forward(self, x):
        x = self.body(x)
        p2, p3, p4, p5 = self.fpn(x).values()

        p3_up = F.interpolate(p3, scale_factor=2)
        p4_up = F.interpolate(p4, scale_factor=4)
        p5_up = F.interpolate(p5, scale_factor=8)

        fuse = torch.cat([p2, p3_up, p4_up, p5_up], 1)
        fuse = self.asf(fuse)

        maps = {}
        prob_map = self.prob(fuse)
        maps['prob_map'] = prob_map
        if self.training:
            thresh_map = self.thresh(fuse)
            maps['thresh_map'] = thresh_map
            bin_map = 1 / (1 + torch.exp(-20 * (prob_map - thresh_map)))
            maps['bin_map'] = bin_map
        return maps

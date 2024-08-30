import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import FeaturePyramidNetwork
from torchvision.models.resnet import resnet18
from torchvision.models.feature_extraction import create_feature_extractor


class ASF(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(1024, 1024, 3, padding=1)

        self.spatial = nn.Sequential(
            nn.Conv2d(1, 1, 3, bias=False, padding=1),
            nn.ReLU(),
            nn.Conv2d(1, 1, 1, bias=False),
            nn.Sigmoid(),
        )
        self.attention = nn.Sequential(nn.Conv2d(1024, 1, 1, bias=False), nn.Sigmoid())

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
        return_nodes = {
            "layer1": "layer1",
            "layer2": "layer2",
            "layer3": "layer3",
            "layer4": "layer4",
        }
        self.body = create_feature_extractor(backbone, return_nodes)

        self.fpn = FeaturePyramidNetwork([64, 128, 256, 512], 256)
        self.asf = ASF()

        self.prob = nn.Sequential(
            nn.Conv2d(1024, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, 2, 2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 1, 2, 2),
            nn.Sigmoid(),
        )
        self.thresh = nn.Sequential(
            nn.Conv2d(1024, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, 2, 2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 1, 2, 2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.body(x)
        p2, p3, p4, p5 = self.fpn(x).values()

        p3_up = F.interpolate(p3, scale_factor=2)
        p4_up = F.interpolate(p4, scale_factor=4)
        p5_up = F.interpolate(p5, scale_factor=8)

        fuse = torch.cat([p2, p3_up, p4_up, p5_up], 1)
        fuse = self.asf(fuse)

        prob = self.prob(fuse).squeeze(1)
        thresh = self.thresh(fuse).squeeze(1)
        binary = 1 / (1 + torch.exp(-50 * (prob - thresh)))
        return binary
        
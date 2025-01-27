import torch
import torch.nn as nn
from torchvision.ops import FeaturePyramidNetwork
import torch.nn.functional as F


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.stage1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, (1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, (3, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, (1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.stage4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, (1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.fpn = FeaturePyramidNetwork([64, 128, 256, 512], 64)

    def forward(self, x):
        x = self.conv1(x)
        x2 = self.stage1(x)
        x3 = self.stage2(x2)
        x4 = self.stage3(x3)
        x5 = self.stage4(x4)

        return x2, x3, x4, x5


class Neck(nn.Module):
    def __init__(self):
        super().__init__()

        self.reduce5 = nn.Conv2d(512, 128, kernel_size=1, bias=False)
        self.reduce4 = nn.Conv2d(256, 128, kernel_size=1, bias=False)
        self.reduce3 = nn.Conv2d(128, 128, kernel_size=1, bias=False)
        self.reduce2 = nn.Conv2d(64, 128, kernel_size=1, bias=False)

        self.smooth5 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.smooth4 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.smooth3 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.smooth2 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)

        self.conv = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

    def upsample_cat(self, p2, p3, p4, p5):
        size = p2.size()[2:]
        p3 = F.interpolate(p3, size=size)
        p4 = F.interpolate(p4, size=size)
        p5 = F.interpolate(p5, size=size)
        return torch.cat([p2, p3, p4, p5], dim=1)

    def forward(self, x):
        c2, c3, c4, c5 = x

        p5 = self.reduce5(c5)
        p4 = F.interpolate(p5, size=c4.size()[2:]) + self.reduce4(c4)
        p3 = F.interpolate(p4, size=c3.size()[2:]) + self.reduce3(c3)
        p2 = F.interpolate(p3, size=c2.size()[2:]) + self.reduce2(c2)
        p5 = self.smooth5(p5)
        p4 = self.smooth4(p4)
        p3 = self.smooth3(p3)
        p2 = self.smooth2(p2)

        p = self.upsample_cat(p2, p3, p4, p5)
        p = self.conv(p)
        return p


class Head(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(512, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, 2, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 1, 2, 2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.conv(x).squeeze(1)


class FAST(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone()
        self.neck = Neck()
        self.head = Head()

    def forward(self, images):
        x = self.backbone(images)
        x = self.neck(x)
        x = self.head(x)
        return x

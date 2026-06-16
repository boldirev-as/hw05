import torch
from torch import nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(a, b, 3, padding=1),
            nn.BatchNorm2d(b),
            nn.ReLU(inplace=True),
            nn.Conv2d(b, b, 3, padding=1),
            nn.BatchNorm2d(b),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Model(nn.Module):
    def __init__(self, steps=20, learned=True):
        super().__init__()
        self.e1 = Block(3, 32)
        self.e2 = Block(32, 64)
        self.e3 = Block(64, 128)
        self.mid = Block(128, 256)
        self.d3 = Block(256 + 128, 128)
        self.d2 = Block(128 + 64, 64)
        self.d1 = Block(64 + 32, 32)
        self.out = nn.Conv2d(32, 3, 1)

    def forward(self, y, psf):
        x1 = self.e1(y)
        x2 = self.e2(F.max_pool2d(x1, 2))
        x3 = self.e3(F.max_pool2d(x2, 2))
        x = self.mid(F.max_pool2d(x3, 2))
        x = F.interpolate(x, size=x3.shape[-2:], mode="bilinear", align_corners=False)
        x = self.d3(torch.cat([x, x3], 1))
        x = F.interpolate(x, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.d2(torch.cat([x, x2], 1))
        x = F.interpolate(x, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.d1(torch.cat([x, x1], 1))
        return torch.sigmoid(self.out(x))

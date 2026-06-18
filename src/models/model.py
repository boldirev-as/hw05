import torch
from torch import nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.skip = nn.Identity() if a == b else nn.Conv2d(a, b, 1)
        self.net = nn.Sequential(
            nn.Conv2d(a, b, 3, padding=1),
            nn.GroupNorm(8, b),
            nn.SiLU(inplace=True),
            nn.Conv2d(b, b, 3, padding=1),
            nn.GroupNorm(8, b),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x) + self.skip(x)


class Model(nn.Module):
    def __init__(self, steps=20, learned=True, base=48):
        super().__init__()
        c = base
        self.e1 = Block(3, c)
        self.e2 = Block(c, c * 2)
        self.e3 = Block(c * 2, c * 4)
        self.mid = Block(c * 4, c * 8)
        self.d3 = Block(c * 8 + c * 4, c * 4)
        self.d2 = Block(c * 4 + c * 2, c * 2)
        self.d1 = Block(c * 2 + c, c)
        self.out = nn.Conv2d(c, 3, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

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
        return (y + self.out(x)).clamp(0, 1)

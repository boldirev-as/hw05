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
    def __init__(self, steps=20, learned=True, base=48, in_channels=3):
        super().__init__()
        c = base
        self.e1 = Block(in_channels, c)
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
        skip = y[:, :3]
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
        return (skip + self.out(x)).clamp(0, 1)


class SmallNet(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        self.net = nn.Sequential(
            Block(3, base),
            Block(base, base),
            nn.Conv2d(base, 3, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return (x + self.net(x)).clamp(0, 1)


class ADMM(nn.Module):
    def __init__(self, steps=20, learned=False, denoiser=False, base=32):
        super().__init__()
        self.steps = steps
        self.log_rho = nn.Parameter(torch.full((steps,), -4.0), requires_grad=learned)
        self.log_lam = nn.Parameter(torch.full((steps,), -5.0), requires_grad=learned)
        self.denoiser = SmallNet(base) if denoiser else None

    def forward(self, y, psf):
        psf = psf / psf.sum((-2, -1), keepdim=True).clamp_min(1e-8)
        hf = torch.fft.rfft2(psf)
        yf = torch.fft.rfft2(y)
        hty = torch.conj(hf) * yf
        z = torch.zeros_like(y)
        u = torch.zeros_like(y)
        x = y
        for i in range(self.steps):
            rho = self.log_rho[i].exp()
            rhs = hty + rho * torch.fft.rfft2(z - u)
            den = (torch.conj(hf) * hf).real + rho
            x = torch.fft.irfft2(rhs / den.clamp_min(1e-8), s=y.shape[-2:]).real
            v = (x + u).clamp(0, 1)
            if self.denoiser is None:
                lam = self.log_lam[i].exp()
                z = ((1 - lam) * v + lam * F.avg_pool2d(v, 3, stride=1, padding=1)).clamp(0, 1)
            else:
                z = self.denoiser(v)
            u = u + x - z
        return z.clamp(0, 1)


class ModularADMM(nn.Module):
    def __init__(self, steps=5, base=48, pre=True, post=True, fuse=False):
        super().__init__()
        self.fuse = fuse
        self.pre = SmallNet(base) if pre else nn.Identity()
        self.core = ADMM(steps, True, True, max(base // 2, 16))
        if post:
            self.post = Model(steps, True, base, 6 if fuse else 3)
        else:
            self.post = nn.Identity()

    def forward(self, y, psf):
        z = self.pre(y)
        x = self.core(z, psf)
        if isinstance(self.post, nn.Identity):
            return x
        if self.fuse:
            return self.post(torch.cat([x, y], 1), psf)
        return self.post(x, psf)


def build_model(name="unet", steps=20, base=48):
    if name == "unet":
        return Model(steps, True, base)
    if name == "admm":
        return ADMM(100 if steps == 20 else steps, False, False, base)
    if name == "leadmm":
        return ADMM(steps, True, True, base)
    if name == "modular":
        return ModularADMM(steps, base, True, True)
    if name == "modular_fuse":
        return ModularADMM(steps, base, True, True, True)
    if name == "modular_pre":
        return ModularADMM(steps, base, True, False)
    if name == "modular_post":
        return ModularADMM(steps, base, False, True)
    raise ValueError(name)

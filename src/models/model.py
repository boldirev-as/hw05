import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, steps=20, learned=True):
        super().__init__()
        self.steps = steps
        self.mu = nn.Parameter(torch.full((steps,), 1e-2), requires_grad=learned)
        self.denoiser = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, padding=1),
        )

    def forward(self, y, psf):
        psf = psf / psf.sum((-2, -1), keepdim=True).clamp_min(1e-8)
        hf = torch.fft.rfft2(psf)
        yf = torch.fft.rfft2(y)
        x = torch.fft.irfft2(torch.conj(hf) * yf / (torch.conj(hf) * hf + 1e-2), s=y.shape[-2:]).real
        for i in range(self.steps):
            hx = torch.fft.irfft2(torch.fft.rfft2(x) * hf, s=y.shape[-2:]).real
            grad = torch.fft.irfft2(torch.fft.rfft2(hx - y) * torch.conj(hf), s=y.shape[-2:]).real
            x = (x - self.mu[i].abs() * grad).clamp(0, 1)
        return (x + self.denoiser(x)).clamp(0, 1)

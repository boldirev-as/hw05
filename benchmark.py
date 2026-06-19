import argparse
import time

import torch

from src.config import read_config
from src.models import build_model


def predict(model, y, psf, tta):
    pred = model(y, psf)
    if not tta:
        return pred
    yh = torch.flip(y, (3,))
    yv = torch.flip(y, (2,))
    yhv = torch.flip(y, (2, 3))
    ph = torch.flip(model(yh, psf), (3,))
    pv = torch.flip(model(yv, psf), (2,))
    phv = torch.flip(model(yhv, psf), (2, 3))
    return (pred + ph + pv + phv) / 4


def main():
    base = argparse.ArgumentParser(add_help=False)
    base.add_argument("--config", default=None)
    known, _ = base.parse_known_args()
    p = argparse.ArgumentParser(parents=[base])
    p.add_argument("--ckpt", default=None)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--base", type=int, default=64)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--model", default="modular")
    p.add_argument("--runs", type=int, default=100)
    p.add_argument("--no-tta", action="store_true")
    p.set_defaults(**read_config(known.config))
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(args.model, args.steps, args.base).to(device).eval()
    if args.ckpt:
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
    y = torch.rand(1, 3, args.size, args.size, device=device)
    psf = torch.zeros_like(y)
    psf[:, :, args.size // 2, args.size // 2] = 1
    with torch.no_grad():
        for _ in range(10):
            predict(model, y, psf, not args.no_tta)
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(args.runs):
            predict(model, y, psf, not args.no_tta)
        if device == "cuda":
            torch.cuda.synchronize()
    print({"sec_per_image": (time.time() - start) / args.runs})


if __name__ == "__main__":
    main()

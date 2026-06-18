import argparse
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from src.datasets import make_loader
from src.models import Model


def save(x, path):
    x = x.detach().cpu().clamp(0, 1)[0].permute(1, 2, 0).numpy()
    Image.fromarray((x * 255).astype("uint8")).save(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="recon")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--steps", type=int, default=20)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = Model(args.steps, True).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    loader = make_loader(args.data, "test", args.size, 1, None, False)
    with torch.no_grad():
        for batch in tqdm(loader):
            pred = model(batch["lensless"].to(device), batch["psf"].to(device), batch["label"].to(device))
            save(pred, out / f"{batch['id'][0]}.png")


if __name__ == "__main__":
    main()

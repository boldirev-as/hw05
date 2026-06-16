import argparse
from pathlib import Path

import lpips
import torch
from PIL import Image
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from tqdm import tqdm


def load(path, size):
    image = Image.open(path).convert("RGB")
    if size:
        image = image.resize((size, size))
    x = torch.tensor(list(image.getdata()), dtype=torch.float32).view(image.height, image.width, 3)
    return (x.permute(2, 0, 1) / 255)[None]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gt", required=True)
    p.add_argument("--pred", required=True)
    p.add_argument("--size", type=int, default=256)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    psnr = PeakSignalNoiseRatio(data_range=1).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1).to(device)
    lp = lpips.LPIPS(net="vgg").to(device)
    vals = {"mse": 0, "psnr": 0, "ssim": 0, "lpips": 0}
    n = 0
    for pred_path in tqdm(sorted(Path(args.pred).glob("*.png"))):
        gt_path = Path(args.gt) / pred_path.name
        if not gt_path.exists():
            continue
        pred = load(pred_path, args.size).to(device)
        gt = load(gt_path, args.size).to(device)
        vals["mse"] += float(((pred - gt) ** 2).mean())
        vals["psnr"] += float(psnr(pred, gt))
        vals["ssim"] += float(ssim(pred, gt))
        vals["lpips"] += float(lp(pred * 2 - 1, gt * 2 - 1).mean())
        n += 1
    print({k: v / max(n, 1) for k, v in vals.items()} | {"count": n})


if __name__ == "__main__":
    main()

import argparse
import os
from pathlib import Path

from comet_ml import Experiment
import torch
from tqdm import tqdm

from src.datasets import make_loader
from src.models import Model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/model")
    p.add_argument("--data", default=None)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--comet", action="store_true")
    p.add_argument("--project", default="hw05")
    p.add_argument("--workspace", default=None)
    p.add_argument("--name", default=None)
    return p.parse_args()


def comet_key():
    key = os.environ.get("COMET_API_KEY")
    if key:
        return key
    try:
        from kaggle_secrets import UserSecretsClient

        return UserSecretsClient().get_secret("COMET_API_KEY")
    except Exception:
        return None


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train_loader = make_loader(args.data, "train", args.size, args.batch, args.limit, True)
    val_loader = make_loader(args.data, "test", args.size, 1, 64, False)
    model = Model(args.steps, True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    run = Experiment(api_key=comet_key(), project_name=args.project, workspace=args.workspace) if args.comet else None
    if run:
        run.log_parameters(vars(args))
        if args.name:
            run.set_name(args.name)
    best = 0
    step = 0
    for epoch in range(args.epochs):
        model.train()
        for batch in tqdm(train_loader):
            y = batch["lensless"].to(device)
            psf = batch["psf"].to(device)
            target = batch["target"].to(device)
            pred = model(y, psf)
            loss = (pred - target).abs().mean() + ((pred - target) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            if run:
                run.log_metric("loss", loss.item(), step=step, epoch=epoch)
            step += 1
        psnr = evaluate(model, val_loader, device)
        if run:
            run.log_metric("psnr", psnr, step=step, epoch=epoch)
        if psnr > best:
            best = psnr
            torch.save(model.state_dict(), out / "best.pt")
            if run:
                run.log_model("best", str(out / "best.pt"))
        torch.save(model.state_dict(), out / "last.pt")
        print(epoch, psnr)
    if run:
        run.end()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    vals = []
    for batch in loader:
        y = batch["lensless"].to(device)
        psf = batch["psf"].to(device)
        target = batch["target"].to(device)
        pred = model(y, psf)
        mse = ((pred - target) ** 2).mean().clamp_min(1e-8)
        vals.append(float(10 * torch.log10(1 / mse)))
    return sum(vals) / len(vals)


if __name__ == "__main__":
    main()

import argparse
import csv
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
    p.add_argument("--ckpt", default=None)
    p.add_argument("--data", default=None)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--base", type=int, default=48)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--l1", type=float, default=0.1)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--val-limit", type=int, default=128)
    p.add_argument("--no-tta", action="store_true")
    p.add_argument("--dp", action="store_true")
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


def save_result(args, best, best_epoch):
    path = Path("runs") / "experiments.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["name", "size", "batch", "epochs", "steps", "base", "lr", "l1", "ema", "tta", "dp", "limit", "val_limit", "best_psnr", "best_epoch", "out"])
        w.writerow([args.name, args.size, args.batch, args.epochs, args.steps, args.base, args.lr, args.l1, args.ema, not args.no_tta, args.dp, args.limit, args.val_limit, best, best_epoch, args.out])


def update_ema(model, ema_model, decay):
    with torch.no_grad():
        for p, q in zip(model.parameters(), ema_model.parameters()):
            q.mul_(decay).add_(p, alpha=1 - decay)
        for p, q in zip(model.buffers(), ema_model.buffers()):
            q.copy_(p)


def unwrap(model):
    return model.module if hasattr(model, "module") else model


def load_weights(model, path, device):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train_loader = make_loader(args.data, "train", args.size, args.batch, args.limit, True)
    val_loader = make_loader(args.data, "test", args.size, 1, args.val_limit, False)
    model = Model(args.steps, True, args.base).to(device)
    if args.ckpt:
        load_weights(model, args.ckpt, device)
    ema_model = Model(args.steps, True, args.base).to(device)
    ema_model.load_state_dict(model.state_dict())
    ema_model.eval()
    if args.dp and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        ema_model = torch.nn.DataParallel(ema_model)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    run = Experiment(api_key=comet_key(), project_name=args.project, workspace=args.workspace) if args.comet else None
    if run:
        run.log_parameters(vars(args))
        if args.name:
            run.set_name(args.name)
    best = 0
    best_epoch = -1
    step = 0
    for epoch in range(args.epochs):
        model.train()
        for batch in tqdm(train_loader):
            y = batch["lensless"].to(device)
            psf = batch["psf"].to(device)
            target = batch["target"].to(device)
            pred = model(y, psf)
            loss = ((pred - target) ** 2).mean() + args.l1 * (pred - target).abs().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            if args.ema > 0:
                update_ema(model, ema_model, args.ema)
            if run:
                run.log_metric("loss", loss.item(), step=step, epoch=epoch)
            step += 1
        psnr = evaluate(ema_model if args.ema > 0 else model, val_loader, device, not args.no_tta)
        if run:
            run.log_metric("psnr", psnr, step=step, epoch=epoch)
        if psnr > best:
            best = psnr
            best_epoch = epoch
            torch.save(unwrap(ema_model if args.ema > 0 else model).state_dict(), out / "best.pt")
            if run:
                run.log_model("best", str(out / "best.pt"))
        torch.save(unwrap(model).state_dict(), out / "last.pt")
        print(epoch, psnr)
        scheduler.step()
    save_result(args, best, best_epoch)
    if run:
        run.log_metric("best_psnr", best)
        run.log_metric("best_epoch", best_epoch)
        run.end()


@torch.no_grad()
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


@torch.no_grad()
def evaluate(model, loader, device, tta):
    model.eval()
    vals = []
    for batch in loader:
        y = batch["lensless"].to(device)
        psf = batch["psf"].to(device)
        target = batch["target"].to(device)
        pred = predict(model, y, psf, tta)
        mse = ((pred - target) ** 2).mean().clamp_min(1e-8)
        vals.append(float(10 * torch.log10(1 / mse)))
    return sum(vals) / len(vals)


if __name__ == "__main__":
    main()

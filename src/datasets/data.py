from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def image_to_tensor(image, size, rotate=False):
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    image = image.convert("RGB")
    if size:
        image = image.resize((size, size))
    x = np.array(image).astype("float32") / 255
    x = torch.from_numpy(x).permute(2, 0, 1)
    if rotate:
        x = torch.rot90(x, 2, (1, 2))
    return x


def psf_to_tensor(x, size):
    if isinstance(x, (str, Path)):
        x = np.load(x)
    x = torch.tensor(np.array(x), dtype=torch.float32)
    if x.ndim == 2:
        x = x[None]
    if x.ndim == 3 and x.shape[-1] in (1, 3):
        x = x.permute(2, 0, 1)
    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)
    if size:
        x = torch.nn.functional.interpolate(x[None], (size, size), mode="bilinear", align_corners=False)[0]
    return x / x.sum((-2, -1), keepdim=True).clamp_min(1e-8)


def default_psf(size):
    x = torch.zeros(3, size, size)
    x[:, size // 2, size // 2] = 1
    return x


class CustomDirDataset(Dataset):
    def __init__(self, root, size=256, train=True):
        self.root = Path(root)
        self.size = size
        self.train = train
        self.ids = sorted(p.stem for p in (self.root / "lensless").glob("*"))

    def __len__(self):
        return len(self.ids)

    def find_image(self, folder, name):
        for ext in [".png", ".jpg", ".jpeg"]:
            path = self.root / folder / f"{name}{ext}"
            if path.exists():
                return path
        return None

    def __getitem__(self, i):
        name = self.ids[i]
        lensless = image_to_tensor(self.find_image("lensless", name), self.size, True)
        psf = psf_to_tensor(self.root / "masks" / f"{name}.npy", self.size)
        target_path = self.find_image("lensed", name)
        if target_path is None:
            target = lensless
        else:
            target = image_to_tensor(target_path, self.size)
        if self.train:
            lensless, target = augment(lensless, target)
        return {"id": name, "lensless": lensless, "psf": psf, "target": target}


class HFDataset(Dataset):
    def __init__(self, split, size=256, limit=None, train=False):
        self.data = load_dataset("bezzam/DigiCam-Mirflickr-MultiMask-10K", split=split)
        if limit:
            self.data = self.data.select(range(min(limit, len(self.data))))
        self.size = size
        self.train = train

    def __len__(self):
        return len(self.data)

    def pick(self, row, names):
        for name in names:
            if name in row and row[name] is not None:
                return row[name]
        return None

    def __getitem__(self, i):
        row = self.data[i]
        lensless = self.pick(row, ["lensless", "lensless_image", "measurement"])
        target = self.pick(row, ["lensed", "image", "original"])
        psf = self.pick(row, ["psf", "mask", "masks"])
        if psf is None:
            psf = default_psf(self.size)
        lensless = image_to_tensor(lensless, self.size, True)
        target = image_to_tensor(target, self.size)
        if self.train:
            lensless, target = augment(lensless, target)
        return {
            "id": str(row.get("id", i)),
            "lensless": lensless,
            "target": target,
            "psf": psf if torch.is_tensor(psf) else psf_to_tensor(psf, self.size),
        }


def make_loader(path=None, split="train", size=256, batch_size=2, limit=None, shuffle=False):
    if path:
        data = CustomDirDataset(path, size, shuffle)
    else:
        data = HFDataset(split, size, limit, shuffle)
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True)


def augment(lensless, target):
    if torch.rand(()) < 0.5:
        lensless = torch.flip(lensless, (2,))
        target = torch.flip(target, (2,))
    if torch.rand(()) < 0.5:
        lensless = torch.flip(lensless, (1,))
        target = torch.flip(target, (1,))
    return lensless, target

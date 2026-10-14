"""Datasets.

Expected data format (the one produced by best_convex_model/create_dataset.ipynb):
    an HDF5 file with one image per key, each an array of shape (C, H, W) (or (H, W)),
    float32 in [0, 1]. Keys are arbitrary strings ("1", "2", ... or "0", "1", ...).
    Patch files in the same layout (e.g. CRR's train.h5 with 238400 keys of (1, 40, 40)) also work.
"""
import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class H5ImageDataset(Dataset):
    """One image per HDF5 key. Optional random crops + D4 augmentation (train) or full images (val/test)."""

    def __init__(self, path, patch_size=None, augment=False, divisible_by=1, repeat=1, max_items=None):
        self.path = os.path.expanduser(str(path))
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"data file not found: {self.path}")
        with h5py.File(self.path, "r") as f:
            keys = list(f.keys())
        try:
            keys = sorted(keys, key=lambda k: int(k))
        except ValueError:
            keys = sorted(keys)
        if max_items is not None:
            keys = keys[: int(max_items)]
        self.keys = keys
        self.patch_size = patch_size
        self.augment = augment
        self.divisible_by = int(divisible_by)
        self.repeat = int(repeat)
        self._file = None  # opened lazily in each worker

    def __len__(self):
        return len(self.keys) * self.repeat

    def _load(self, i):
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        arr = np.asarray(self._file[self.keys[i]], dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[None]
        return torch.from_numpy(arr)

    def __getitem__(self, idx):
        img = self._load(idx % len(self.keys))
        if self.patch_size is not None:
            p = int(self.patch_size)
            _, H, W = img.shape
            if H < p or W < p:
                raise ValueError(f"image {self.keys[idx % len(self.keys)]} of size {H}x{W} smaller than patch {p}")
            i = int(torch.randint(0, H - p + 1, (1,)))
            j = int(torch.randint(0, W - p + 1, (1,)))
            img = img[:, i : i + p, j : j + p]
        if self.augment:  # random element of the dihedral group D4
            k = int(torch.randint(0, 4, (1,)))
            if k:
                img = torch.rot90(img, k, dims=(1, 2))
            if int(torch.randint(0, 2, (1,))):
                img = torch.flip(img, dims=(2,))
        d = self.divisible_by
        if d > 1:
            img = img[:, : (img.shape[1] // d) * d, : (img.shape[2] // d) * d]
        return img.contiguous()


class SyntheticImageDataset(Dataset):
    """Piecewise-smooth random images (blobs + rectangles + ramps), deterministic per index.
    Only for smoke tests when no data file is available."""

    def __init__(self, n=64, size=64, seed=0):
        self.n, self.size, self.seed = int(n), int(size), int(seed)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        g = torch.Generator().manual_seed(self.seed * 100003 + idx)
        s = self.size
        yy, xx = torch.meshgrid(torch.linspace(0, 1, s), torch.linspace(0, 1, s), indexing="ij")
        img = 0.5 + 0.2 * (torch.rand(1, generator=g) - 0.5) * (xx + yy)
        for _ in range(4):  # smooth blobs
            cx, cy = torch.rand(2, generator=g)
            r = 0.05 + 0.2 * torch.rand(1, generator=g)
            a = 0.6 * (torch.rand(1, generator=g) - 0.5)
            img = img + a * torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r ** 2))
        for _ in range(3):  # rectangles with sharp edges
            x0, y0 = torch.rand(2, generator=g)
            w, h = 0.1 + 0.4 * torch.rand(2, generator=g)
            a = 0.6 * (torch.rand(1, generator=g) - 0.5)
            img = img + a * ((xx > x0) & (xx < x0 + w) & (yy > y0) & (yy < y0 + h)).float()
        return img.clamp(0, 1)[None].contiguous()


def fixed_noise(shape, sigma, seed, device=None):
    """Deterministic Gaussian noise (for reproducible validation / test)."""
    g = torch.Generator().manual_seed(int(seed))
    return (sigma * torch.randn(*shape, generator=g)).to(device)


def build_train_dataset(cfg):
    """cfg: the 'data' section of a config."""
    if cfg.get("synthetic", False):
        return SyntheticImageDataset(n=cfg.get("synthetic_n", 64), size=cfg.get("patch_size", 64) or 64, seed=0)
    return H5ImageDataset(
        cfg["train_file"],
        patch_size=cfg.get("patch_size", 64),
        augment=cfg.get("augment", True),
        repeat=cfg.get("repeat", 1),
        max_items=cfg.get("train_max_items", None),
    )


def build_val_dataset(cfg):
    if cfg.get("synthetic", False):
        return SyntheticImageDataset(n=cfg.get("synthetic_val_n", 8), size=cfg.get("synthetic_val_size", 96), seed=1)
    return H5ImageDataset(
        cfg["val_file"],
        patch_size=cfg.get("val_patch_size", None),
        augment=False,
        divisible_by=cfg.get("val_divisible_by", 1),
        max_items=cfg.get("val_max_items", None),
    )

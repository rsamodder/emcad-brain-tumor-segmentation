"""Slice-level dataset over preprocessed 4-channel axial slices."""
import glob

import numpy as np
import torch
from torch.utils.data import Dataset


class BratsSlices(Dataset):
    """Training keeps all tumor slices plus `keep_empty_frac` of tumor-free brain slices
    (random flips as augmentation). Validation keeps all slices with brain.
    Full volumes are kept in `imgs` / `labs` for 3D evaluation."""

    def __init__(self, folder, train=True, keep_empty_frac=0.2, seed=0):
        self.train = train
        self.imgs, self.labs, self.index = [], [], []
        rng = np.random.RandomState(seed)
        for p in sorted(glob.glob(folder + "/*_img.npy")):
            x = np.load(p)                            # (D, 4, S, S) float16
            y = np.load(p.replace("_img", "_lab"))    # (D, S, S) uint8
            pid = len(self.imgs)
            self.imgs.append(x)
            self.labs.append(y)
            for z in range(x.shape[0]):
                if not (np.abs(x[z]) > 0).mean() > 0.02:      # skip slices with almost no brain
                    continue
                if (y[z] > 0).any() or (not train) or rng.rand() < keep_empty_frac:
                    self.index.append((pid, z))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        pid, z = self.index[i]
        x = torch.from_numpy(self.imgs[pid][z].astype(np.float32))    # (4, S, S)
        y = torch.from_numpy(self.labs[pid][z].astype(np.int64))      # (S, S)
        if self.train:
            if np.random.rand() < 0.5:
                x, y = x.flip(-1), y.flip(-1)
            if np.random.rand() < 0.5:
                x, y = x.flip(-2), y.flip(-2)
        return x, y

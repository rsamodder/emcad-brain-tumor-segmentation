"""Convert MSD Task01 BrainTumour NIfTI volumes into 4-channel, 128x128 axial slices (.npy).

MSD volumes are (240, 240, 155, 4) with modality order FLAIR, T1w, T1gd (T1ce), T2w.
Each modality is z-score normalised over brain (non-zero) voxels, then slices are resized.
"""
import argparse
import os

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F


def process(img_path, lab_path, out_path, size):
    x = nib.load(img_path).get_fdata().astype(np.float32)
    y = nib.load(lab_path).get_fdata().astype(np.uint8)
    ch = [i for i, s in enumerate(x.shape) if s == 4][0]    # modality axis
    x = np.moveaxis(x, ch, 0)                               # (4, H, W, D)
    x = np.transpose(x, (0, 3, 1, 2))                       # (4, D, H, W): axial slices
    y = np.transpose(y, (2, 0, 1))                          # (D, H, W)
    for c in range(4):                                      # per-modality z-score on brain voxels
        m = x[c] > 0
        if m.sum() > 0:
            x[c][m] = (x[c][m] - x[c][m].mean()) / (x[c][m].std() + 1e-8)
        x[c][~m] = 0
    xt = F.interpolate(torch.from_numpy(x).permute(1, 0, 2, 3), size=(size, size),
                       mode="bilinear", align_corners=False)               # (D, 4, S, S)
    yt = F.interpolate(torch.from_numpy(y)[:, None].float(), size=(size, size), mode="nearest")
    np.save(out_path + "_img.npy", xt.numpy().astype(np.float16))
    np.save(out_path + "_lab.npy", yt[:, 0].numpy().astype(np.uint8))


def run(img_dir, lab_dir, out_dir, n, size):
    os.makedirs(out_dir, exist_ok=True)
    for f in sorted(os.listdir(lab_dir))[:n]:
        if not f.endswith((".nii", ".nii.gz")):
            continue
        base = os.path.join(out_dir, f.replace(".nii.gz", "").replace(".nii", ""))
        if not os.path.exists(base + "_lab.npy"):
            process(os.path.join(img_dir, f), os.path.join(lab_dir, f), base, size)
    print("done:", out_dir)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train_images", required=True)
    ap.add_argument("--train_labels", required=True)
    ap.add_argument("--val_images", required=True)
    ap.add_argument("--val_labels", required=True)
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--n_train", type=int, default=80, help="number of training patients to use")
    ap.add_argument("--n_val", type=int, default=20, help="number of validation patients to use")
    a = ap.parse_args()
    run(a.train_images, a.train_labels, os.path.join(a.out_dir, "train"), a.n_train, a.size)
    run(a.val_images, a.val_labels, os.path.join(a.out_dir, "val"), a.n_val, a.size)


if __name__ == "__main__":
    main()

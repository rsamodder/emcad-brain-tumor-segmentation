"""Overlay figure: best / median / worst patient (by mean 3D Dice) at their largest-tumor slice,
plus the worst-performing tumor slice overall. Needs results/per_patient_metrics.csv (run evaluate.py first)."""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from src.dataset import BratsSlices
from src.inference import predict_volume
from src.model import make_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--emcad_dir", default="EMCAD")
    ap.add_argument("--weights", default="pretrained_pth/pvt/pvt_v2_b2.pth")
    ap.add_argument("--ckpt", default="checkpoints/best.pth")
    ap.add_argument("--metrics", default="results/per_patient_metrics.csv")
    ap.add_argument("--out", default="results/overlays.png")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_ds = BratsSlices(os.path.join(a.data_dir, "val"), train=False)
    model = make_model(a.emcad_dir, a.weights, verbose=False).to(device)
    model.load_state_dict(torch.load(a.ckpt, map_location=device))
    preds = {pid: predict_volume(model, val_ds.imgs[pid], device) for pid in range(len(val_ds.imgs))}

    df = pd.read_csv(a.metrics)
    df["mean_dice"] = df[["Dice_WT", "Dice_TC", "Dice_ET"]].mean(axis=1)
    order = df.sort_values("mean_dice").patient.tolist()
    best_p, med_p, worst_p = order[-1], order[len(order) // 2], order[0]

    def biggest_slice(pid):
        return int(np.argmax((val_ds.labs[pid] > 0).sum((1, 2))))

    fail = (None, None, 2.0)   # worst per-slice WT Dice among slices with a reasonably sized tumor
    for pid in range(len(val_ds.imgs)):
        g, p = val_ds.labs[pid] > 0, preds[pid] > 0
        for z in range(g.shape[0]):
            if g[z].sum() > 150:
                d = 2 * (g[z] & p[z]).sum() / (g[z].sum() + p[z].sum() + 1e-8)
                if d < fail[2]:
                    fail = (pid, z, d)

    cases = [(best_p, biggest_slice(best_p), "Best patient"),
             (med_p, biggest_slice(med_p), "Median patient"),
             (worst_p, biggest_slice(worst_p), "Worst patient"),
             (fail[0], fail[1], f"Failure slice (WT Dice {fail[2]:.2f})")]

    cmap = ListedColormap([(0, 0, 0, 0), (0, 1, 0, 0.55), (0, 0.4, 1, 0.55), (1, 0, 0, 0.6)])
    cols = ["FLAIR", "T1ce", "Ground truth", "Prediction"]
    fig, ax = plt.subplots(len(cases), 4, figsize=(12, 3 * len(cases)))
    for r, (pid, z, title) in enumerate(cases):
        x = val_ds.imgs[pid][z].astype(np.float32)
        g, p = val_ds.labs[pid][z], preds[pid][z]
        flair, t1ce = x[0], x[2]    # MSD channel order: FLAIR, T1w, T1gd, T2w
        ax[r, 0].imshow(flair, cmap="gray")
        ax[r, 1].imshow(t1ce, cmap="gray")
        ax[r, 2].imshow(flair, cmap="gray")
        ax[r, 2].imshow(g, cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
        ax[r, 3].imshow(flair, cmap="gray")
        ax[r, 3].imshow(p, cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
        for c in range(4):
            ax[r, c].axis("off")
            if r == 0:
                ax[r, c].set_title(cols[c])
        ax[r, 0].text(-0.05, 0.5, f"{title}\n(val patient {pid}, slice {z})", transform=ax[r, 0].transAxes,
                      ha="right", va="center", fontsize=9)
    fig.legend(handles=[Patch(color=(0, 1, 0), label="Edema (1)"),
                        Patch(color=(0, 0.4, 1), label="Non-enhancing (2)"),
                        Patch(color=(1, 0, 0), label="Enhancing (3)")], loc="lower center", ncol=3)
    plt.tight_layout(rect=[0.08, 0.04, 1, 1])
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    plt.savefig(a.out, dpi=150, bbox_inches="tight")
    print("saved", a.out)


if __name__ == "__main__":
    main()

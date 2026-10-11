"""Per-patient 3D Dice and HD95 (WT, TC, ET) on the validation patients."""
import argparse
import os

import numpy as np
import pandas as pd
import torch

from src.dataset import BratsSlices
from src.inference import predict_volume
from src.metrics import dice, hd95, regions
from src.model import make_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--emcad_dir", default="EMCAD")
    ap.add_argument("--weights", default="pretrained_pth/pvt/pvt_v2_b2.pth")
    ap.add_argument("--ckpt", default="checkpoints/best.pth")
    ap.add_argument("--out_dir", default="results")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out_dir, exist_ok=True)
    val_ds = BratsSlices(os.path.join(a.data_dir, "val"), train=False)

    model = make_model(a.emcad_dir, a.weights, verbose=False).to(device)
    model.load_state_dict(torch.load(a.ckpt, map_location=device))

    rows = []
    for pid in range(len(val_ds.imgs)):
        pred = predict_volume(model, val_ds.imgs[pid], device)
        rp, rg = regions(pred), regions(val_ds.labs[pid])
        row = {"patient": pid}
        for k in ["WT", "TC", "ET"]:
            row[f"Dice_{k}"] = dice(rp[k], rg[k])
            row[f"HD95_{k}"] = hd95(rp[k], rg[k])
        rows.append(row)

    df = pd.DataFrame(rows)
    summary = pd.DataFrame({
        "Dice": [df[f"Dice_{k}"].mean() for k in ["WT", "TC", "ET"]],
        "HD95 (mm)": [df[f"HD95_{k}"].mean() for k in ["WT", "TC", "ET"]],
        "n HD95 undefined": [df[f"HD95_{k}"].isna().sum() for k in ["WT", "TC", "ET"]],
    }, index=["WT", "TC", "ET"])
    summary.loc["Mean"] = [summary["Dice"].mean(), summary["HD95 (mm)"].mean(), np.nan]
    print(summary.round(3))
    df.to_csv(os.path.join(a.out_dir, "per_patient_metrics.csv"), index=False)
    summary.round(3).to_csv(os.path.join(a.out_dir, "summary_metrics.csv"))


if __name__ == "__main__":
    main()

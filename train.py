"""Train PVT-EMCAD-B2 on 4-channel BraTS/MSD slices; keep the best epoch by mean slice-level Dice."""
import argparse
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import BratsSlices
from src.losses import total_loss
from src.metrics import regions
from src.model import make_model
from src.utils import set_seed


@torch.no_grad()
def evaluate(model, loader, device):
    """Pooled slice-level Dice over the whole validation set (quick proxy used for model selection)."""
    model.eval()
    inter = {k: 0.0 for k in ["WT", "TC", "ET"]}
    tot = {k: 0.0 for k in inter}
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device, dtype=torch.float16, enabled=(device == "cuda")):
            outs = model(x)
        pred = sum(o.float() for o in outs).argmax(1)
        rp, rg = regions(pred), regions(y)
        for k in inter:
            inter[k] += (rp[k] & rg[k]).sum().item()
            tot[k] += rp[k].sum().item() + rg[k].sum().item()
    d = {k: 2 * inter[k] / (tot[k] + 1e-8) for k in inter}
    d["mean"] = float(np.mean([d["WT"], d["TC"], d["ET"]]))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--emcad_dir", default="EMCAD")
    ap.add_argument("--weights", default="pretrained_pth/pvt/pvt_v2_b2.pth")
    ap.add_argument("--out_dir", default="checkpoints")
    ap.add_argument("--log", default="logs/train_log.txt")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    set_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(a.log) or ".", exist_ok=True)

    train_ds = BratsSlices(os.path.join(a.data_dir, "train"), train=True)
    val_ds = BratsSlices(os.path.join(a.data_dir, "val"), train=False)
    print("train slices:", len(train_ds), "| val slices:", len(val_ds))
    train_loader = DataLoader(train_ds, batch_size=a.bs, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)

    model = make_model(a.emcad_dir, a.weights).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    best = 0.0
    with open(a.log, "w") as logf:
        for ep in range(a.epochs):
            model.train()
            t0, tl = time.time(), 0.0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                with torch.autocast(device, dtype=torch.float16, enabled=(device == "cuda")):
                    outs = model(x)
                loss = total_loss([o.float() for o in outs], y)
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                tl += loss.item()
            sched.step()
            d = evaluate(model, val_loader, device)
            line = (f"ep {ep+1}/{a.epochs} | loss {tl/len(train_loader):.3f} | "
                    f"Dice WT {d['WT']:.3f} TC {d['TC']:.3f} ET {d['ET']:.3f} mean {d['mean']:.3f} | "
                    f"{time.time()-t0:.0f}s")
            if d["mean"] > best:
                best = d["mean"]
                torch.save(model.state_dict(), os.path.join(a.out_dir, "best.pth"))
                line += "\n   saved best.pth"
            print(line)
            logf.write(line + "\n")
            logf.flush()


if __name__ == "__main__":
    main()

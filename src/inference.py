"""Volume-level inference: prediction = argmax of the sum of the four decoder-head logits."""
import numpy as np
import torch


@torch.no_grad()
def predict_volume(model, vol, device="cuda", bs=32):
    """vol: (D, 4, S, S) float16/32 array -> (D, S, S) uint8 label map."""
    model.eval()
    out = []
    for i in range(0, len(vol), bs):
        x = torch.from_numpy(vol[i:i + bs].astype(np.float32)).to(device)
        with torch.autocast(device, dtype=torch.float16, enabled=(device == "cuda")):
            o = model(x)
        out.append(sum(t.float() for t in o).argmax(1).cpu().numpy())
    return np.concatenate(out).astype(np.uint8)

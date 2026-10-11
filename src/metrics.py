"""Nested tumor regions and Dice / HD95 metrics."""
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

# approximate voxel spacing in mm: (between slices, H, W) after resizing 240 -> 128
SPACING = (1.0, 240 / 128, 240 / 128)


def regions(lbl):
    """Labels: 1 = edema, 2 = non-enhancing, 3 = enhancing.
    WT = 1,2,3 | TC = 2,3 | ET = 3. Works on numpy arrays and torch tensors."""
    return {"WT": lbl > 0, "TC": (lbl == 2) | (lbl == 3), "ET": lbl == 3}


def dice(p, g):
    if p.sum() == 0 and g.sum() == 0:
        return 1.0
    return 2 * (p & g).sum() / (p.sum() + g.sum())


def hd95(p, g, spacing=SPACING):
    """95th percentile symmetric surface distance in mm; NaN if either mask is empty."""
    if p.sum() == 0 or g.sum() == 0:
        return np.nan
    pb, gb = p ^ binary_erosion(p), g ^ binary_erosion(g)
    d_to_g = distance_transform_edt(~gb, sampling=spacing)
    d_to_p = distance_transform_edt(~pb, sampling=spacing)
    return np.percentile(np.concatenate([d_to_g[pb], d_to_p[gb]]), 95)

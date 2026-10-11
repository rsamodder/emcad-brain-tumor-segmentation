"""PVT-EMCAD-B2 adapted to 4-channel MRI input and 4-class output.

Only two changes are made to the official EMCAD model:
  1. the first patch-embedding conv is widened from 3 to 4 input channels
     (first 3 channels keep the pretrained weights, 4th = mean of the pretrained 3);
  2. num_classes=4, so all four segmentation heads output 4 channels.
The decoder (MSCAM, LGAG, EUCB), skip connections and heads are used unchanged.
"""
import os
import sys

import torch
import torch.nn as nn


def make_model(emcad_dir, weights, num_classes=4, verbose=True):
    emcad_dir = os.path.abspath(emcad_dir)
    weights = os.path.abspath(weights)
    assert os.path.basename(weights) == "pvt_v2_b2.pth", "EMCADNet expects the file name pvt_v2_b2.pth"
    if emcad_dir not in sys.path:
        sys.path.insert(0, emcad_dir)
    from lib.networks import EMCADNet     # official EMCAD repository

    m = EMCADNet(num_classes=num_classes, encoder="pvt_v2_b2", pretrain=True,
                 pretrained_dir=os.path.dirname(weights))

    ck = torch.load(weights, map_location="cpu")
    sd = m.backbone.state_dict()
    n = sum(1 for k, v in ck.items() if k in sd and sd[k].shape == v.shape)
    if verbose:
        print(f"pretrained keys matched: {n} of {len(sd)} backbone tensors")

    for name, mod in m.backbone.named_modules():
        if isinstance(mod, nn.Conv2d) and mod.in_channels == 3:
            new = nn.Conv2d(4, mod.out_channels, mod.kernel_size, mod.stride, mod.padding,
                            bias=mod.bias is not None)
            with torch.no_grad():
                new.weight[:, :3] = mod.weight
                new.weight[:, 3:] = mod.weight.mean(1, keepdim=True)
                if mod.bias is not None:
                    new.bias.copy_(mod.bias)
            parent = m.backbone
            *path, last = name.split(".")
            for p in path:
                parent = getattr(parent, p)
            setattr(parent, last, new)
            if verbose:
                print("replaced first conv:", name)
            break
    return m

#!/usr/bin/env bash
# Clone the official EMCAD repository and download the original-format PVTv2-b2 ImageNet weights.
set -e
[ -d EMCAD/lib ] || git clone https://github.com/SLDGroup/EMCAD.git EMCAD
touch EMCAD/lib/__init__.py
mkdir -p pretrained_pth/pvt
[ -s pretrained_pth/pvt/pvt_v2_b2.pth ] || \
  wget -q -O pretrained_pth/pvt/pvt_v2_b2.pth \
  https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b2.pth
ls -la pretrained_pth/pvt/pvt_v2_b2.pth

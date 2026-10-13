#!/usr/bin/env bash
set -eu
cd /home/gaob/Dual-RGBT
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
exec conda-envs/dual_rgbt/bin/python scripts/train_executor.py \
  --run stage_a_spatial_full_20261013 --epochs 5 --samples 10000 --batch-size 8 --workers 4 \
  --resume runs/stage_a_spatial_20261013/latest.pth

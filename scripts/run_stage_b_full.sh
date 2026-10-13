#!/usr/bin/env bash
set -u
cd /home/gaob/Dual-RGBT
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
conda-envs/dual_rgbt/bin/python scripts/train_policy.py \
  --executor runs/stage_a_spatial_full_20261013/best.pth \
  --run stage_b_refreshed_20261013 --samples 10000 --epochs 5 --batch-size 8 --workers 4 --seed 2027
status=$?
printf '%s\n' "$status" > runs/stage_b_refreshed_20261013.exit_code.txt
exit "$status"

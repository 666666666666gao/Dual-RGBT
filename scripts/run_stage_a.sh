#!/usr/bin/env bash
set -eu
cd /home/gaob/Dual-RGBT
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
exec conda-envs/dual_rgbt/bin/python scripts/train_executor.py \
  --run stage_a_diagnostic_20261013 --epochs 2 --samples 2000 --batch-size 4 --workers 4

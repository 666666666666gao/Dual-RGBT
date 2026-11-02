# Server25 GPU0 Incident

Read-only SSH audit: 2026-11-02 00:22-00:28 UTC
(2026-11-01 evening America/New_York). Host alias: 2025.

- SSH works. RAM has about 237 GiB available; no evidence of RAM exhaustion.
- GPU0, RTX 3090, PCI 0000:3b:00.0, UUID
  GPU-3a8ce0ab-ed75-1685-29a0-316cd32572bc:
  NVIDIA reports "Unable to determine the device handle ... Unknown Error".
  lspci reports revision ff. Sysfs current_link_speed is Unknown and
  current_link_width is 63. These are abnormal device-access observations;
  the exact hardware/driver cause is not established.
- Individual NVIDIA queries for GPU1/2/3 succeed: each 2 MiB, utilization 0%.
- Root filesystem: 916G total, 863G used, 6.6G available, df reports 100%.
  Inode use is 6%. /data2 is separate: 7.3T total, 904G available.
  Root space pressure is independently confirmed; it is not proven to have
  caused the GPU failure. /home/gaob uses about 65G.
- GPU0 CADTrack task train_cadtrack_ep20_lasher_r7_20261101 remains marked
  RUNNING. Actual child 779800 is Rl with 99.8% CPU.
  CUDA_VISIBLE_DEVICES=0. Its log stopped at epoch13, step850/2500:
  2026-11-01 22:37:20 +0800 (14:37:20 UTC), almost 10 hours before audit.
  This process is stalled; RUNNING is not evidence of training progress.
- The requested E0 fine-tune is a separate GPU1 task:
  train_e0_executor_ep20_r1_20261101 is COMPLETE/exit0 at
  2026-11-01 16:38:45 UTC. Log reaches epoch40, step99950 and
  EPOCH_COMPLETE 40. COMPLETE.json and epoch_036.pth through epoch_040.pth
  exist in /data2/gaob/Dual-RGBT/e0_executor_ep20_lasher_r1_20261101.
  These are training completion observations, not benchmark scores.
- Kernel logs are inaccessible to gaob. sudo -n dmesg requires a password;
  journalctl explicitly reports restricted visibility. No absence-of-Xid
  conclusion can be drawn. Readable PCI AER counters are zero, which does
  not establish a healthy device.

No workloads were launched, processes stopped, files deleted, GPU reset,
driver reload or server reboot performed.

Next diagnostic action requires administrator access to kernel logs around
2026-11-01 14:37 UTC and afterward. Preserve logs and checkpoints before
planning any reset/reboot. Do not automatically restart the stalled task.

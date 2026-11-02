# P26 LasHeR TRAIN Root Path Audit

Audit time: 2026-11-02 19:28-19:38 UTC

## Findings

- P26 manifest `/data2/gb/Dual-RGBT/data/manifest.json` lists 882 TRAIN
  sequences and 97 dev sequences.
- The configured TRAIN root
  `/data2/gb/Dual-RGBT/data/LasHeR/train` is a symlink resolving to
  `/data/wangwj/dataset/LasHeR0428/trainingset`.
- The resolved root contains 979 sequence directories. All 882 sequence
  names in the TRAIN manifest are present.
- `/data2/gb/Dual-RGBT/data/LasHeR-0428/trainingset` contains exactly the
  97 sequence names in the dev manifest. It must not be used for training.
- The DropMoE ep20 checkpoint is absent at
  `/data2/gb/Dual-RGBT/checkpoints/CADTrack_ep0020.pth.tar`.
- The P26 E0 parent checkpoint exists and matches the P25 bound input SHA256
  `ee60704e7fe1e281584430332dea7807a335bd4d723a769e4783217b0ac6f56c`.

## Correction

A previous preflight tested a different, incorrect dataset path and
concluded that P26 lacked LasHeR TRAIN data. That conclusion was wrong:
`find` did not follow the configured symlink. The current inventory confirms
all TRAIN sequence directories are present. This check is directory-level;
it does not hash every image file.

No training task was launched on P26. The reviewed P1A pair is already
running on P25, while P1B depends on P1A completion and requires its own
source review. The missing ep20 checkpoint also prevents direct deployment
of that reviewed P1A configuration to P26 without a newly bound plan.

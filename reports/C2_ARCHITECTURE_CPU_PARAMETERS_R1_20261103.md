# C2 Architecture CPU Reconstruction and Parameter Count

This is a read-only inspection of the already completed ten-update C2 TRAIN
smoke artifacts on both hosts. It does not use an active full-training
checkpoint, launch a GPU task, collect states, or produce benchmark evidence.

| Variant | Total parameters | Backbone | Decode | Box head | Preview | Quality | Gain | Missing |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| zero | 103,374,215 | 86,045,184 | 10,619,136 | 6,474,117 | 14,336 | 4,865 | 216,577 | 0 |
| FiLM | 103,868,807 | 86,045,184 | 10,619,136 | 6,474,117 | 14,336 | 4,865 | 216,577 | 494,592 |

The zero and FiLM counts are identical on P25 and P26. The FiLM difference
is exactly the learned missing-stream conditioning module. Each smoke plan
had 103 source bindings; its native receipt was COMPLETE/exit0 with ten
routes and strict reload. The CPU reconstruction loaded each `epoch_001.pth`
with `C2DualRGBT(..., **saved["c2_config"])` and `strict=True`, verified the
serialized config and finite state, and left CUDA uninitialized.

Host-specific raw report files:

- `reports/c2_architecture_cpu_parameters_r1_20261103/dropmoe.json`
- `reports/c2_architecture_cpu_parameters_r1_20261103/sot.json`

Both reports bind the plan, checkpoint and smoke-receipt hashes. The source
and smoke receipts are historical admitted artifacts; this inspection does
not upgrade them to full-executor acceptance. Full C2 completion, last-five
checkpoint retention, actual C2 reconstruction at the terminal checkpoint,
policy refresh, complete TEST inference and FPS/MAC measurement remain
pending.

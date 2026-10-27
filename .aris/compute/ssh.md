# SSH compute environment — active server26

Provider: SSH alias2026; physical GPU0/1/2/3, NVIDIA RTX3090. GPU2/3 were admitted after their original owners exited on2026-10-26.
Project: /data2/gb/Dual-RGBT.
Conda prefix: /data2/gb/Dual-RGBT/conda-envs/dual_rgbt.
Actual versions: Python3.7.16 / PyTorch1.12.1+cu113 / timm0.9.12 / NumPy1.21.5.
Relocatable archive SHA256: 9272d842a1163e63e814427ef1675cb4e56f94007b99286faa852a054565d919.

Acceptance: INFERENCE_ENVIRONMENT_ACCEPTED. Actual GPU0 seed2027 CUDA
forward/backward/AdamW/update/strict-reload kernel witness passed. The actual
11-frame train-split full tracker smoke exited0, with exact box, route,
template and history reload. Fresh documented invocation R2 passed without
further correction. R1 PS5 module-resolution failure and its logs are retained;
use the documented current PowerShell7 invocation.

Evidence: reports/SERVER26_ENVIRONMENT_RELOCATION_20261026.json,
reports/SERVER26_GPU_KERNEL_WITNESS_20261026.json,
reports/SERVER26_DOCUMENTED_LAUNCH_VERIFICATION_20261026_R2.json,
reports/SERVER26_ENVIRONMENT_ACCEPTANCE_20261026.json.

Warm-reuse this accepted inference environment. Any new training experiment
still needs its separate real forward/backward/update/reload admission.
The old server25 active project was retired to its verified source archive;
do not launch old25 jobs. Runtime PIDs, observation nodes and results belong
in the single canonical HANDOFF.md.

2026-10-27 user capacity update: target approximately18GiB per GPU, staged meaningful configuration concurrency. Same accepted conda environment; no package/spec rebuild. Host allocator environment changes require separate paired real-update/memory admission before new tasks use them. The original eight processes remain unchanged.

2026-10-27 user reauthorized additional server25 GPU0/1/2/3 at /home/gaob/Dual-RGBT. Its independent ledger is ssh-server25.md. Existing26 jobs/paths/sources stay unchanged. Old25 archive remains retired; only new25 task identities may launch.

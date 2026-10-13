# SSH compute environment

Provider: server2025, only physical GPU0/1, NVIDIA RTX3090.
Environment specification: Python3.7 / PyTorch1.12.1+cu113 / timm0.9.12,
relocated from the working pytracking conda environment with conda-pack.
Prefix: `/home/gaob/Dual-RGBT/conda-envs/dual_rgbt`.
Status: packing in progress; not yet accepted as ready.
Acceptance: actual seeded CUDA forward/backward/AdamW/strict checkpoint reload
on LasHeR training images, then independent documented-invocation verification.
Do not rebuild a successfully witnessed unchanged environment.

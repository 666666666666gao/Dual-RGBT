# SSH compute environment

Provider: server2025, only physical GPU0/1, NVIDIA RTX3090.
Environment specification: Python3.7 / PyTorch1.12.1+cu113 / timm0.9.12,
relocated from the working pytracking conda environment with conda-pack.
Prefix: `/home/gaob/Dual-RGBT/conda-envs/dual_rgbt`.
Status: relocation complete; GPU0 actual M0 passed in 15.93 seconds.
Kernel witness: reports/sanity_gpu0.json, 10 real optimizer updates, strict reload
bit equal; actual prefix is the home project environment, peak 2999788544 bytes.
Agent-follows-doc verification: PASS, native session80288 exit0, real GPU1 M0
14.78 seconds. Record: reports/environment_doc_verification.md. No command fixes.
Environment is accepted for this executor. Warm-reuse the current environment.
Acceptance: actual seeded CUDA forward/backward/AdamW/strict checkpoint reload
on LasHeR training images, then independent documented-invocation verification.
Do not rebuild a successfully witnessed unchanged environment.

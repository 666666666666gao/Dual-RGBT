# Server25 document verification — 2026-10-27

**DOC_VERIFICATION_PASS**. A fresh agent executed the exact PowerShell command in `docs/SERVER25_REJOIN_20261027.md` once. The native tool returned exit code **0** (chunk `b6f1d2`, 17.1193311 seconds) and `ENVIRONMENT_KERNEL_AND_REAL_TRACKER_PASS`. No edits, retries or fallback were used.

- Prefix: `/home/gaob/Dual-RGBT/conda-envs/dual_rgbt`.
- Python 3.7.16; Torch 1.12.1+cu113; timm 0.9.12; NVIDIA GeForce RTX 3090; physical GPU0.
- Tiny seeded CUDA kernel: finite loss 1.087954044342041, one optimizer update, strict reload bit-equal.
- Fixed B checkpoint: **103218887 total parameters**; TRAIN sequence `2up`, 11 real image pairs, 10 tracked frames.
- Tracker reload: box/route and templates/history bit-equal; zero optimizer updates and backward calls.
- Measured smoke FPS: 4.711286213743169 (2.122562618002121 seconds for 10 frames). **This is environment smoke evidence, not formal benchmark FPS or accuracy.**
- Six existing PyTorch `__floordiv__` deprecation warnings are retained in the raw output. No documented requirement diverged.

The command did not query server26, start formal training, restart old queues, or modify sources/environment. GPU availability and user authorization were supplied by the parent task and read from the provider ledger; no additional GPU query was issued.

Review attribution: `same-family`, `provisional`; actual model and reasoning **UNATTESTED**. This validates the documented environment invocation on GPU0; it does not establish full tracker-training viability or four-GPU admission.

## Evidence

- Detailed report: `reports/SERVER25_ENVIRONMENT_DOC_VERIFICATION_20261027.json`.
- Raw native request/response and unaltered combined output: `.aris/traces/run-experiment/2026-10-27_server25_doc_verification/`.
- Independently produced remote run: `/home/gaob/Dual-RGBT/runs/server25_environment_doc_witness_20261027`.
- Source review: `reports/SERVER25_ENVIRONMENT_WITNESS_SOURCE_REVIEW_20261027.json`.
- Parent native result: `reports/SERVER25_ENVIRONMENT_PRIMARY_NATIVE_20261027.json`.

```powershell
ssh -T -o BatchMode=yes -o ClearAllForwardings=yes -o ConnectTimeout=10 2025 'env CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID OPENCV_FOR_THREADS_NUM=1 /home/gaob/Dual-RGBT/conda-envs/dual_rgbt/bin/python /home/gaob/Dual-RGBT/scripts/server25_environment_witness.py --run server25_environment_doc_witness_20261027'
```


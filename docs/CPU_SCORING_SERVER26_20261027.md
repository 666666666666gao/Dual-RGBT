# Server26 offline CPU scoring environment

Prefix: `/data2/gb/Dual-RGBT/conda-envs/cpu-scoring`.
Spec: `.aris/compute/cpu-scoring-env.json`, identical to the accepted server25
CPU environment. Python3.10.14, NumPy1.26.4, Matplotlib3.8.4; no Torch.
The training environment remains unchanged.

The offline bundle contains the same 72 conda archives, original explicit URLs
and MD5 hashes, per-file SHA256 hashes, micromamba2.0.5, and the unchanged
user-supplied scoring toolkit. Bundle SHA256:
`e70ee61f63cce3a2319a6d45e22641d96736471d1d7b29e6c2ecfa01d78506b4`.
Archives come from the accepted server25 cache. No new dependency solve or
internet download is performed. Local file URLs used during installation are
recorded separately from original package URLs.

Build invocation (once only; refuses an existing prefix):

```bash
python3 /data2/gb/Dual-RGBT/tools/install_cpu_scoring_offline.py --project /data2/gb/Dual-RGBT --bundle bootstrap/cpu_scoring_offline_20261027 --protocol reports/VRAM18_EVALUATION_PROTOCOL_20261027.json --run cpu_scoring_offline_server26_20261027
```

After build completion, run this independent verification verbatim from
Windows PowerShell. It writes a new file and refuses to overwrite it:

```powershell
ssh -T -o BatchMode=yes -o ClearAllForwardings=yes -o ConnectTimeout=10 2026 "env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg MPLCONFIGDIR=/data2/gb/Dual-RGBT/.cache/matplotlib-cpu /data2/gb/Dual-RGBT/conda-envs/cpu-scoring/bin/python /data2/gb/Dual-RGBT/tools/check_cpu_scoring_environment.py --project /data2/gb/Dual-RGBT --protocol reports/VRAM18_EVALUATION_PROTOCOL_20261027.json --output /data2/gb/Dual-RGBT/reports/cpu_scoring_doc_verification_server26_20261027.json"
```

Expected: native exit0 and `CPU_SCORING_WITNESS_PASS`. The witness checks a
seeded NumPy calculation/save/reload, toolkit source seals, all test annotation
identities (LasHeR245, RGBT210210, RGBT234234), and seven analytic metric
fixtures. It does not read checkpoints or model predictions, run inference,
or produce tracking accuracy. Use the original NN Python for model admission
and inference, and this CPU Python for scoring saved predictions.

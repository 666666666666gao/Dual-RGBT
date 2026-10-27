# CPU scoring environment

This environment runs the user's unchanged RGBT toolkit. It contains no Torch
and performs no GPU inference. Keep the accepted `conda-envs/dual_rgbt` training
environment unchanged. Its Python3.7 cannot import the toolkit's `list[str]`
annotations. The separate scorer uses Python3.10, NumPy1.26.4 and
Matplotlib3.8.4. NumPy matches the existing local CPU scorer.

Spec: `.aris/compute/cpu-scoring-env.json`.
Bootstrap: the official conda-forge micromamba2.0.5 package; no shell
initialization or system package installation. Tool, conda root/cache and
environment all stay inside the project.

## Build once on server25

The initial direct download in `runs/cpu_scoring_build_20261027` failed with
curl exit28 after180 seconds at about12KB/s, before creating an environment.
Its status and log are preserved. The second attempt used HTTP CONNECT against
a SOCKS endpoint and failed immediately with native exit56. The registered
replacement is `runs/cpu_scoring_build_r3_20261027`, using the correct SOCKS5h
protocol through a temporary isolated SSH tunnel. The package source and
versions are unchanged. Do not start another build
against that prefix. Build status and exact conda package URLs stay under
the run; the current log is `reports/cpu_scoring_build_r3_20261027.log`.
The proxy variables below apply only to installation and are unnecessary for
later scoring. The temporary tunnel is closed after the download/install ends.

R3 created the environment successfully. Its original post-install export used
`list --explicit`, unsupported by micromamba2.0.5, leaving a stale RUNNING
status after the builder exited. The correct command is `env export --explicit`.
The renderer is corrected; the separate post-install completion receipt records
successful export and the parent CPU witness without rebuilding the environment.
Original statuses/logs remain unchanged.

```bash
env HTTP_PROXY=socks5h://127.0.0.1:17898 HTTPS_PROXY=socks5h://127.0.0.1:17898 ALL_PROXY=socks5h://127.0.0.1:17898 python3 /home/gaob/Dual-RGBT/tools/prepare_cpu_scoring_env.py --project /home/gaob/Dual-RGBT --run cpu_scoring_build_r3_20261027 --protocol reports/SERVER25_JOINT_EVALUATION_PROTOCOL_20261027.json
```

## Independent verification after the build

Run the following command verbatim from Windows PowerShell. This is a fresh
CPU witness, not another installation. It writes a new verification file and
refuses to overwrite an existing one.

```powershell
ssh -T -o BatchMode=yes -o ClearAllForwardings=yes -o ConnectTimeout=10 2025 "env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg MPLCONFIGDIR=/home/gaob/Dual-RGBT/.cache/matplotlib-cpu /home/gaob/Dual-RGBT/conda-envs/cpu-scoring/bin/python /home/gaob/Dual-RGBT/tools/check_cpu_scoring_environment.py --project /home/gaob/Dual-RGBT --protocol reports/SERVER25_JOINT_EVALUATION_PROTOCOL_20261027.json --output /home/gaob/Dual-RGBT/reports/cpu_scoring_doc_verification_20261027.json"
```

Expected exit0 and `CPU_SCORING_WITNESS_PASS`: deterministic NumPy
calculation/save/reload, supplied-toolkit code seals, manifest/annotation
identity for LasHeR245, RGBT210210 and RGBT234234, and seven analytic synthetic
metric checks. These checks produce no tracking performance result and read
no checkpoint or model prediction. Native failed Python3.7 import remains
preserved in `reports/SERVER25_EVALUATION_TOOLKIT_GT_IDENTITY_20261027.json`.

After acceptance, run model admission/NN tracking with the original training
Python; run `tools/joint_candidate_results.py score` and `freeze` using this CPU
Python. Full-C/native0, smoke, complete DEV, result audit and frozen test
selection remain mandatory. A successful environment witness alone does not
authorize model evaluation or establish PR/NPR/SR/FPS.

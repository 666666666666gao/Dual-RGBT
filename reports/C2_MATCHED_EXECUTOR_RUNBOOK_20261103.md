# C2 Matched Executor Runs

Full zero and FiLM runs on both hosts have already launched and are live.
Do not repeat the preparer, launcher, smoke or start-witness archiver.
P25 first epochs completed in1054.50s/1054.41s; P26 in1283.04s/1283.20s.
All four have2500 sequential updates and250 per each of ten routes.
At07:09UTC both P25 runs were in epoch2; P26 epoch1 passed at07:11UTC.
Actual timing, evidence and next C2-specific policy-loader boundary:
`reports/C2_FIRST_EPOCH_TIMING_20261103.md`.

Both hosts have completed all four native ten-update TRAIN smokes:
zero/no-LoRA, missing/FiLM, shared LoRA and routed LoRA.
Exact evidence resides in
`reports/c2_training_smoke_r2_20261103/<branch>/<variant>/runtime/`.
Do not repeat those smokes or overwrite their archives.

The first full comparisons are zero versus FiLM, each initialized from its
own branch's fixed P1A epoch20 executor. Both use the same new full-timeline
causal sampler and seed2027, B8, four workers, 20000 pairs per epoch,
20 epochs, backbone LR1e-5, head/new-parameter LR1e-4, milestone10,
weight decay1e-4 and gradient clipping0.1. Training retains epochs16-20.
Policy modules remain frozen. These trained executors still require
versioned policy refresh before claims about a fully adapted tracker.

Original disabled drafts are preserved in
`reports/c2_training_full_r1_20261103/<branch>/<variant>/`.
Do not rerun the preparer for existing variants.
Concrete review files must be SOURCE_PASS/0 with stage
`P2_C2_executor_training`, the exact host-specific 103-source map and
both corresponding canonical plan bindings.

Historical invocation already executed successfully on both hosts; do not repeat:

```powershell
E:/anaconda/python.exe -B work/launch_c2_executor_full.py dropmoe --variants zero film
E:/anaconda/python.exe -B work/launch_c2_executor_full.py sot --variants zero film
```

The launcher checks all four branch smokes have completed and wrappers have
exited, exact input/source hashes, current same-GPU C2 native tasks,
at least20480MiB free GPU memory and12GiB free disk. It writes only the new
review and finalized full plans into the existing isolated C2 working copy,
then executes canonical full admission before invoking the native wrapper.
GPU1 runs zero; GPU2 runs FiLM. Check existing launch logs/native tasks before
any retry. A failed attempt must retain its evidence.

P25 isolated root:
`/data2/gaob/Dual-RGBT/c2_training_dropmoe_r2_20261103`.
P26 isolated root:
`/data2/gb/Dual-RGBT/c2_training_sot_r2_20261103`.
Run roots are the corresponding parent `c2_runs/`.
Read actual native status, child PID, log/CSV and GPU ownership; a LAUNCHED
receipt alone does not establish optimizer updates or completed epochs.
Use180-300s checks at predicted progress boundaries, not repeated short polls.

The full trainer flushes its loss CSV every25 updates. Estimate epoch duration
from actual varied TRAIN batches after startup; ten-update smoke timing is
only a rough capacity/timing observation, not full-loader throughput or
convergence. Keep the two branch results distinct.

The old prefix-sampled P1A executor is not a matched control for the new
sampler. TEST remains exploratory and ineligible for training, threshold
tuning or checkpoint/branch selection. Keep source review, native evidence,
training convergence and full benchmark evidence separate.

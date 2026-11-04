# C2 policy evaluation entry

`scripts/eval_c2_policy_completed.py` is LOCAL source preparation for
evaluation after a genuine completed C2 policy FIT. It has not been deployed
or run with actual weights. No concrete evaluation plan is admitted.

The independent entry takes one enabled `--plan`. Stage must be
`P2_C2_policy_evaluation`. Its review must authenticate exact source bytes
and the canonical plan binding excluding only `launch_allowed` and `review`.
The plan names the executing root/task/run/GPU/argv, source map, benchmark
manifest, fixed epoch005 checkpoint and the loader's eight FIT evidence pins.
Host/variant/C2 task/cost must match the FIT. Cost is fixed at0.05.

Runtime requires its actual RUNNING native wrapper with matching parent PID,
task, command and CUDA GPU environment. It delegates full FIT terminal and
checkpoint authentication to `load_completed_policy`, which reconstructs the
original C2 architecture. It never invokes the old P1B/DualRGBT evaluator.

The unchanged Tracker protocol uses search256/template128/two templates.
Full TEST must match245seq/220703frames LasHeR,210/104706 RGBT210 or
234/116650 RGBT234. A smoke may only consume the first11 frames of the first
LasHeR TRAIN sequence. There is no dev, checkpoint selection, cost tuning,
optimizer or backward call.

Prediction/GT/route trace/progress/per-frame timing exports follow the
existing E0 protocol. RGBT234 retains separate visible/infrared annotations;
RGBT210 exports its shared GT to both modality files. GT is written after
online tracking, never passed to the tracker beyond the initial box.
Timing includes decode and initialization, excludes model load and is under
concurrent system load. Parameter count is recorded. These sources alone
establish neither measured FPS/MACs nor benchmark accuracy.

All sequences must finish with positive finite predictions and timing.
Model state must remain unchanged and gradients absent. All three precision
flags must be false before writing SMOKE/COMPLETE, which binds config,
checkpoint and plan hashes. A partial failure leaves artifacts without a
successful completion receipt. Existing runs cannot be overwritten.

Focused local verification:

```powershell
& E:/Anaconda/python.exe -B -m pytest -q tests/test_eval_c2_policy_completed.py tests/test_c2_policy_evaluation.py tests/test_c2_policy_refresh.py
```

Actual result:157 passed. Tests cover plan/source/input/fixed recipe gates,
full count rejection, three-benchmark artifact export, first11 TRAIN smoke,
and mocked entry lifecycle with successful receipt versus state mutation or
invalid box. This is CPU synthetic/mocked evidence, not an actual C2 model
or GPU witness.

Exact source-only review is SOURCE_PASS/0:
`reports/P2_C2_POLICY_EVALUATION_ENTRY_SOURCE_REVIEW_R1_20261104.json`,
SHA256 `32558730b14525bcb2e8efa1ea28fd84f59fc2366d11213caa7628c513e2ae41`.
Attribution is UNATTESTED/same-family/provisional. Execution bindings are
empty and scientific acceptance is false. This grants no runtime plan.

Future work after genuine FIT COMPLETE0: pin authentic terminal inputs and
the exact host-specific inference dependency map in a separate evaluation
review; create a new isolated root, require actual CPU model reconstruction,
own TRAIN smoke/native exit0 and fresh runtime admission before full TEST.
A separate C2-aware scoring/archive path must authenticate this entry's
plan/config/terminal schema before scoring once. Existing P1B scorers do not
automatically admit C2 outputs. Preserve all active FIT/replay/evaluation
snapshots and every failed/partial artifact.

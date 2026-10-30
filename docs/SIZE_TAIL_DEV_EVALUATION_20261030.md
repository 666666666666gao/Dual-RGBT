# Size-tail dev evaluation

This experiment retains CandidateEvidencePolicy and its existing tracker. Only
`executor.box_head.conv5_size.weight` and `.bias` may differ from the completed
server25 quality-k8 parent (66 parameters, 103,218,887 total parameters).
The parent SHA256 is
`21e07cd1f645c6f7237e063c81cf5d60f6a14bca710e0180e8a3ca29dc827073`.

The separate evaluator `scripts/eval_size_tail.py` reuses the original model,
tracker, first-frame initialization, image reader and query-evidence collector.
It fixes cost=0.15 and K=8 through the recorded plan/reference configuration.
It does not edit or relax the registered formal candidate evaluation contract.
Adapted weights require a completed native training receipt, strict reload proof,
exact checkpoint hash and exactly the two permitted changed tensors. These runs
are dev diagnostics; they are not eligible for test selection.

For the zero-update parent control, each saved prediction line and each action /
modality pair must equal the original completed parent dev run. A small control
uses the first 11 dev frames. A full control covers all 97 sequences / 54,646
frames and fails at the first differing sequence. Timings are not compared:
GPUs also run recovery evaluation jobs. Service FPS is not an isolated benchmark.

`tools/score_size_tail.py` checks the native exit, full split coverage, checkpoint,
source and manifest hashes, zero optimizer updates and original ground truth.
It invokes the unchanged `scripts/score_tracking.py` and supplied RGBT toolkit.
Toolkit hashes come from `JOINT_CANDIDATE_EVALUATION_PROTOCOL_20261026.json`.
The result records source, native, prediction/trace and metric-curve hashes.
The CPU watcher polls a running native task every 180 seconds.

## Execution

Use the accepted server25 environment at `/home/gaob/Dual-RGBT`. Explicitly pass
`OPENCV_FOR_THREADS_NUM=1` to the native wrapper launcher: this server's pinned
wrapper version does not add that variable itself. The first control smoke
(`eval_size_tail_parent_control_smoke_20261030`) failed at this environment check,
before prediction; its native FAILED receipt remains intact. The second attempt
has the distinct identity `eval_size_tail_parent_control_smoke_r2_20261030`.

From the project root, the parent control command is:

```bash
OPENCV_FOR_THREADS_NUM=1 conda-envs/dual_rgbt/bin/python scripts/run_task_four_gpu.py \
  --task eval_size_tail_parent_control_dev_20261030 --gpu 1 -- \
  /home/gaob/Dual-RGBT/conda-envs/dual_rgbt/bin/python \
  /home/gaob/Dual-RGBT/scripts/eval_size_tail.py \
  --plan reports/SIZE_TAIL_PARENT_CONTROL_PLAN_20261030.json \
  --run size_tail_parent_control_dev_20261030 \
  --task eval_size_tail_parent_control_dev_20261030 --smoke-frames 0
```

Launch the separate CPU scorer after the native status file exists:

```bash
conda-envs/cpu-scoring/bin/python tools/score_size_tail.py \
  --run size_tail_parent_control_dev_20261030 \
  --output reports/size_tail_dev_20261030/parent_control.json --wait
```

Use new run/task identities for retries. Do not overwrite failed evidence or edit
source files while a source-pinned evaluation is active. Formal epoch training,
last-five full-dev selection and convergence early stopping remain separate
pending work; four-step sanity loss is not an accuracy or convergence result.

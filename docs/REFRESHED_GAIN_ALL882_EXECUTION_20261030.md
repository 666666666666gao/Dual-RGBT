# All882 refreshed gain execution

Provider25: `/home/gaob/Dual-RGBT`. Reuse the accepted NN and CPU-scoring
environments. No package, trainer, evaluator, model or original recipe changes.
The preregistered scientific protocol remains
`docs/REFRESHED_GAIN_ALL882_PROTOCOL_20261030.md`.

Entry point:

```bash
/home/gaob/Dual-RGBT/conda-envs/dual_rgbt/bin/python /home/gaob/Dual-RGBT/tools/run_refreshed_gain_all882.py
```

Launch only after independent source review has accepted this exact queue,
preparer and unchanged trainer/evaluator/scorer hashes in
`reports/REFRESHED_GAIN_ALL882_SOURCE_REVIEW_20261030.json`.
Only one controller instance may own this run.

The controller waits for the existing E6 queue's verified process and complete
state, polling every 180 seconds. The preparer requires all sixteen audited
complete caches, exact disjoint coverage of 882 TRAIN sequences, fixed706/176
partition, original native hashes and the original parent-loss recipe.

The two-epoch smoke runs on GPU2 with at least8GiB free. This is gain-head
training. The original parent initializes its four trainable head tensors;
frozen E6 outputs verify the recorded features. The unchanged trainer checks
finite real optimization, strict reload, unchanged frozen weights and inputs.
Insufficient recovery-state coverage ends with `ADMISSION_REJECTED`.
No threshold, split or input is changed to force admission.

After a passing smoke, the formal plan binds the exact same inputs, recipe
and source. Formal training stays onGPU2 and requires free memory at least
the measured smoke peak reserved plus1GiB. The existing stopping rule runs
at least8 and at most30epochs, retaining the latestfive checkpoints.

Each retained weight then receives an independent11frame dev smoke followed
by all97sequences/54646frames. Three evaluation slots share GPU2/3/3, each
requiring3GiB free before launch. GPU0/1 remain reserved for B640/B576 training.
Official scoring starts only after the owning inference wrapper has exited.
Every retained weight must have its own score; these shared timings are not
isolated FPS measurements.

Queue state: `reports/REFRESHED_GAIN_ALL882_QUEUE_20261030.json`.
Scores: `reports/refreshed_gain_all882_dev_20261030/eNNN.json` and curves.
Native wrappers: `runs/pipeline_tasks/{train,eval}_<run>/status.json`.
These are new run identities; the matched100 negative admission is preserved.

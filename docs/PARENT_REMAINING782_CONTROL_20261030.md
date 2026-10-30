# Frozen parent control over the remaining 782 TRAIN sequences

The existing parent-policy cache covers the first 100 seeded TRAIN sequences.
E6 refresh covers all 882. Complete the missing parent-policy coverage so
future all882 comparisons can hold sequence membership and partition fixed.
This collection does not establish a tracking-accuracy improvement.

Reuse `collect_frozen_query_features.py` without changing its causal replay,
four-family sampling, label stride 10, seed 2027 or parent checkpoint.
Derive each shard from the registered remaining782 E6 plan, preserving its
fit/holdout sequence membership. Do not recollect the existing first100.
No dev or test sequences enter this cache.

Run `tools/run_parent_remaining_control.py --prepare` with the existing
dual_rgbt environment on each provider. It checks the originating registry,
manifest and parent evaluation contract before creating new plans.
After independent source acceptance, run the same script without arguments.
Each shard first runs a real 31-frame native smoke. Require bit-equal parent
trajectory, frozen tensors, no optimization, artifact audit and exact sampled
frame coverage. Only then register and launch its complete collection.
The original collector checks the full plan's binding to its exact smoke.
The queue reads terminal native status only after its owning wrapper exits.

Server26 owns even shards 0/2/4/6, assigned to GPU0/1/2/3 respectively.
Server25 owns odd shards 1/3/5/7, assigned to GPU2/2/3/3 respectively.
Server25 GPU0/1 remain available for the existing B640/B576 training.
Additional GPU3 full-dev jobs share that device with these low-memory
collectors; no existing wrapper, source or recipe is modified.

The queue records all native identities and audits in
`reports/PARENT_REMAINING782_CONTROL_QUEUE_20261030.json`.
Poll at 180 seconds; use measured completion estimates for manual checks.
Each provider retains its original receipts and data paths.

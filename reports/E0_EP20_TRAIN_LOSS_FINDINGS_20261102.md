# Completed E0 TRAIN loss and runtime audit

The actual P25 task train_e0_executor_ep20_r1_20261101 is COMPLETE/exit0.
The current config, steps.csv, native status and run.log hashes exactly
match E0_EP20_TRAIN_COMPLETE_P25_20261102.json. A fresh CSV audit checks
100000 consecutive updates,40epochs,2500updates/20000sampled pairs per
epoch and250updates per each of the ten modality/action routes. Every
logged loss is finite. Raw audit: E0_EP20_TRAIN_LOSS_AUDIT_20261102.json.

The fixed recipe used seed2027,batch8,executor LR1e-5/backbone LR1e-6.
MultiStepLR steps at epoch end: the first decay applies from epoch11
(executor1e-6/backbone1e-7), the second from epoch31
(executor1e-7/backbone1e-8). No recipe or weight changed in this audit.

| Epochs | Mean TRAIN loss | Mean minutes per epoch |
|---|---:|---:|
|1-5|2.2315552861|20.0405|
|6-10|2.1780895329|22.1353|
|11-15|2.1286924358|19.9047|
|16-20|2.1210167889|18.6175|
|21-25|2.1106107509|17.9572|
|26-30|2.1186514282|18.0067|
|31-35|2.1044774384|17.6728|
|36-40|2.1175793430|17.2050|

1. Loss decreases early and then fluctuates around2.11-2.13. The last
   five-epoch block is higher than31-35, so later blocks do not show
   monotonic improvement. The first-ten/last-ten mean losses are
   2.2048224095/2.1110283907, a4.2540% relative reduction. This is
   descriptive TRAIN loss, not a tracking-accuracy gain or proof of an
   optimal stopping epoch. Each epoch samples different seeded pairs,
   and non-STOP route losses include correction supervision.
2. Mean actual epoch duration is18.9424746594minutes; the last five
   average17.2049580018minutes. Native task duration is12.6341948340hours.
   Per-step logged time excludes loader waiting and is not inference FPS.
   These timings describe this batch8 recipe and concurrent server load,
   not a universal LasHeR epoch cost.
3. Directory inventory contains exactly epoch036-040, each1237700041bytes.
   They remain dependencies of fixed full-benchmark evaluations. No
   retrospective TRAIN-loss or TEST-based checkpoint deletion/promotion
   is performed. Their individual benchmark evidence is recorded in
   separate full-result findings.

No preregistered early-stop rule or independent stopping-epoch evidence
exists for this completed run; convergence_established remainsfalse.
There is no evidence here to justify extending beyond the finished40epochs.
The immediate next work remains full LasHeR evaluation and completion of
the already fixed relative-M2 TRAIN collection/training, with any future
stopping rule specified using TRAIN evidence before a new run.

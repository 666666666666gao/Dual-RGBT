# M1 Regularization Control

The baseline M1 TRAIN run has a measured fit/component-holdout gap. Between
epochs5 and12, fit macro recall rises from68.3946% to78.9976%, while component
holdout macro recall drops from53.1125% to47.1850%. Holdout cross entropy
rises from0.650104 to0.797714. This motivates one regularization control,
without evidence that the control will improve tracking.

The control changes only AdamW weight decay from0.0001 to0.001. The existing
117507-parameter head, six reviewed source files,882-sequence paired TRAIN
collection,706/176 sequence partition, fit-only normalization, seed2027,
12epochs, batch4096 and learning rate0.001 are unchanged. TEST scores do not
enter the decision. The fixed epoch5 head used by the six active full
evaluations remains unchanged.

The original GPU training/update/strict-reload admission already executed the
same trainer, head and data-loading path. This is a plan-only recipe control,
not a new implementation or a claim of independent holdout generalization:
upstream E0 saw these component holdout sequences.

Plan review:
`reports/E0_M1_WD1E3_CONTROL_PLAN_REVIEW_R1_20261101.json`,
SHA256 `c9469c0942f8c3229d1fd6f756ee2aa30eae1b5d0d58667b081d0a3b00704001`.
It records SOURCE_PASS/0, UNATTESTED/same-family/provisional, scientific
acceptance false. No Astra execution is claimed.

Draft SHA256:
`65d346ed85dce3319acfc92ffbad035148e7e1e5fe90df39ea77dffe445a2427`.
Final plan SHA256:
`8337c0d27c3b741a0d6120b3b437a1ab98d23141533f6bcf63962c15c9d7054c`.
Native task `train_protected_e0_m1_wd1e3_r1_p26_20261101` was submitted on26
GPU3 after a free-memory check. Wrapper1803940, child1803941 were directly
confirmed alive. Output:
`runs/protected_e0_offline_m1_wd1e3_r1_20261101`.

Submission and a live child are not completed training. After native
COMPLETE/exit0, verify receipt artifacts,216 updates,12 history rows, changed
weights, fit normalization and strict reload; compare all history rows and the
TRAIN-component-selected best epoch. Do not substitute these offline metrics
for full tracking PR/NPR/SR, or automatically replace the active epoch5 head.
Retain best/latest and the final five numbered epochs.

## Completed Result

Native COMPLETE/exit0 after31.85seconds;12epochs/216updates and all nine
receipt artifact hashes verified. Strict reload passed. Saved fit normalization
equals the baseline, and best head weights differ, confirming a real recipe
change. Retained numbered weights are epochs8-12 plus best/latest.

Every epoch's component-holdout macro recall is identical to baseline.
Best remains epoch5: macro53.1125%, lost recall42.3843%.
Final cross entropy changes only from0.797714094 to0.797673239.
Increasing weight decay tenfold did not improve this control's classification
decisions and does not resolve the observed fit/held gap. Do not promote the
new head or launch its full benchmarks on this evidence alone.

Raw config/history/COMPLETE/native receipts are in
`reports/e0_m1_wd1e3_runtime_20261101/`; compact result is
`reports/E0_M1_WD1E3_CONTROL_RESULT_R1_20261101.json`. Full tracking performance
of the original fixed epoch5 head remains pending the existing evaluations.

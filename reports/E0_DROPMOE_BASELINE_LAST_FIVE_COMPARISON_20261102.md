# User Clarified DropMoE Baseline Comparison

User clarification2026-11-02: P25 pretrained CADTrack uses the DropMoE
backbone with the three CADTrack modules removed. The current P25 fine-tune
uses that backbone. The earlier baseline used OSTrack; do not conflate it
with the DropMoE baseline below. Baseline values are user supplied, not
independently reproduced by this comparison.

| Epoch | LasHeR PR | NPR | SR | RGBT234 MPR | MSR |
|---|---:|---:|---:|---:|---:|
| User supplied DropMoE baseline |75.6365|71.7728|60.1257|90.6771|66.8452|
|36|74.8722|71.1310|59.6292|90.2865|67.4090|
|37|74.5482|70.9739|59.4823|90.8606|67.6351|
|38|75.2462|71.2026|59.7761|89.9571|67.2896|
|39|74.9466|71.2422|59.6664|91.0212|67.7770|
|40|75.5874|71.6144|60.0960|90.3620|67.4701|

Epoch39 LasHeR is now native COMPLETE/exit0,245sequences/220703frames.
Official scoring and compute summarization completed once on2026-11-02.
Score SHA414cf7d8c0943f85433fff3675262d5525f6d07f87ffa3be71bf54720ab665e1;
compute SHA8f4a9f3e241ff68283d8248ea136514e1beaca5ae7e82c02ae02a100a96ad149.
All15 fixed full evaluations and their native receipts are now archived.

Descriptive maxima among all five complete LasHeR runs all occur at
epoch40. Relative to the user supplied DropMoE baseline, deltas are
PR-.0491153785, NPR-.1583751702, SR-.0296982023 percentage points.
It does not exceed that baseline on any of the three metrics.

The five RGBT234 evaluations are complete. Descriptive maxima both occur
at epoch39: MPR91.0211715721/MSR67.7770063172, deltas+.3440715721/
+.9318063172 percentage points. Both exceed the user supplied baseline.

RGBT210 has no baseline supplied in this clarification. Its five results
are in E0_EP20_FIXED_LAST_FIVE_SCOREBOARD_20261102.json; do not infer a
same-backbone baseline comparison from the absolute target.

These are descriptive full-TEST maxima requested by the user, not checkpoint
promotion, TEST-based training/tuning, or independent final validation.
LasHeR245seq220703frames and RGBT234234seq116650frames per completed run.
Exact scores/checkpoint/native/compute hashes remain in the fixed inventory.
No three-dataset claim combines different epochs into one checkpoint result.

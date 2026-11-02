# Relative Current-E0 TRAIN Execution Preparation

## Actual GPU Smoke Completed

The R1smoke review finalized SOURCE_PASS/0blockers with66independent negative
checks, SHA256
`353769f3c279f7c84c2ae12cd4b4f915944a507ab106155e475dbc50d17e18a0`.
The bound plan SHA256 is
`dea09e34eaf3d0810f1bb6daec7d9d006a10098557fe7d9d4432ac18f2826d7a`.
Only11added source/test files and bound report/audit/draft/plan were deployed;
all139active collector source files remained unchanged. Actual26CPU15tests
and canonical admission passed before launching.

Native task `train_e0_relative_update_smoke_r1_p26_20261102`, wrapper1330454,
ran onGPU0 from05:30:45to05:31:16UTC and finished COMPLETE/exit0. The actual
six TRAIN prefix sets received5updates; losses were
1.848789/1.663493/1.530694/1.435642/1.362194. Matching parameters changed,
the frozen no_match scalar did not, all required gradients were finite,
and strict save/reload outputs were bit-equal. Total head parameters116,482;
measured peak allocated/reserved GPU memory5,836,288/8,388,608bytes.
Update/check phase took13.24seconds; native task time includes admission
and input checking. No protected E0 model or online policy was updated.

Independent terminal verifier returned RELATIVE_CURRENT_E0_UPDATE_SMOKE_VERIFIED:

- Acceptance `E0_M2_RELATIVE_UPDATE_SMOKE_ACCEPTANCE_R1_20261102.json`,
  SHA256 `d7fbd019ba12ed81e708fd45ecf6a008f164be12b591c20c202545450b738320`.
- COMPLETE receipt SHA256
  `f30c8693b2407f042b62ac4d5e2ce69e7fe7899c24d1f2738418f97b3d2fd828`.
- Native status SHA256
  `2bee929423db8ad3a582dc89bb7e83c63fad6377be0bf6400af05b667a7d9276`.

Config/receipt/native/log/launch are archived under
`e0_relative_update_smoke_r1_20261102/`; local hashes match the actual
verifier's identities. Smoke weights stay remote and are not promoted.
One prefix is from component-held TRAIN: smoke losses are not validation,
and full training must restart from the original epoch9initialization.
No benchmark gain or full training acceptance follows from this smoke.

The separate154source full execution review finalized SOURCE_PASS/0blockers,
SHA256 `b4c5337cc45e55aa45d8604e8316efe6d779fb63e382561c5565eca6d6592e2a`.
Four full-only sources and the review/draft were deployed to26; actual17CPU
tests, all154source hashes and terminal smoke reverification passed.
The full collection acceptance is still required before full training.
At05:34UTC the full supplement still had all8real children live,
199/882completed sequences and94,194/459,435completed frames; current ages
were<=6.9seconds. Three actual E0evaluation children remained live:
ep40LasHeR144seq137,160frames, ep39LasHeR64seq56,764frames,
ep39RGBT210158seq81,735frames. Newest traces were<=16.7seconds old.
No full acceptance, new E0 score or free evaluation slot exists yet.

The sections below record the preceding preparation state; the real smoke
above supersedes their initial pending-launch status.

The actual full current-E0 evidence supplement remains in progress on26GPU0/1.
At2026-11-02 05:18UTC all8actual collector children and3E0evaluation children
were live. Collection completed126/882sequences and55,014/459,435frames;
current in-progress frames bring the processed count to61,522. Current
updates were0.6-4.1seconds old. None of these counts is full acceptance.

E0ep40LasHeR had141completed sequences/130,919frames; ep39LasHeR63/49,781;
ep39RGBT210151/72,432. Actual newest CSV traces were4.8-7.4seconds old.
GPU0/1 used9,095/15,883MiB, with15,032/8,244MiB free; availableRAM~66.8GiB.
GPU2/3 remain occupied by other existing work and were not changed.
The previously unchanged ep39LasHeR completed count advanced after its long
sequence; no restart or duplicate evaluation was warranted.

## Core Review

The independent engineering report
`E0_M2_RELATIVE_TRAIN_ENGINE_SOURCE_REVIEW_R1_20261102.json` is
ENGINE_SOURCE_PASS/0blockers, final SHA256
`54da4d4eb2ca3076b57d4c1e97eb55fe860fee33f0fd717f7330b858ee42017e`.
Its exact core/helper source hashes were checked locally. Attribution is
UNATTESTED/same-family/provisional and scientific_result_audit_accepted=false.
The report does not grant execution or online integration admission.

## Real Update Smoke

New source files:

- `scripts/train_e0_relative_smoke.py`
- `tools/relative_e0_smoke_contract.py`
- `tools/prepare_relative_e0_smoke.py`
- `tests/test_relative_e0_smoke.py`

The disabled proposal is
`e0_relative_update_smoke_r1_20261102/draft.json`, SHA256
`04b88482b02ee73300bc2e28ce43400bbf33aea22f5186f97e9de8bb037cc70a`.
It binds150sources, GPU0, seed2027, the original epoch9 identity head
`7e7df5d25d3e8667fe38a4ca8f6ec3662308ffcb52a09bd16aecaab59a9cb4be`,
and exactly five AdamW updates with learning rate0.0003/weight decay0.0001.
The six sets come from the three genuinely accepted31frame TRAIN prefixes;
the original24candidate arrays, anchors and labels are reused. Both current
and candidate relative IoUs use float64; original mixed-precision candidate
IoUs are verified using the original collector arithmetic.

The actual26read-only host audit
`E0_M2_RELATIVE_UPDATE_SMOKE_HOST_AUDIT_R1_20261102.json` rehashed all139
existing full-collector source files, verified the original initialization,
reran the canonical accepted prefix verifier, and confirmed new task/run
absence. Task `train_e0_relative_update_smoke_r1_p26_20261102` and run
`runs/e0_relative_update_smoke_r1_p26_20261102` have not been launched.
The complete150source/draft smoke review was requested in the already
authorized reviewer chat. No new files were deployed or GPU smoke started.

The smoke entry checks exact review/draft/source binding and the native
wrapper parent PID, hostname, physical GPU and absolute argv before updates.
It requires finite gradients, unchanged frozen no_match, changed matching
weights, strict save/reload equality and terminal native COMPLETE/exit0.
The terminal verifier binds all artifact and initialization hashes.
Full training and online integration remain explicitly disallowed by this
smoke, even when it succeeds.

## Full Training

New full-only sources, outside the150smoke source map:

- `scripts/train_e0_relative.py`
- `tools/relative_e0_full_contract.py`
- `tools/prepare_relative_e0_full.py`
- `tests/test_relative_e0_full.py`

The disabled154source full proposal is
`e0_relative_full_training_r1_20261102/draft.json`, SHA256
`499d301bab553c0b9384bb86b9af8c214292247fe29a71d057e0e267de84ec46`.
It fixes12epochs/batch512/LR0.0003/WD0.0001/seed2027/GPU0. The only late
bindings are the source review hash, complete collection acceptance hash,
actual smoke plan/receipt/native hashes and launch_allowed. Recipe, task,
source map and all paths remain immutable. Missing hashes are not launch
permission. The full source review passed and its exact reviewed files
are deployed; collection acceptance remains pending.

Admission requires the canonical full882sequence collection verifier to
reproduce exactly the saved full acceptance, plus a terminal verified real
update smoke. Only706fit sequences receive updates;176component-held TRAIN
sequences retain the fixed2,899reliable/800lost populations. Qualification
requires zero reliable crossings belowIoU0.5, nonnegative reliable
sequence-macro IoU change and more than179lost recoveries. Rank qualifying
epochs by overall sequence-macro change, recovery count and earliest epoch.
No qualifying epoch means retain protected E0 and create no best checkpoint.
Preserve latest, lastfive and bestqualifying. No TEST-derived selection.

## Verification And Remaining Work

Local17CPUtests PASS across relative head, data, training engine, smoke and
full contracts. These exercise actual small CPU optimizer updates, known
versus unknown loss targets, uncovered-state exclusion, independent metrics,
prefix joins, exact native bindings, wrong-stage review rejection and
fixed-recipe/late-binding negatives. They are not real GPU smoke, full
training or benchmark evidence.

Next: wait for all8collectors COMPLETE/exit0 and save the canonical full
acceptance. Read workspace `work/bind_launch_relative_full.py`, bind the
actual collection and smoke evidence, check capacity and canonical admission,
then start the single fixed full training task on26GPU0.
Do not modify the139active collector source files or restart live work.

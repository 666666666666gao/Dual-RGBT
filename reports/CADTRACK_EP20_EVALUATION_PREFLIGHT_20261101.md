# CADTrack ep0020 evaluation preflight

## Observed scope

The live host25 task is `train_cadtrack_ep20_lasher_r7_20261101`.
Its trainer child PID779800 was observed alive on GPU0 at
2026-11-01 03:39:38 UTC, using19098 MiB. The latest log then showed
epoch2 update1200/2500. No benchmark result is available from this run.

The training entry point `lib/train/train_script.py` constructs
`build_cadtrack(cfg)`, not `dual_rgbt.model.DualRGBT`. Therefore this run
fine-tunes the CADTrack trunk/head initialization used by the project.
It does not by itself train the E0 preview, quality, gain or token-cost
components. Do not describe its future direct CADTrack tracking scores as
the complete protected E0 system's scores.

## Evaluation incompatibilities

- `scripts/run_benchmark_tracking.py` constructs `DualRGBT` for variant E0
  and validates historical policy-training receipts and state fields.
  A CADTrack trainer checkpoint cannot be passed directly to that route.
  `DualRGBT` uses `decode.*` and additional preview/quality/gain state;
  CADTrack uses `decode_fuse_search.*`. A deliberate, validated transfer
  into an identified E0 parent is needed for a genuine E0 comparison.
- `experiments/cadtrack/cadtrack_lasher_ft_ep20_r5.yaml` explicitly sets
  `DATA.SEARCH.SIZE=256` but has no TEST section.
  `lib/config/cadtrack/config.py` defaults `TEST.SEARCH_SIZE=320` and
  `TEST.SEARCH_FACTOR=5.0`, whereas training uses factor4.0.
  `lib/test/parameter/cadtrack.py` reads those TEST values and
  `lib/test/tracker/cadtrack.py` strictly reloads the checkpoint.
  A separate evaluation configuration must explicitly match the trained
  architecture and declare inference settings before launch. Do not alter
  the source/config bytes bound to the running trainer.
- `tools/profile_cadtrack_baseline.py` targets an older bound plan and
  deliberately pauses two historical GPU0 trainer groups. It is unsuitable
  for this run and must not be invoked against the current training task.

## Required follow-through

### Host25 CPU transfer audit

Executed `tools/audit_cadtrack_e0_transfer.py` over SSH stdin from the
project root using its existing Python environment. The initial invocation
outside the project root failed to unpickle the CADTrack training settings
with `ModuleNotFoundError: lib`; rerunning from the documented project root
succeeded. No GPU or weight file was created.

- Source ep0020 SHA256:
  `d20c8641e4263df3ceadf9e430d0ac96c181c13ad482bca28c2dddef592a8bb8`.
- E0 parent `runs/stage_c_all882_20261013/latest.pth` SHA256:
  `ee60704e7fe1e281584430332dea7807a335bd4d723a769e4783217b0ac6f56c`.
- Result `TENSOR_COMPATIBLE`:250 mapped tensors, no missing targets,
  shape/dtype mismatches or unexpected preserved tensors.
- Mapping preserves `backbone.*` and `box_head.*`, renames only
  `decode_fuse_search.*` to `decode.*`. Fifteen E0 states remain from the
  parent: six preview tensors, four quality tensors, four gain tensors,
  and the `token_cost` buffer.
- This proves tensor compatibility only. It is not strict model reload,
  a real forward/update, policy calibration, or benchmark evidence.
- Existing `scripts/train_executor.py` is not an admissible unchanged launch:
  it automatically evaluates dev every epoch and freezes policy heads.
  A separate train-only finetuning entry must make the initialized parent,
  updated components, scheduler and retention explicit.

### E0 executor finetuning implementation

Added `scripts/finetune_e0_ep20.py` as a separate entry. It preserves the
parent policy and cost buffer and updates the backbone, decode and box head
on the existing ten E0 execution routes. This is E0 executor adaptation,
not joint policy-head training or a claim that the retained policy remains
calibrated after changing its executor.

Host25 Python3.7 syntax compilation passed. CPU initialization strictly loaded
all265 E0 states after the250-tensor transfer;15 parent states were retained.
A real batch from the882-sequence TRAIN manifest decoded successfully:
templates `[8,2,6,128,128]`, search `[8,6,256,256]`, targets `[8,4]`.
No GPU forward, optimizer update or formal E0 run has occurred for this entry.
Independent supplemental engineering review was requested in the already
authorized review chat, with output
`reports/E0_EXECUTOR_EP20_SOURCE_REVIEW_20261101.json`.

Proposed smoke uses host25 GPU1, batch8,80 samples, one epoch and `--smoke`,
covering ten optimizer updates/routes. It requires finite loss and gradient,
changed backbone/decode/head tensors, unchanged policy state, and strict
checkpoint reload. Formal training remains pending review and successful smoke.

### Admission and launch completed

The specific supplemental engineering review is now SOURCE_PASS/0,
SHA256 `afcfa298e24723caf1cdb6a12c4df2998683670214f53899c00e00d81cb0c88b`.
Its exact source map, actual input hashes and host25 native wrapper hash were
checked before smoke and checked again before formal launch. This is
provisional engineering review, not mandated-model scientific acceptance.

Smoke `smoke_e0_executor_ep20_r1_20261101` completed with native exit0:
ten updates covering ten routes, finite losses1.570-3.319, cumulative changed
tensors backbone151/decode4/head54, preserved policy state, strict reload.
Wall time45.4s including initialization and reload; the smoke epoch took19.6s.
Checkpoint SHA256:
`52d43c396900c21d5c62a4bd938a4528710fe3ee10101bec4337639d65912603`.
No nonzero-gradient claim for each module on each route or output-equality
claim is made by these cumulative checks.

Formal task `train_e0_executor_ep20_r1_20261101` started on host25 GPU1:
wrapper897269, trainer897282, output
`/data2/gaob/Dual-RGBT/e0_executor_ep20_lasher_r1_20261101`.
Native RUNNING and a real optimizer-step log were verified; GPU allocation
was9044 MiB at the initial observation. Recipe40 epochs,20000 samples/epoch,
batch8, seed2027, LR1e-5/backbone1e-6, milestones10/30, gamma0.1, final five
epoch files. Preview/quality/gain remain frozen. Native launch inputs and
smoke evidence are bound in
`reports/E0_EXECUTOR_EP20_FORMAL_LAUNCH_R1_20261101.json`.
The older CADTrack-only task on GPU0 is preserved and distinct.

Preserve the current live trainer. Verify native COMPLETE/exit0 and retained
epochs36-40 before scoring those outputs. Prepare a separate evaluation
configuration and validate checkpoint loading plus real TRAIN-frame inference.
Keep direct CADTrack initialization results distinct from an E0 transfer
experiment. Any E0 transfer must identify the parent policy, parameter mapping,
preserved policy state, and its own admission evidence.

Run complete LasHeR, RGBT210 and RGBT234 evaluations with exact sequence/frame
coverage. Report these as exploratory test results, including FPS scope;
do not claim independent final validation. Measure parameters and MACs on the
actual evaluated model and input sizes. Current training loss does not establish
convergence or any benchmark target.

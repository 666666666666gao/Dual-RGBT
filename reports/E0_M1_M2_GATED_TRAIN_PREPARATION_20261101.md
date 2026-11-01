# M1-Gated M2 TRAIN Candidate

## Implementation

New dual_rgbt/e0_m1_m2_online.py leaves the R15 source bytes untouched.
It inherits initialization, M2 weight loading and sampled-sequence setup
from E0M2Tracker, then wraps its direct E0 base in the existing
M1TemplateGuard. Each frame executes that guard once. Only predicted
lost (class2) at the unchanged sampled frame invokes candidate generation.
Reliable/uncertain/off-cadence frames preserve the exact guard output.
No-match preserves E0. Positive selection updates box/motion history using
the existing R15 history convention and does not write recovery templates.
close restores the observer hook.

This intentionally retains the unresolved mixed-history semantics:
confidence/modality/action refer to the E0 decision, movement/scale to a
replacement. It is a separate gated ablation, not a claimed solution to
all identity or preservation-ranking issues. No M3 is implemented.

Five local CPU tests passed with E:/anaconda/python.exe, torch1.13.1+cpu.
They cover cadence/state gating, exact non-intervention output, close
delegation, sampled-lost no-match/positive branches including history,
counter and template effects with stub candidates/head, and initialization
record reset using real E0Observation.begin. No real model forward, GPU,
full observer lifecycle or real multi-sequence reset is validated by these
tests. Syntax compilation also passed.

Source SHA256 3763fb415dc314c4996d3a0dee8b55e492183b0e7d29b377d46afe20f644b1a4.
Test SHA256 ee51c9877de824bec39b39b09ee38bd1928faa39282e7e1bce548137081f9481.
R1 independent review found a real stale-sequence diagnostic defect:
inherited initialize left M1/observer records from the previous sequence.
The class now clears guard.record and calls observer.begin(False) before
parent initialization. R1 SOURCE_FAIL is preserved; exact-hash R2 review
requested in the human-authorized existing review chat. No deployment,
smoke, TRAIN replay or full TEST launch yet.

R2 subsequently returned SOURCE_PASS/0 for those exact class/test hashes,
report SHA256
745857aa0d3df134d82ec97fe9c434573a776a9c5afff874a39388235006af58.
It covers engineering behavior only, not runner or deployment admission.

## Separate TRAIN Witness Draft

New scripts/check_e0_m1_m2_train.py binds TRAIN-only three-sequence/31-frame
execution, E0/M1/M2 input hashes, native PID/argv/GPU/hostname, exact source
map and independently reviewed nonlaunchable draft. It checks one E0
observation per frame, sequence reset, actual gate/call agreement, unchanged
weights/no gradients, hook restoration, all-frame synchronized timing and
artifact hashes. A zero-call run remains an execution witness only; it
cannot be promoted to recovery-path acceptance.

New tools/prepare_e0_m1_m2_train_smoke.py produced
reports/e0_m1_m2_train_smoke_r1_20261101/draft.json SHA256
950913364a64dc3b85f4e8c0753d957a89e67ea9adc8656cf62733830490097c,
launch_allowed=false. Four additional CPU admission tests pass, including
rejection of TEST split, unreviewed plan mutation and wrong M1 identity.

Host26 read-only preflight verified138 existing sources and all5 inputs;
zero mismatches. Five missing sources are exactly the new class, runner,
two test files and preparer. New task/run paths do not exist. Three local
dirty inherited files differ from actual26; exact matching remote snapshots
are under reports/source_snapshots/m1_m2_train_witness_r1_p26_20261101.
These snapshots do not replace local dirty or live remote source files.

Fresh runner+draft engineering review requested, stage
E0_M1_M2_train_witness. No new files deployed or GPU launched yet. After
exact review passes, deploy only new files and review/draft/plan; verify
remote CPU tests/canonical admission before native smoke on26GPU1.
Existing GPU authorization already covers that TRAIN execution.

## Required Next Execution

A separate TRAIN entrypoint/plan must pin the E0/M1/M2 checkpoints,
source maps, TRAIN sequence membership, absolute interpreter/argv and
new task/run IDs on host26 GPU0/1. Existing R15 evaluator is not an
admission route for this new class. A source review of this class alone
cannot admit a new runner or promote it to full TEST.

First validate observer ownership, initialization across two sequences,
one E0 predict per frame, positive/no-match execution, unchanged model
weights and actual candidate-call counts. Real E0/M1/M2 forwards, complete
frame/timing records and precise native bindings are required for smoke.

Then compare the frozen TRAIN E0/M1 path with gated decisions. At the
actual M2 candidate-decision frames, record both predicted M1 state and
GT-defined E0 overlap strata, candidate availability, selected-box overlap,
reliable-to-worse replacements and lost-to-recovered replacements. GT is
diagnostic only, never an input to the executable gate. M1's original
collection cadence differs from the M2 cadence; measure this exposure
change rather than joining old cache rows by position.

Use original accepted caches if available; host26 currently has none of
their882 NPZ files. A fixed-state shadow replay cannot prove closed-loop
improvement after state divergence. Establish TRAIN protection/recovery
criteria before a separately reviewed closed-loop comparison. Do not
select thresholds, checkpoints or seeds using completed TEST scores.

## Current Protected Run

2026-11-01 22:56:54 UTC: LasHeR R15 child3795734 remains live on26GPU0.
Completed98/245sequences,83437frames; long sequence boyshead9684 trace
frame7470,4.5seconds old. Keep source/plan unchanged and do not restart
because completed-sequence count pauses during this long sequence.
Host25 is temporarily unused; only26GPU0/1 are authorized.

Source diagnosis:
reports/E0_M2_R15_FAILURE_SOURCE_DIAGNOSIS_20261101.md SHA256
72933f763f189770bbfdaf26f880a0ec16315145b588c3388ca75d0a932d9518.
It is same-family provisional engineering diagnosis, actual_model=UNATTESTED,
not a launch review or scientific acceptance.

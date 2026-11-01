# Fixed M1-M2 TRAIN Diagnostic

## Predeclared Scope

Only host26 GPU1 is assigned. Host25 is temporarily unused; GPU0 keeps
the protected LasHeR R15 evaluation. No task is assigned to GPU2/3.

The first four fit and first four component-holdout sequences in the
accepted M2 collection plan order are fixed before this execution:
boysumbrella441, boyshorts636, blackcarturn183,
leftclosedexersicebook1034, whiteboyback175, gonemoto_ab144,
stronggirl338, ab_motoinrain125. Total3076frames,3068trackedframes.
This is a small diagnostic, not full TRAIN coverage. Upstream E0 has
seen these TRAIN sequences, including the component holdout.

The completed93frame witness had90reliable predictions and zero M2 calls.
The longer diagnostic retains the existing lost-class gate and25valid-frame
sampling cadence. It does not force calls, search thresholds/seeds, train
weights or admit TEST execution.

Draft reports/e0_m1_m2_train_diagnostic_r1_20261101/draft.json SHA256
688433d860fdb9efaff079051bde8ace80f819707676ff27db79ef6acfa4888a
binds146sources and9inputs. Executor read-only preflight found143existing
sources exact, only the3new files absent, and all9inputs exact. All
3076paired images exist. Two geometry/IoU CPU tests and nine inherited
class/witness tests pass. A fresh separate engineering review is pending.

## Analysis Defined Before Execution

Report fit and component holdout separately, with per-sequence counts.
Use existing M1 label definitions for valid annotated frames: reliable
IoU>=0.5, uncertain0.1<=IoU<0.5, lostIoU<0.1. Count the confusion matrix
of predicted M1 state against the reconstructed original E0 proposal,
on all valid tracked frames and on actual candidate-decision cadence.

Report M2 invocation, no-match, positive replacement and blocked-template
counts. For replacements, report output-minus-original IoU, counts better,
equal and worse; original-reliable to output-below0.5; original-lost to
output-at-least0.5. Report valid IoU0.5 persistence for subsequent1,5,10
frames only where those frames exist and are valid. Future annotations
are read only by analysis after tracker execution.

Original E0 proposals share the history already changed by previous M1/M2
decisions. They are local per-frame proposals, not an independently run
matched E0 baseline. Aggregate IoU and persistence do not establish a
causal benchmark gain. Zero M2 calls remains a valid diagnostic result
and cannot count as exercising recovery.

After completion verify native COMPLETE/exit0, exact plan/review/source/
input binding, immutable model weights, zero backward/optimizer updates,
all8traces,3076timing samples, and the complete artifact map. Preserve
failed evidence if any check fails. No automatic TEST promotion follows.

## Deployment

2026-11-01 23:43 UTC: independent engineering review SOURCE_PASS/0,
report SHA256
8f45f27f121e4091f87311d6d9f027c3b4df20c74421ef7aac4e521573da80de.
Attribution UNATTESTED/same-family/provisional; not scientific acceptance.
Final plan SHA256
b4bcae6ce38bbe2e645aea21a886a1d16080832fb25cf3fb5854e8819856c393
differs from the reviewed draft only in launch_allowed=true.

Only three new source files and review/draft/plan were deployed.
Remote two CPU tests and canonical146source/9input admission passed.
Native task diagnose_e0_m1_m2_fixed_r1_p26_20261101 is RUNNING on GPU1,
wrapper496915 and actual child496957. Initial model-load status is not
completion or evidence of recovery. Expected duration approximately
5-10minutes, with actual gated candidate calls affecting runtime.

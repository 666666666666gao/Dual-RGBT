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

## Completed Evidence

Native COMPLETE/exit0 after360.920282seconds. All8sequences and3076frames
completed, including3068tracked/3019valid annotated tracked frames.
The separate summarizer verified the complete11artifact tree, native,
receipt/config/plan/draft/review,146sources/9inputs and all timing/trace
coverage. Two summary tests check confusion orientation, label boundaries,
invalid-frame handling and persistence eligibility.

Receipt SHA256
39a76aff03b0c6f24e049368eb1cc05d95e94e078bc26bf86f19a8dfb5d09127.
Native SHA256
ffc32575d0a09d35137ff329f309eef3136a511477123239094c6db0216b9aff.
Raw artifacts and summary are archived in
reports/e0_m1_m2_train_diagnostic_r1_20261101.

| Partition | Valid tracked | Actual lost | Lost predicted lost | M2 calls | Replacement |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fit | 2290 | 200 | 53 | 4 | 3 |
| Component holdout | 729 | 95 | 21 | 1 | 0 |
| Total | 3019 | 295 | 74 | 5 | 3 |

On component holdout,74/95lost proposals were predicted reliable, while
21/95were recognized lost. At the actual candidate cadence, only1/5lost
holdout proposals were recognized lost;4/5were predicted reliable.
These are small fixed diagnostic samples, not population estimates or
evidence that cadence alone causes the error.

Three positive replacements include two improvements and one deterioration:

- boysumbrella26: IoU0.655129 to0.654724, delta-0.000405.
- leftclosedexersicebook376:0 to0.565935, then frame3770.469811,
  frame3780.457385, frame3790.576551.
- leftclosedexersicebook876:0 to0.267773, an improvement still below0.5.

One event crossed lost-to-IoU0.5, but no uninterrupted1/5/10frame
persistence window passed. Falling below0.5 at377 is uncertain, not
lost, and379recovers above0.5; do not claim permanent failure.
Two M2 calls selected no-match.411template writes were proposed and49
blocked. No reliable-to-below0.5replacement occurred in the3events.
This sparse observation cannot establish a protection guarantee.

103608204combined parameters and9.253248FPS over332.423825 synchronized
tracker/decode/init seconds under concurrent load. This small TRAIN
diagnostic timing is not full TEST FPS. No complete route-weighted MAC
result is inferred from traces that omit the requisite route counters.
Zero optimizer/backward calls, immutable model weights, no TEST admission.

## Next Training Candidate

Direct inventory confirms all8original M1 cache shards remain on26:
882feature/label files,71843fit/19248holdout states. M2 candidate NPZ
caches being absent does not mean M1 caches are absent. No recollection
is needed for a same-feature M1 loss experiment.

Original M1 fitting over12epochs improves fit lost recall from0.559861
to0.890950 while holdout lost recall falls from0.416127 to0.343206.
Best holdout macro-recall checkpoint is epoch5, lost recall0.423843.
This is direct evidence of a fit/holdout gap, not proof that class
imbalance alone explains it. Fit counts are48570reliable/5749uncertain/
17524lost. The next fixed candidate changes only cross-entropy class
weights to N/(3*n_class) computed from fit, retaining architecture,
seed2027, AdamW/LR0.001/batch4096/12epochs and holdout macro-recall
selection. It is an optimization experiment, not a novel module.

Historical source check found one real mismatch: the accepted collection
binds run_task_four_gpu.py at cc2b87b3..., whereas the current wrapper
adds OpenCV thread propagation and hostname. Exact original CRLF bytes
were recovered from Git8103296 into a separate pinned snapshot; current
wrapper is preserved. The loader now accepts an explicit hash-checked
historical source mapping for this verified provenance issue. It still
verifies every shard/native/receipt/artifact/label/partition. A separate
new source/draft review and remote CPU/canonical checks are required
before actual GPU1 training; diagnostic completion is not its admission.

## Open-Source Mechanism Reference

Official repository: https://github.com/visionml/pytracking .
Source inspected through the official raw endpoint:
pytracking/tracker/keep_track/keep_track.py, master as retrieved
2026-11-02 UTC. This is a mechanism reference, not a reproduced baseline.

Its localize_target_by_candidate_matching checks the previous frame
candidate set, matches previous/current descriptors plus coordinates/
scores, updates CandidateCollection with associations, and saves the
current candidate set for the next frame (retrieved lines268-321).
extract_matches supplies both frames' descriptors/coordinates/scores
to the matching network (461-474). Thus it provides a concrete reference
for temporal candidate association, which the current independent M2
anchor argmax lacks.

This observation supports investigating persistence/identity association
after the M1 recognition issue is measured; it does not establish that
adding this mechanism improves our benchmarks or constitutes novelty.
No source code from that repository is copied into the new M1 loss
experiment. Any future temporal-module design needs actual TRAIN
supervision, closed-loop comparison, and separate review/admission.

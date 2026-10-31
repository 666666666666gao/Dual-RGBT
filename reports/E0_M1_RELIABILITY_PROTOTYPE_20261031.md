# Protected-E0 Reliability Prototype

## Implemented Scope

`dual_rgbt/e0_reliability.py` is an independent offline localization-reliability
prototype. It is not imported by the E0 model, tracker, observer or collector.
The original E0 path and existing collection admission draft are unchanged.

The head follows the existing small MLP style: 914 observation features ->
128 hidden units with GELU -> three logits, 117507 parameters. Forward takes
only features, not GT or a mutable tracker object. Targets and cross-entropy
are separate training functions:

- reliable: final IoU >= 0.5;
- uncertain localization: 0.1 <= final IoU < 0.5;
- lost: final IoU < 0.1.

These targets describe overlap, NOT verified target identity, visibility,
recoverability or calibrated uncertainty. The current features contain
candidate-local primary deep ROI evidence and cheap initial-template preview
descriptors, not a newly trained deep identity anchor. This head does not
establish the full proposed M1 contribution and does not implement M2/M3.
There is no online trigger, threshold deployment, recovery action or template
commit modification. No real-data training has run.

## Actual Input Coverage Audit

Ran `tools/audit_e0_reliability_prefix.py` on canonical server26 with CPU Python.
Reverified protected checkpoint, native COMPLETE/exit0 command, full bound plan,
122 source hashes, 189 input hashes and seven supporting artifact bindings.
The actual finite float32 matrix has shape 90x914. All three source sequences
are fit sequences: 2up, carturn, motocross; no component-holdout observations.

| Category | Existing Prefix States |
|---|---:|
| reliable | 89 |
| uncertain localization | 1 |
| lost | 0 |

Executed E0 actions are STOP34, PRIMARY2, SECONDARY13, PREVIEW9, EXPAND32.
The prefixes exercise correction routes, but contain NO lost-class examples.
They cannot validate failure detection, calibrate a useful recovery trigger,
or replace complete paired TRAIN collection. Do not train on these prefixes
and claim that M1 works. No component-holdout performance exists.

Final report: `reports/E0_M1_PREFIX_INPUT_AUDIT_R1_20261031.json`,
SHA256 `1d232a46cfff9103f3af3716321cf6a23391343f7142fa135b0c18aecdfb620c`.
R1 reports the draft's explicit admission flag rather than claiming a
global process census from a draft. Earlier remote audit output is preserved.

## Engineering Checks

Five new synthetic CPU tests cover exact overlap boundaries, invalid labels,
features-only/read-only forward, 117507 parameter count, synthetic head-only
updates and strict state reload. All ten E0 observer/scope/reliability tests
pass locally and on canonical server26 Python3.7, CUDA_VISIBLE_DEVICES empty.
These are engineering checks, not learned failure recall or benchmark metrics.
The synthetic frozen linear module in the isolation test is NOT the real E0;
actual E0 source identity is instead bound by the real input audit.

Head SHA256:
`e179851ee5c35570f86e2d4712463fe4c64048f1847986a9472cc6ec8840ca21`.
Audit source SHA256:
`e9631949c69c5e202bfbfde1d9564e348a1d8d3a95583bb8240441fac27d9e39`.

## Outstanding Admission

Full collection remains blocked on a fresh collector source review, not on
permission to use the GPUs. No authorized review-chat message exists; no
message was sent. Existing observer review cannot cover the later collector
or this reliability prototype. Formal scientific review remains unavailable.
No independent-review or training-readiness PASS is claimed.

After legitimate collector review/admission, use all eight disjoint TRAIN
shards, preserve 706fit/176component holdout and verify complete sequence/state
coverage. Audit actual lost/high-confidence cases before class balancing.
Initial learning must freeze E0; keep feature-only inference and TRAIN-only
targets separate. Compare to final response confidence on the SAME E0
observations. Fit thresholds on fit only; report held lost recall and good
interruption, not just classification loss. Upstream E0 saw component holdout.
No new dev, test-label fitting, gain-only grid or blanket template freeze.

## Runtime At 02:31 UTC

All17 existing inference children remain alive, trace ages <=7.8 seconds;
eight GPUs remain assigned. E12=193/245seq,189647 observed frames; matched
parent=65seq,58301 observed frames; all882E8=108seq,107180 observed frames.
No new full score or unexpected exit. E12 roughly40minutes remaining from
a349second interval, buffered concurrent estimate, not formal FPS.
No speculative weight deletion. Targets remain unmet.

# Related Work: Risk-Aware Action and Memory Admission

## Project evidence motivating this review

The fixed P26 TRAIN context-gate audit (R4) found 1,305 reliable-to-below-
half IoU crossings and 275 lost-to-at-least-half recoveries for the raw M2
action. The M1 gate reduced these to 231 harmful crossings and 114
recoveries. It failed all predeclared gates: zero harmful crossings,
nonnegative reliable sequence-macro IoU delta, and more than 179 recoveries.
The gate is not eligible for online use.

Separately, the P25 context-recovery TRAIN experiment increased held
recoveries from 52/800 to 275/800 but raised reliable harmful changes from
12/2,899 to 1,305/2,899. No epoch qualified. This indicates that optimizing
recovery frequency alone is not enough; the action's risk to already reliable
tracks must be constrained. These are TRAIN diagnostics, not benchmark
metrics or independent validation.

## Public implementations examined

- [UncTrack](https://github.com/ManOfStory/UncTrack) is the official TIP
  2025 implementation of an uncertainty-aware prototype memory tracker. Its
  directly relevant idea is to make memory updates depend on confidence,
  rather than treating every predicted state as equally trustworthy.
- [SENTRY](https://github.com/HamadYA/SENTRY) is an ECCV 2026 public
  implementation for SAM2-based tracking. It preserves the host prediction
  by default and admits severe rescues only when short tracklets pass
  temporal and neighbor/distractor consistency checks.
- [DAM4SAM](https://github.com/jovanavidenovic/DAM4SAM) provides a
  distractor-aware memory and memory-management strategy for SAM2 trackers.
  It is a useful reference for write/update policy, but its segmentation
  memory and benchmarks differ from this paired RGB-T box-tracking system.
- [Selective Prediction Toolkit](https://github.com/ercedut/selective-prediction-toolkit)
  reports risk-coverage curves and calibrates acceptance thresholds on a
  calibration split. This supports measuring both harm among accepted
  actions and coverage/recovery, rather than optimizing either alone.
- [Google Uncertainty Baselines](https://github.com/google/uncertainty-baselines)
  is a broad uncertainty/robustness benchmark collection; it is useful as a
  protocol reference, not as a tracker dependency.
- [Conformal Risk Control](https://github.com/aangelopoulos/conformal-risk)
  includes examples controlling monotone risk functions. Its calibration
  guarantees cannot be assumed to transfer directly to temporally dependent
  tracking frames; sequence-level calibration and a fit-for-purpose risk
  definition would be required.

## Transfer hypothesis for Dual-RGBT

Test a conservative, state-conditioned admission rule for M2 proposals:
keep the existing state by default; admit an identity/recovery proposal only
when (1) M1 estimates the current state is at risk, (2) the proposed identity
is temporally consistent over a short history and sufficiently distinct from
nearby distractors, and (3) its predicted benefit exceeds a calibrated
intervention-risk margin. Treat these as separate measurable conditions so
the audit can identify whether harm comes from failure detection, identity
ranking, or temporal verification.

First test this as a TRAIN-only diagnostic with a sequence-level fit /
calibration / held partition and fixed acceptance gates. Report harmful
crossings, recoveries, reliable-state delta, coverage, risk-coverage curves,
and per-sequence macro results. The existing M1/M2 holdout has already
participated in development and must not be relabeled independent. Do not
use benchmark TEST to set thresholds, select checkpoints, or tune the rule.

## Scope and status

This is a literature-derived hypothesis, not an implementation, proven
novelty claim, or approved GPU recipe. The cited repositories address
related but non-identical tasks; their reported results do not predict
Dual-RGBT performance. P1A DropMoE/SOT TRAIN remains in progress. No code,
model architecture, or experiment admission was changed by this review.

# Measured Harm And Closed-Loop Data Coverage

## Current Evidence

The fixed original TRAIN replay found326/2,899reliable states falling below
IoU0.5 after M2 replacement, alongside179/800lost recoveries. This establishes
a harmful replacement problem on baseline E0 states. It does not establish
which error sources dominate after multiple online interventions. See
`E0_M2_RESTORATION_AND_STATE_REPLAY_20261102.md` for exact populations.

The new explicit-current-E0 alternative addresses this measured baseline
problem: train KEEP when no known observed candidate improves current E0,
and enforce reliable-state protection before considering a checkpoint.
The new full supplement still runs the protected original E0 trajectory.
It does not collect states induced by an accepted M2 intervention.

## Primary Open-Source Reference

Ross, Gordon and Bagnell, AISTATS2011, "A Reduction of Imitation Learning
and Structured Prediction to No-Regret Online Learning":
https://proceedings.mlr.press/v15/ross11a.html

The paper treats sequential prediction in which earlier decisions affect
later observations. Its DAgger procedure gathers supervision on states
visited under the learner's induced distribution rather than relying only
on an expert's fixed trajectories. This is a reason to examine closed-loop
coverage after interventions; it is not evidence that distribution shift
already explains this tracker's measured regression.

Maintained implementation and official documentation inspected2026-11-02:

- Repository: https://github.com/HumanCompatibleAI/imitation
- Source: https://github.com/HumanCompatibleAI/imitation/blob/master/src/imitation/algorithms/dagger.py
- Documentation: https://imitation.readthedocs.io/en/latest/algorithms/dagger.html

The implementation includes InteractiveTrajectoryCollector/DAggerTrainer
and supports collecting trajectories under a mixture of learner/expert
actions with expert labels. This tracker does not use that RL environment
API and has no expert online at TEST. No dependency or code was imported.
This reference informs data collection and measurement, not a claim that
our relative scoring architecture is DAgger or a novel imitation algorithm.

## Conditional Next Measurement

Keep the current fixed relative TRAIN recipe unchanged. If a checkpoint
passes its predeclared protection/recovery gates, the next separately
reviewed M1-gated paired TRAIN integration should record the first
intervention and subsequent actual tracker states. Replay labels alone
cannot prove recovery after the crop/template/search state changes.

Use only TRAIN annotations for offline supervision and diagnostics. Record
baseline and intervention trajectories separately, with frame keys,
candidate coverage, decisions, first divergence, later loss/recovery,
parameter hashes and actual encoder/MAC counts. Annotation must not enter
the runtime decision. Compare reliable harm and recovery on these actual
states before deciding whether additional learner-trajectory TRAIN data
is needed.

No new collection, threshold, epoch, seed or TEST-based choice is authorized
by this note. Existing reviews, paired parity checks and full acceptance
still govern execution. A future trajectory aggregation experiment would
need its own fixed TRAIN plan and source review. Three-module innovation
and benchmark gains remain unproven until implemented and measured.

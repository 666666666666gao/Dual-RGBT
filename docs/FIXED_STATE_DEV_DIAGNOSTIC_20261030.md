# Fixed-state dev diagnosis

This experiment diagnoses the frozen parent and E6 overlay at evaluation cost
0.15 and K=8. It does not admit another training run. E6 is the best completed
cached-gain checkpoint by full-dev SR among E6-E9; E10 remains in evaluation.

Use `midboyblue`, `belowrightwhiteboy`, and `moto2north1` as repeated regressions,
with `browncar2north` as a positive control. These are deliberately selected
diagnostic sequences, not an unbiased sample of all 97 dev sequences. Within
each policy's own visited trajectory, execute all candidate requests every
fifth frame, using the saved pre-track templates and primary features. Also
execute the other modality's primary path at that same state. Different policies
can have different histories at the same sequence frame.

The observer returns the original preview/gain outputs without changes. Its
crop hook records the actual dynamic-template crop call and verifies the
post-write template values in the exact channels written by the native route.
The full replay compares six-decimal predictions and integer actions, modality,
request index and candidate count with the existing saved run. A separate
11-frame control compares exact boxes, templates, history and counters with
and without observation. Every counterfactual verifies that the live state,
templates, history and counters did not change. All 265 model state tensors
remain frozen; no backward or optimizer operations occur.

Labels are computed after online decisions and diagnostic candidate execution.
Dev annotations are diagnostic only and must never enter training. Frames with
nonpositive or nonfinite GT sizes are excluded from geometric label counts.
Report sampled recoverable states separately from recovery queries and low-IoU
STOP frames. Observed template writes at low IoU are correlation, not proof of
template-induced drift. One-step oracle gain is not future trajectory accuracy.

## Execution

The existing server25 Python environment and native task wrapper are reused.
The active E10 source, plan and GPU1 process are not edited. The new files are:
`scripts/diagnose_fixed_state_dev.py`, `tools/launch_fixed_state_dev.py`,
`tools/queue_fixed_state_dev.py`, and `tools/analyze_fixed_state_dev.py`.

Parent and E6 `browncar2north` 11-frame smokes completed native exit0.
Each executed 36 candidate queries at two states, recorded two real template
writes, preserved the control trajectory and froze all 265 model state tensors.
These are execution checks, not benchmark improvements.

The eight sequence/policy diagnostics run across server25 GPUs0/2/3 with a
180-second queue cadence. Record actual native completion and artifact hashes
before accepting the CPU summary. Evaluation batch remains one causal state;
adding unused GPU allocation would not improve this diagnostic. Training
batch/LR rules apply when training is justified by these results.

## Other completed controls

All numbers below are official LasHeR dev (97 sequences / 54,646 frames),
PR / NPR / SR, evaluation cost0.15. The size overlays are four-update resource
smokes, not converged models.

| Control | PR | NPR | SR |
|---|---:|---:|---:|
| Frozen parent | 75.967585 | 70.762121 | 56.047161 |
| Size B384 smoke overlay | 73.205319 | 68.258088 | 54.232825 |
| Size B576 smoke overlay | 75.158551 | 70.135285 | 55.712312 |
| Static dynamic-template slots | 68.987885 | 64.691062 | 53.070734 |

None supports promotion or directly disabling dynamic template updates.
Raw results are in `reports/size_tail_dev_20261030/` and
`reports/template_dev_20261030/`. FPS there includes concurrent workloads and
must not be interpreted as an isolated speed comparison.

## Public implementation references

GitHub API and raw source were read on 2026-10-30. No source code is copied.
These are design references to revisit after diagnosis, not validated changes.

- `visionml/pytracking`, commit `7eb9e74bd3d40e29dbcec444902237da13de247b`:
  `pytracking/tracker/keep_track/candidates.py` carries candidate object
  identities across matching updates and handles low-confidence assignments.
  `keep_track.py` distinguishes `not_found`, `uncertain`, and hard-negative
  states and restricts model updates for uncertain/missing targets. Use this
  reference if apparently recoverable queries switch to distractors.
- The same repository's `pytracking/tracker/atom/atom.py` separates target
  localization, `refine_target_box`, and update decisions. It provides a
  concrete reference for separating box refinement from state updates.
  No ATOM or PrDiMP module has been integrated.
- `rajevv/Multi_L2D`, branch AISTATS2023, commit
  `3f2ad6bc93afdfd250faac21225f6b55074b6cf8`: README identifies shared surrogate
  losses in `lib/` and confidence calibration experiments. Its multi-expert
  classification setting does not establish closed-loop tracking calibration.
  A binary confidence cannot replace the current gain in IoU units silently.
- `Zhangyong-Tang/MoETrack`, commit
  `2be4b703bf7d4ed9b7d9394bd91cf9f0ba5428a0`: README explicitly describes
  "when to fuse", jointly trained experts, and an adaptive modality switcher.
  The visible tree distributes implementation as `MoETrack (2).zip`; only the
  README/tree were inspected here, not the archived implementation. Modality
  switching alone is therefore not a new contribution of this project.

The three research directions remain hypotheses: target-conditioned modality
reliability; recovery-versus-harm evidence acquisition; and separating box
quality from state/template write confidence. They are not three validated
innovations until controlled full-dev improvements support them.

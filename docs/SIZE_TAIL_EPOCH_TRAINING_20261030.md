# Resumable epochs and full-dev stopping

This is a candidate execution path, not an admitted formal training result.
Before a formal plan may set `formal_launch_allowed=true`, inspect the full
parent-equivalence control, B384 adapted dev diagnostic, and measured B576
memory/finite-update/reload results. Bind the two completed dev reports by hash
in the plan. Do not promote crop-loss improvement to tracking accuracy.

`scripts/train_size_tail_epoch.py` executes one sampled training epoch per native
task. It changes only the existing final size-layer weight and bias (66
parameters), freezes all other weights and BN statistics, and uses the existing
localization loss. Every epoch gets a complete checkpoint, native exit status,
source/manifest/parent hashes and an exact checkpoint-reload check. Epoch two
onward restores both the previous network and AdamW state. Optimizer step counts
must match the previous receipt and then advance by the actual batch count.
Sampling seeds advance by the number of pairs; RGB/TIR alternation follows the
cumulative update index.

`tools/run_size_tail_epochs.py` provides the following sequence on one selected
server25 GPU:

1. Wait until the existing GPU memory is within the admitted plan's budget.
2. Train one epoch using the accepted neural environment and native wrapper.
3. After native exit0, evaluate the checkpoint on all97 LasHeR-dev sequences /
   54,646 frames with the already-smoked, unchanged size-tail evaluator.
4. Verify and score using the existing CPU environment and official toolkit.
5. Record PR/NPR/SR, concurrent-load FPS, train loss, timing and allocator peaks.
6. Continue for at least five and at most ten epochs. Stop after three epochs
   without SR improving by more than0.1 percentage points over the last progress
   anchor. Training loss does not control this decision.
7. Keep five recent checkpoints during training, removing an older checkpoint
   only after its evaluation is complete. At the end, rank the latest five by
   SR, then PR, then NPR; record the selection and retain only its checkpoint.

Cleanup is confined to the exact epoch checkpoint paths constructed for this new
run and verifies their hashes. The original parent and historical runs are never
cleanup candidates. A failed native task aborts the controller before selection
or final cleanup. Existing completed reports and checkpoints remain available.
The predeclared exception to minimum-five convergence monitoring is severe dev
regression: if SR drops more than5 percentage points below the parent, stop the
candidate immediately with `REJECTED_REGRESSION` and a `REJECTION.json` receipt.
Do not select it or delete its remaining forensic checkpoints. This rule responds
to the observed historical readout-adaptation SR collapse of14.7686 points;
it is not a response to hypothetical failure. The5-point threshold is a declared
experiment budget decision, not a learned threshold or benchmark target.
This controller has no automatic restart/adoption mode: inspect its actual PID
and native statuses before deciding how to recover an interrupted invocation.

The controller records its PID, current phase, active native PID, next check time
and full epoch history in `runs/<name>/status.json`. It polls at180s intervals.
The final `SELECTION.json` contains all evaluated epoch numbers, the five-epoch
selection window, selected metrics and checkpoint hash, stopping reason and the
parent/protected-E1 comparison values. `DEV_SELECTED` remains test-ineligible;
it is not evidence that any requested benchmark target has been reached.

## Validation

`tests/test_size_tail_epoch_controller.py` uses isolated temporary synthetic
native/scoring fixtures. Six tests pass on Linux, covering minimum epoch count,
patience resets, metric priority, last-five window, actual resume arguments,
rolling/final cleanup, parent preservation and refusal to select after training
failure, plus immediate rejection after severe dev regression. These tests have
no scientific metric claim. Windows executes the two
pure decision tests and skips the Linux flock workflow tests.

Real GPU validation separately uses B8/LR1e-6 and32 train pairs per epoch. The
first and second resumable epochs passed four updates each, frozen-branch checks
and strict reload; cumulative optimizer steps were4 then8. Loading the optimizer
state is bit-exact. Against an uninterrupted64-pair/two-epoch control, all265
final network tensors and losses match exactly. Final weight-moment differences
are2.235e-8 for exp_avg and2.328e-10 for exp_avg_sq. Repeating the uninterrupted
control itself also gives moment differences (8.382e-9 and2.328e-10 respectively)
while all265 network tensors still match. These results do not establish exact
final optimizer equivalence and do not identify a particular CUDA kernel.
They are consistent with the reproducibility limits documented by
[PyTorch1.12.1](https://github.com/pytorch/pytorch/blob/v1.12.1/docs/source/notes/randomness.rst):
disabling cuDNN benchmarking alone does not guarantee deterministic kernels.
No numerical workaround or extra compatibility branch was added. This
short numerical control is not formal training and does not admit a long run.
Its plan explicitly records `TWO_EPOCH_UNINTERRUPTED_NUMERICAL_WITNESS`.

The currently running original training/evaluation sources remain unchanged.
The real numerical audit is saved as
`reports/SIZE_TAIL_RESUME_NUMERICAL_AUDIT_20261030.json`.
Formal admission still requires the pending full-dev evidence, not merely the
synthetic workflow tests or successful training probes.

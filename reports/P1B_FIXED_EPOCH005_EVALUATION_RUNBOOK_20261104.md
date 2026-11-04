# P1B Fixed Epoch005 Evaluation

Both original P1B round1 FITs have independently passed the read-only CPU
terminal audit in `work/audit_p1b_completed_for_evaluation.py`.
Raw evidence is archived once in
`reports/p1b_epoch005_evaluation_audit_r1_20261104/{dropmoe,sot}/`.
Each run has native COMPLETE/0, five epochs, 890 updates and 91,091 records.
All five retained checkpoint hashes match. Epoch005 config and executor
identity match; non-policy tensors are unchanged and all three policy
prefixes contain changed tensors. CUDA was not initialized.

Fixed checkpoints:

| Branch | Epoch005 SHA256 |
| --- | --- |
| DropMoE project label | 6f29659e3f03b5d972296059d3c2d2f00ca93cb743e155b7ad5a71a53e63f0f4 |
| SOT | 8e516d27303c6da86d6fe943a767a231f6dc03e1b38b595942587ff3da7829e5 |

The original upstream identity of the DropMoE label remains unverified.
These weights were fixed as the final recipe epoch, not selected on TEST.

`work/prepare_p1b_evaluation.py` produces the disabled R3 drafts, one per
host. Each preserves its own original 49 E0 inference/scoring sources,
adds `scripts/eval_p1b_completed.py` and includes eight original-host
factory pickle dependencies (58 pinned sources). The two original maps differ in
`tools/joint_candidate_results.py` and
`scripts/joint_evaluation_contract.py`; do not copy the P25 map over P26.
R1 drafts were superseded before deployment because R2 adds the P1B
authentication entry. Both R2 smokes failed with exit1 before inference:
factory unpickling required the omitted lib.train package. Preserve their
isolated roots, exact reviews, source snapshots and failed native/log bytes.
Raw failure/dependency evidence:
`reports/p1b_evaluation_dependencies_r3_20261104/{dropmoe,sot}/`.
The actual pickle references Settings, EnvironmentSettings, AverageMeter
and StatValue. R3 copies the eight transitive original package files,
performs an actual CPU factory load after deployment and uses fresh task IDs.

The additive entry authenticates the exact FIT evidence and state before
delegating unchanged E0 inference. It preserves the native argv, sets
cuDNN benchmark false and records precision. The inner evaluator command
must omit `-B`, since Python removes that option from sys.argv while the
native wrapper records it. The outer wrapper may use `-B`; bytecode writes
are disabled through PYTHONDONTWRITEBYTECODE.

After exact separate source/plan review, use
`work/operate_p1b_evaluation.py <branch> deploy`, then `smoke`.
Deployment creates a new isolated root from the matching host's pinned
source bytes and the additive entry. Existing snapshots stay unchanged.
The TRAIN smoke runs 11 frames. The `full` action verifies its native
COMPLETE/0, exact config/checkpoint/source/precision and unchanged-state
receipt before fresh GPU0 admission. It launches only full LasHeR:
245 sequences / 220,703 frames. No RGBT210/234 job is implicitly launched.

Fresh GPU admission requires >=6,000 MiB free, <500 MiB used, no compute
process on its UUID, temperature<85 C and exact power limit (250 W P25,
350 W P26). At least 5 GiB destination disk space is required.
Observe the actual wrapper/child and native log after launch.
Use 180-300 second boundaries for long-running observations.

When full inference completes, authenticate native/source/config/precision,
all sequences, predictions, GT and timings, then invoke the unchanged
`tools/score_e0_finetuned.py` once with the host-matching pinned toolkit
and protocol. Never rerun a scorer if its output already exists.
Runtime config retains the engine's E0_finetuned_evaluation schema; the
new review and pinned FIT evidence identify it as P1B epoch005.
FPS includes decode and initialization under concurrent load; total
parameters come from the actual loaded model. A later compute summary
must use source-matched route MAC evidence. The scoring helper
`work/score_p1b_completed.py` needs its separate source review before use.
Read-only smoke and full-start metadata can be archived once using
`work/archive_p1b_evaluation_witness.py`.

All full TEST results remain exploratory and test-ineligible. Do not use
them to select checkpoints, alter cost/recipe or declare independent final
validation. P1B round1 evaluation is not C2 or a second TRAIN policy round.
The goal remains unmet until actual full metric evidence exists.

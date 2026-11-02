# Fixed Paired TRAIN Runtime Preparation

The preceding eight-sequence diagnostic reports an E0 proposal on a history
already modified by earlier interventions. Its own receipt explicitly says
it is not an independent matched E0 control. This limits interpretation of
later recovery and motivates independent trajectories after relative TRAIN
acceptance, rather than another unconditional TEST run.

## Fixed Scope

The new disabled draft is
`e0_relative_pair_train_r1_20261102/draft.json`, SHA256
`44fcab8eebfb37e8eaea0cdf8d8da435049592e635c403b1d0d935c2825b0e97`.
It binds166source files and the same eight TRAIN sequences/3076frames as
the existing diagnostic, in the same order. Its GPU is26GPU0, seed2027,
cost0. There is no threshold, architecture, cadence, seed or sequence search.
The two arms are independent E0+M1 control and E0+M1+relative-M2.
Both have separate models, M1heads, templates, history and predictor hooks.

New files are `tools/relative_e0_pair_contract.py`,
`scripts/check_e0_relative_pair_train.py`,
`tools/prepare_e0_relative_pair_train.py` and
`tests/test_relative_e0_pair.py`. All154active collection/full-training
source files remain unchanged. The already reviewed runtime core is reused.

Admission reproduces the existing full TRAIN terminal verifier and requires
exact equality with its saved acceptance, a qualifying epoch, accepted
best.pth hash and actual training-plan hash. A missing or rejected checkpoint
cannot run. A separate SOURCE_PASS must bind this complete166source map,
fixed inputs and exact disabled draft. Runtime native/parentPID/host/GPU/
absoluteinterpreter/argv are checked before model execution. Late binding
can change only launch flag and actual review/training/checkpoint evidence.

The entry installs the established six-channel crop helper before Tracker
imports. Each frame runs both independent arms before reading GT for
diagnostics. Initial state and every frame before the first replacement
must match exactly. Traces include both boxes, IoUs, M1states, template
write decisions, histories, query counters, state hashes, actual route/token
counts and current/candidate encoder counts. Timings synchronize the entire
initialization and track call for each arm; shared decode is recorded
separately. GT diagnostics, hashing and model load are excluded.

Frozen parameters are compared exactly after execution, gradients must be
absent, and both predictor hooks must be restored. The receipt cannot grant
benchmark admission; zero interventions cannot establish recovery benefit.
Actual trace acceptance and subsequent analysis still require verification
of real native/artifact evidence. No result is inferred from this draft.

## Current Checks And Boundary

Five local CPU contract tests pass. They exercise forbidden recipe/TEST
changes, malformed or absent evidence bindings, epoch/checkpoint mismatch,
rejected TRAIN outcomes, stale saved acceptance and exact native argv.
Mocks supply terminal-verifier return values in these negative tests;
they are not real full training or a GPU trajectory test.

Independent source review is active in the authorized review chat, expected
report `E0_M2_RELATIVE_PAIR_TRAIN_SOURCE_REVIEW_R1_20261102.json`,
stage`M2_relative_current_E0_paired_train`. No source review outcome has
been claimed. The four new files/draft are local only: no deployment,
launcher, actual plan or GPU execution exists. Missing full TRAIN evidence
remains null and launch_allowed=false.

At07:31:30UTC current supplement collection was741/882sequences and
361828/459435frames. Shard4 was already COMPLETE0; all other seven
children were live with current updates<=5.8seconds. The existing full
TRAIN handoff controller was live. Both LasHeR evaluation children remained
live (epoch40:181sequences/185000frames; epoch39:105/97426).
P25all ten evaluation children and automatic CPUscore controller were live;
all four cards250W, temperatures83/71/70/68C. No new full score or actual
full relative training is established by those observations.

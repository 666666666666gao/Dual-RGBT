# Frozen-parent training query cache

Purpose: obtain the actual955-dimensional pre-query features and executed IoU
outcomes absent from the historical training CSVs. This is a data-collection
pilot, not model training or a benchmark accuracy experiment.

The parent is the native-complete server25quality-k8 train-cost0.1 checkpoint,
SHA21e07cd1f645c6f7237e063c81cf5d60f6a14bca710e0180e8a3ca29dc827073.
Replay cost0.15,K8,seed2027. Model state is frozen. All online decisions,
templates and history use the existing tracker. GT is read for the initial
box and, after each labelled online step and sampling decision, for query labels.
Teacher queries reuse pre-track templates and the actual primary features.
They cannot write online state or influence later online choices.

Plan:reports/FROZEN_QUERY_CACHE_PLAN_20261030.json. Shuffle sorted882train names
with Python random.Random(2027); choose first20,first16fit,last4holdout. No dev
or test sequence is eligible. Full replay would cover9435frames,923valid
every-tenth-frame states. Sample one request uniformly from each of4kinds on
the deployed primary modality, retaining its sampling probability. Include the
deployed query as selected_extra if not already sampled; keep its purpose label
separate so it is not mistaken for a uniform candidate sample. Expected minimum
3692query labels; up to4615if every sampled state needs an extra query.

Observe the existing candidate_gains call and a pre-hook on query_gain. Copy
the input tensor; verify it reproduces the recorded gain vector without GT.
Save per-sequence float32features.npy and rows in labels.csv in identical order.
Labels include query kind, selection, sampling probability, predicted gain,
main/query IoU and confidence. Train/holdout partition is by sequence.
The recorded probability is conditional on query kind and the visited state;
it is not a frame/modality probability or a behavior-policy propensity. The
four holdout sequences were used in the parent's original882-sequence training.
They are reserved only for subsequent fitting/calibration of new components,
not an unseen-data evaluation of the complete parent system.

Before collection, run31frames of the first sampled train sequence with the
original tracker, then repeat with observation and teacher labels. Require exact
box/action/modality/template/history equality throughout those30tracking steps;
check teacher state isolation, frozen265model tensors and absent gradients.
No backward or optimizer is used. This does not establish full-replay equality.
Source hashes, native command, parent evaluation contract, manifest split and
input hashes are checked. Independent source audit must precede GPU smoke.

Full collection remains disabled until smoke/native completion and source
review. A separately registered full plan may enable collection and bind the
smoke receipt, native COMPLETE/exit0 status, matching source/configuration and
all smoke artifacts. training_allowed=false remains. Estimated full collection is
roughly30-60minutes at current shared-card tracking rates plus teacher-query
cost; replace with measured smoke cost before launch. Existing12evaluations
continue; collection has no authority to stop or overwrite them.

Acceptance is a complete, verifiable feature/label cache. Subsequent fitting
requires a separate plan: holdout calibration, comparison with the unchanged
gain head, recovery precision and harm rates, then full97dev closed-loop results.
A20-sequence pilot does not support final benchmark or innovation claims.

## Admission and launch at2026-10-30 01:31UTC

Independent gpt-6-astra/max source review found a missing native/source/plan
binding in the full-mode gate. The revised collector binds native COMPLETE,
exit0, smoke command identity, source maps, original smoke plan and all smoke
artifacts. Follow-up source verdict PASS, same-family/provisional. The reviewer
also checked all20selected sequences/9435frames against original annotations.
Review metadata:reports/FROZEN_QUERY_CACHE_SOURCE_REVIEW_20261030.json.

The actual smoke completed natively with exit0 in36.69seconds overall.
The measured collection/final-verification phase was10.78seconds:31frames,
3labelled states,12teacher queries,265unchanged state tensors,0updates/backward.
Original/observed30-step trajectories, templates and history matched exactly.
CPU audit passed. All3sampled states were lost, with0sampled recoveries and
0deployed queries. Thus the selected_extra GPU branch was not exercised in
this smoke; its assertions remain active during full collection. These counts
describe only a31-frame training prefix, not tracking accuracy or method quality.

Separate full plan:reports/FROZEN_QUERY_CACHE_FULL_PLAN_20261030.json.
Run frozen_query_cache_pilot_20261030 is active on original25GPU3, wrapper1186052,
alongside existing evaluations. No model updates are allowed. ETA45-70minutes
from01:31UTC is a rough projection from the short measured collection phase.
CPU watcher1186412 polls native state every180seconds, then invokes the reviewed
summarizer and writes reports/FROZEN_QUERY_CACHE_PILOT_AUDIT_20261030.json.
Full cache completion and its opportunity counts remain pending.

The source-review JSON was created with CRLF and its exact bytes are bound by
the active full plan. A file-specific -text Git attribute preserves that hash;
active evidence is not rewritten just to normalize line endings.

At01:32:54UTC the first full sequence completed:boysumbrella441frames,
44sampled states,178labels(176sampled+2selected_extra),3selected rows in total.
The selected-query reproduction checks therefore ran on actual deployed queries
in full collection. This is prefix evidence; full cache completion is pending.
The observed first-sequence rate suggests roughly25-40minutes remaining,
subject to different sequence lengths and shared-GPU load.

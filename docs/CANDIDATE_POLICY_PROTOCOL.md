# Single-query candidate-benefit policy

Implementation snapshot2026-10-14: new policy, tracker and sampled B-training driver exist. Independent SOURCE admission, real GPU M0, formal controller fit and new C/dev/test are not complete at this snapshot. The executed-utility result audit is separate. No new parameter, speed or final accuracy improvement has been measured.

## What changes

The old four requests consist of main local, main alternate local, global local and global wide. The new path retains those four request families but makes the region pool variable: two main requests plus a local and wide request at each available dispersed global candidate. Candidate points use the already computed auxiliary broad map, initial-template descriptor, valid image cells and the same spatial suppression as the diagnostic. Real exhausted candidate lists remain shorter. No auxiliary ViT encoding happens while making or scoring the pool.

Reuse the existing scalar955-input benefit MLP and its four family indicators. It receives913 shared context values and42 per-region values. A family id describes local/wide request semantics; it is not the candidate rank. STOP has zero benefit and cost. Choose the maximum predicted benefit minus lambda times normalized token cost, then execute exactly that one raw auxiliary region, or zero queries for STOP. Local input128 uses193 auxiliary tokens; wide256 uses385. These are computation counts, not measured full-system latency.

The separate candidate tracker preserves causal state advancement, selected-modality template writes, auxiliary template writes only after a query, and original margin10 image clipping. Route records include request index, family, candidate rank and real pool count. Existing source17/source30/source34 and the running original pipeline remain untouched.

## Controller fit and its limits

Initialize strictly from the completed fitted-preview checkpoint, using the original executor factory. Freeze all parameters except executor.quality and query_gain: eight state tensors may change and257 must remain identical. Keep the whole model in eval mode so frozen running statistics do not change. AdamW uses learning rate1e-4 and weight decay1e-4, matching the previous controller setting.

Use the existing EvidencePairs distribution: earlier GT templates and GT-jittered current search crops create **offline synthetic tracking states**. Thus the crop geometry has a training GT prior. This is supervised pair fitting, not a causal tracking rollout. Candidate and benefit-head methods have no target argument, and query GT is used for executed IoU labels after proposal/execution. Later C must use actual prior predictions and current history to refresh the state distribution.

For each batch, cycle the four request families. Compute both single-modality primary predictions and sample one valid request per modality from that family; global candidate rank is uniform over its actual available set. Main-search families have one request each. Use the family as the fixed128/256 encoder-size bucket. Train quality against primary IoU and benefit against float32 signed query-IoU minus primary-IoU, using smooth-L1. Do not enumerate every heavy query at training or inference simply to choose an action.

Formal B schedule is five epochs of10000 examples, batch8,6250 updates and100000 labelled query executions, covering all882 training sequences. Save complete private sampled-label records for provenance. Public files may include only aggregate exposure/loss/route counts, hash receipts and source/configuration. GT/boxes/coordinates/label CSV/checkpoints remain private.

## Admission and remaining evaluation

Independent result integrity and fresh SOURCE admission precede NN execution. Real M0 cycles all four families with both modalities, four optimizer updates on32 examples. Verify finite nonzero head gradients, correct Adam steps,257 frozen and8 changed state tensors, and exact strict reload of12 prediction fields. A one-candidate pool must reproduce the old four request geometries and gains bit for bit. Also execute10 causal frames with frame0 GT initialization; independently replay each selected request on the preserved pre-track state and require its box/action/cost to match. That probe performs no further model updates. M0 weights cannot initialize the formal run. Formal admission reads that exact native M0 COMPLETE/exit0, source/plan/input identity and SANITY artifact before model construction.

Formal fit first diagnostic is scheduled after10 minutes to estimate throughput and check loss; later observations should be placed near the estimated completion, with necessary repeats240 seconds. Start long new C only after the original reference pipeline terminates and required GPUs release. Final evidence still needs current closed-loop training,97-dev calibration,245-test PR/NPR/SR, paired complete-system mean/P95 latency and equal-budget core controls. The research target remains78/72/62 with a clear measured cost reduction.

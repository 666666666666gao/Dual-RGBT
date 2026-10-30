# Template memory and failure-state diagnosis

The audited current-grid parent remains quality-k8/train-cost0.1/eval-cost0.15,
with full-dev PR/NPR/SR75.9676/70.7621/56.0472. This diagnosis uses its already
verified predictions, evidence and traces; it does not change any running model.

## Observations

Across54,289 valid noninitial dev frames,13,080 have continuous IoU<0.1. Of those,
6,181 still have confidence>=0.65. The score is the tracker's localization-score
maximum, not a separately calibrated probability of correct target identity.

The legacy traces omit a template-write flag. The matching tracker source hash
89e18853f60eac309e47206005450164931b68b3934ae36a40034cd3b3ce03ed was verified;
its rule is frame%5==0 and confidence>=0.65. Reconstructing that rule gives:

| Reconstructed write event | Count |
|---|---:|
| All valid-GT writes |7,608|
| Current IoU<0.1 |1,215 (15.97%)|
| Current IoU<0.5 |2,163|
| IoU<0.1 after STOP |1,214|
| IoU<0.1 after an executed query |1|

Thus a query-only acceptance/write gate would not cover the dominant bad-write
events. This remains observational evidence: useful writes are also present,
and it does not prove that suppressing updates will improve closed-loop tracking.

Primary IoU<0.1 occurs13,095times:13,019STOP and76queries. Query frequency on
these states is about0.58%, versus about0.65% on the other valid states. Most
frames overall are STOP, so the99.42%unqueried-failure fraction alone is not
evidence that useful queries were missed. Only8executed queries recover from
primary IoU<0.1 to final IoU>=0.5;2reverse that transition. Unexecuted query
benefits remain unknown and need counterfactual execution if investigated.

There are1,315low-IoU episodes,174lasting at least10frames. Examples include
midboyblue frames1930-2699 (770frames,1query,95reconstructed writes),
rightmirrorlikesky98-574 (477frames,0queries,84writes), and
righttallholdball635-989 (355frames,1query,69writes). Invalid-GT frames, sequence
boundaries and frame gaps split episodes. The260invalid-GT frames and97initial
frames are excluded. These are frame-weighted continuous geometry diagnostics,
not the official sequence-averaged benchmark metrics.

## Causal test

`scripts/eval_template_freeze.py` uses the same parent checkpoint, model
architecture, K8, cost0.15, initialization and causal box/history/query logic.
After each normal tracking call it restores both initial templates. This removes
committed dynamic-template changes while leaving the current prediction intact;
future predictions and decisions may naturally diverge. No GT after initialization
is used for decisions. Ground truth is saved only for later scoring.

Every frame checks that the templates equal the initial copies; model tensors
are checked unchanged at completion. A31-frame native smoke must actually observe
and revert at least one mutation. A full test then covers97seq/54,646frames and
uses the same official CPU scoring verifier. All runs are dev diagnostics and
test-ineligible. The implementation still computes the normal template crops
before discarding them, so timing is not an optimized static-template benchmark.

Interpret full-dev comparison as a net effect of removing both good and bad
updates. If it improves, investigate a selective memory-admission mechanism;
if it worsens, do not infer that bad writes were harmless, but reject freezing
as the operational solution. Do not label this ablation as a novel core module.

## Open-source references

[OSTrack](https://github.com/botaoye/OSTrack/blob/main/lib/test/tracker/ostrack.py)
initializes its template once and passes that stored template to subsequent
tracking calls. It provides a concrete fixed-template reference.

[STARK-ST](https://github.com/researchmm/Stark/blob/main/lib/test/tracker/stark_st.py)
keeps the first template and updates additional templates at configured intervals
using the sigmoid of separate classification logits. This differs from using a
localization heatmap peak as the reliability signal. It motivates testing memory
reliability, not directly transplanting its score head or claiming that its
threshold transfers to Dual-RGBT.

No reference implementation was copied into the model. Any later memory module
needs train-only learning, full-dev evaluation and comparison with the protected
historical E1 as well as this parent. The three requested validated innovations
and final benchmark targets remain open.

# Failure-Driven Source References

## Evidence and Current Decision

This note connects the actual matched TRAIN-prefix findings to inspectable
open-source mechanisms. It does not replace the current P1B collection,
admit an architecture change, establish novelty, or report benchmark gains.
The current sequence remains full collection acceptance, five-epoch FIT,
fresh round2 trajectories and FIT, followed by separately admitted evaluation.

The evidence is 32 matched TRAIN sequences and 3,052 frames per branch,
with different branch trajectories:

- DropMoE queries on 78.9646% of sampled states; SOT on 67.6933%.
- Positive immediate cost-adjusted query opportunities in the selected
  modality occur on 46.6252% and 45.7405%, respectively.
- At least 987 DropMoE and 670 SOT queries lack a strictly positive
  immediate net query option. This is an aggregate lower bound, not exact
  false-query classification or an assessment of future template benefits.
- Outside-crop target centres occur on 8.1258% and 5.8322% of states.
- Saved prediction mean TRAIN IoU is 63.3805% and 64.2604%.

These findings support measuring policy calibration and crop coverage.
They do not prove a backbone defect, missing-feature bottleneck, memory
contamination rate, or causal benefit from any method below. Existing
clamped localization targets cannot establish recovery performance.

Raw evidence and definitions:
`reports/E0_EXECUTOR_V2_P1B_MATCHED_PREFIX_FINDINGS_20261102.md`
and the two corresponding MATCHED_PREFIX JSON reports.

## Verified Repositories

Repositories were shallow-cloned for source inspection under local
`work/references/`; no dependency installation, import, training or source
copy into Dual-RGBT occurred. The inspected revisions are pinned below.
Published performance, venue claims and reproducibility were not audited.

### UATrack: Evidence Requires an Observable Input

- Repository: https://github.com/dongdong2061/UATrack
- Revision: `a600bb29cc7a45e6a9fb2822aa08722ba99c4c71`
- Fusion source:
  https://github.com/dongdong2061/UATrack/blob/a600bb29cc7a45e6a9fb2822aa08722ba99c4c71/lib/models/bat/uncertainty_fusion.py
- Training loss:
  https://github.com/dongdong2061/UATrack/blob/a600bb29cc7a45e6a9fb2822aa08722ba99c4c71/lib/train/actors/bat.py

`SelfAttention_For_Fusion_uncertainty.forward` consumes RGB and TIR
features and attention tensors. It averages attention over heads, constructs
positive evidence with Softplus+1, computes uncertainty from evidence
strength, and weights the two feature streams. Its token-weight min/max
is over the supplied tensor; it is not an independently calibrated
probability of localization success. `TemplateRouter` pools two template
features and applies a shared MLP and softmax. The actor contains an
uncertainty weighting loss using TRAIN foreground masks.

Our C1 evidence/query decision occurs before optional auxiliary ViT work.
Both full attention streams therefore are not observable at that boundary.
The useful transfer is the distinction between evidence strength and a
raw peak score, with supervision tied to localization outcomes. A cheap
preview/history evidence head must be trained and tested with only its
actual decision-time inputs. It cannot inherit the external fusion head's
features, calibration claim, or timing.

First falsifiable check after the current policy refresh: determine whether
chosen actions align better with freshly recomputed net query utilities.
A later evidence-head experiment must compare under the same executor and
cost, and count full auxiliary computations in FPS and MACs. No new
threshold is selected from the existing TEST results or this prefix.

### GOLA: Adaptation Mechanics, Not Our Route Semantics

- Repository: https://github.com/MelanTech/GOLA
- Revision: `339c737cda6a24be667b6e5abdc721e8d6046f05`
- LoRA implementation:
  https://github.com/MelanTech/GOLA/blob/339c737cda6a24be667b6e5abdc721e8d6046f05/trackit/models/methods/GOLA/modules/lora/__init__.py
- Merge:
  https://github.com/MelanTech/GOLA/blob/339c737cda6a24be667b6e5abdc721e8d6046f05/trackit/models/methods/GOLA/modules/lora/merge.py
- Grouped layer and orthogonality loss:
  https://github.com/MelanTech/GOLA/blob/339c737cda6a24be667b6e5abdc721e8d6046f05/trackit/models/methods/GOLA/modules/gola/__init__.py

The ordinary LoRA layer computes a scaled low-rank residual. The grouped
layer concatenates retained and grouped A/B parameters; separate helper
code computes pairwise orthogonality penalties. Merge code forms B@A in
FP32 and adds the scaled update to the base linear weight.

The source supports low-rank adaptation and merge mechanics. Its grouped
parameters do not by themselves implement our semantic execution routes:
primary385, local193 and expanded385. A C2 route-conditioned adapter would
be a separate design with explicit route ownership. Missing-modality
features and the FiLM proposal in the supplied plan also are not provided
by this GOLA implementation.

Do not promise zero inference overhead for dynamic route-specific adapters.
One merged shared weight cannot represent multiple different route updates
simultaneously. Keeping multiple merged copies trades memory for runtime;
unmerged low-rank branches add operations. Measure the chosen option.
Before any full experiment, require initial no-op parity, same-route
merged/unmerged parity, and correct route-isolated gradients. These are
design requirements, not completed Dual-RGBT tests.

The prefix alone does not prove C2 is needed; complete the currently
admitted policy refresh first. If introducing C2 later, test missing-feature
completion and route adaptation as separate factors before combining them.

### DTPTrack: Protected Anchor and Gated Historical Reading

- Repository: https://github.com/NorahGreen/DTPTrack
- Revision: `8a6d04c2b918667f1717f73555dca8198d3b11a0`
- Model source:
  https://github.com/NorahGreen/DTPTrack/blob/8a6d04c2b918667f1717f73555dca8198d3b11a0/trackit/models/methods/DTPTrack/DTPTrack.py

The inspected model pools masked template summaries, applies a small MLP
ending in sigmoid, leaves the first template's weight at one, gates later
summaries, and transforms those summaries into prior-token modulation.
The code path is a historical-read weighting mechanism. It does not prove
that the gate is a calibrated probability, implement our template-write
acceptance rule, or supply RGBT-specific identity checks.

For planned C3 memory, preserve the immutable initial identity anchor and
distinguish admission into memory from reading its contents. A write head
answers whether a new observation is suitable for storage; a read weight
answers which already stored observations are useful now. Gating only reads
cannot establish safe writes. The existing sampled outside-crop frequency
is not evidence of a measured memory contamination rate.

Later acceptance criteria should include immutable first-template checks,
rejected-write state equality, controlled harmful-write counts on TRAIN
trajectories, and a fixed full-benchmark comparison after training. These
checks must use cache features and labels designed for memory admission;
the current P1B utility cache is not automatically a full memory dataset.

## Correction to the Supplied Source Inventory

The supplied plan names
https://github.com/Multi-Modality-Tracking/SCDT-CVPR2026
for missing-modality reconstruction. On 2026-11-03 UTC, a direct
`git ls-remote` returned exit0 with no advertised refs. No implementation
revision could be inspected. Treat SCDT as an unverified source lead, not
available implementation evidence. Do not claim to have reproduced or
adopted its reconstruction code.

GitHub's unauthenticated API was rate-limited during this audit; direct
Git read-only clones succeeded for the three pinned repositories above.
This note relies on those actual source checkouts.

## Proposed Ownership, Not Established Innovations

The supplied plan's C1/C2/C3 names refer to future executor-v2 components.
They must not be silently equated with the older protected M1/M2/M3
failure/identity/memory route or treated as trained replacements.

1. C1: decision-time evidence and query acceptance, supervised using the
   actual frozen/current executor's counterfactual localization utility.
2. C2: missing-modality representation and execution-route adaptation;
   these mechanisms require their own evidence and separated ablations.
3. C3: identity-protected memory admission and reliability-weighted reading.

These are candidate responsibilities. Existing uncertainty, LoRA and memory
gating literature prevents claiming those generic mechanisms as our novelty.
Any eventual contribution must be supported by the specific causal
interfaces, supervision, computational savings and full benchmark results.
No claim that these three candidate components meet the goal is made.

# Global auxiliary-content diagnostic: independent source review

**SOURCE_PASS — zero blocking issues.** Review is same-family/provisional;
requested `gpt-6-astra` / `max`, actual routing **UNATTESTED**.

The reviewed source32 is exactly the old opportunity source31 plus
`scripts/diagnose_global_content.py`. All 32 actual file hashes match the new
plan, and all files parse. The active training source30 and training-plan
`b5cf7d472bb1a09787d57a1a15f95cdd9989f0b102908189a74c975dbe7daafd`
are unchanged. The new plan SHA256 is
`7797bae9b9196467d4a56194927a9a1b01477a1d002df266dce2bb3b121afd75`.

## Findings

- **Content controls:** the three conditions share coordinates, auxiliary
  templates, cached main features and token budget. Zeroing changes only
  normalized auxiliary search channels. The half-width/height cyclic roll
  changes only auxiliary search spatial correspondence. The reader selects
  only auxiliary search/template channels, with primary search channels
  checked equal. Local/wide inputs remain 128/256 pixels and 193/385 tokens.
- **Reference state:** templates/history/previous state are captured before
  the factual selected-primary STOP call. The inherited tracker uses its own
  predictions for search, history and selected-template update. Current GT
  appears only in current diagnostic controls and IoU/coverage after first-frame
  initialization. Counterfactual outputs never write future tracker state.
  `np.array_equal` requires the selected-primary recomputation to reproduce STOP.
- **B and freezing:** the driver gates native full B completion/exit0,
  6,250 updates/50,000 pairs, stage B, non-sanity metadata and exact input/source
  hashes. Both the main model and strict-reload clone are eval and explicitly
  frozen. Diagnose/shared probe use no-grad. There is no backward, optimizer or
  checkpoint save in the diagnostic. Input checkpoint/source hashes are checked
  again at completion.
- **Summaries:** six regions × three contents give 18 comparisons/state and
  31,536 rows at 1,752 states. TRUE minus ZERO/SHIFT is paired by state within
  region. The main-centre-outside subset uses counts, sums and sign counts;
  a zero-size subset has no fabricated mean. Full population means remain valid
  because all content conditions are generated for every state.

## Actual checks and receipts

The stdlib/AST checker passed **150/150** checks. The cached NumPy 1.26.4 checker
passed **107/107** checks across both modalities and 128/256 inputs, plus six-region
paired arithmetic and empty-subset cases. These are AST-extracted pure functions
with a NumPy adapter; they are not PyTorch, real-image or NN validation.

| Actual local run | Exit | Evidence |
|---|---:|---|
| Input snapshots | 0 | `snapshot_v1.receipt.json` |
| Supplementary source evidence snapshots | 0 | `snapshot_supplement_v1.receipt.json` |
| Source/AST/plan checker | 0 | `source_check_v1.receipt.json` |
| NumPy algebra/aggregation checker | 0 | `numpy_check_v1.receipt.json` |
| Final input seal and receipt verification | 0 | `final_seal_v1.receipt.json` |

There were **no failed attempts** in this review. All checker source, stdout,
stderr and actual exit records are retained under
`C:/Users/gb/.codex_tmp/dual_rgbt_global_content_source_review_20261014`.
The final check found **50/50 reviewed inputs unchanged**, including source32,
plans, protocol, public historical evidence and the two skill instruction files.
The JSON report carries exact receipts and hashes.

## Runtime acceptance remains pending

This is SOURCE acceptance only. No server call, GPU query, PyTorch/project import,
NN call, model build, raw manifest/GT/checkpoint read or training execution occurred.
Existing public A/B/C admission records separately show real updates 10/2/2,
allowed changed tensors and strict reload; the unchanged training source requires
finite nonzero gradients and optimizer steps. The new read-only driver supplies
no new training-path gradient evidence.

Wait for the original full B native COMPLETE/exit0 and free physical GPU1. Then
the new sanity must actually produce 10 tracks, 2 labelled states, 36 comparisons,
265 unchanged state tensors, the exact seven-field strict-reload probe, and zero
backward/optimizer calls. These counts are specified in the protocol and supported
by the previous frozen-prefix run, but the driver reports some of them instead of
fully asserting them; inspect the new native/SANITY/COMPLETE artifacts before
formal execution. This evidence limitation is nonblocking at SOURCE stage.

Formal acceptance requires the same completed B and 20 sequences / 8,854 tracks /
1,752 states / 31,536 comparisons, frozen state and preserved checkpoint/source
hashes. SOURCE_PASS is not permission to run alongside unfinished B.

## Interpretation and privacy

A TRUE advantage is conditional evidence of auxiliary-search content contribution
at these fixed inputs. GT-centred regions still supply correct-location prior;
their paired result cannot prove a deployed proposal finds the target. Zero input
retains templates and learned priors. Cyclic shift tests spatial correspondence,
not independent distractor identity. A and B own-state visits may differ, so their
aggregate difference is not a paired causal estimate. One seed establishes no
statistical stability, PR/NPR/SR, latency, trajectory bound or three-contribution
success claim.

`content_comparisons.csv` contains GT-derived coordinates and is **PRIVATE_ONLY**.
Keep it under server runs / local `.codex_tmp`; only safe aggregates and native
source/admission evidence may be published. This report includes no raw rows or GT.

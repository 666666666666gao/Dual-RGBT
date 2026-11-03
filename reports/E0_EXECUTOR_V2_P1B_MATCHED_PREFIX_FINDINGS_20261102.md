# Matched TRAIN Prefix Findings

The reviewed CPU diagnostic read the same first eight completed sequences
from each of four shards on each branch: 32 ordered TRAIN sequences and
3,052 stride-five records per branch. Sequence order and per-sequence record
counts match exactly. Manifest and executor bytes match their bound hashes;
token cost comes from each actual executor checkpoint's FP32 buffer.
No model forward, optimizer, CUDA initialization or collector modification
occurred. These are different branch trajectories on matched sequence/frame
identities, not identical tracker states or full benchmark measurements.

| Diagnostic | SOT | DropMoE |
| --- | ---: | ---: |
| Saved predictions' mean TRAIN IoU | 64.2604% | 63.3805% |
| Target centre outside search crop | 178/3052 (5.8322%) | 248/3052 (8.1258%) |
| Actual query action | 2066/3052 (67.6933%) | 2410/3052 (78.9646%) |
| Positive gain available in selected modality | 2197/3052 (71.9856%) | 2213/3052 (72.5098%) |
| Positive gain after query cost in selected modality | 1396/3052 (45.7405%) | 1423/3052 (46.6252%) |
| Mean best cost-adjusted gain in selected modality | 0.13483337 | 0.15332028 |
| Mean chosen-action cost-adjusted regret | 0.07124273 | 0.07651608 |
| Checked quality/gain/target values finite | All | All |

For a saved state, the diagnostic compares STOP=0 and the four saved
counterfactual localization-loss gains minus the actual FP32 token cost
times cost_weight=0.05. Regret is their maximum minus the actual action's
value. It is not a benchmark loss, global optimality result or causal
prediction of a future policy's performance.

The count difference proves at least 670 SOT and 987 DropMoE query actions
occurred on sampled states with no strictly positive cost-adjusted query
option in the selected modality. This is a lower bound; aggregate counts
do not identify the exact overlap. These are one-frame localization
utilities and do not include future template or history effects.

DropMoE's higher query frequency and lower sampled IoU are consistent with
a policy calibration issue after executor changes, but do not prove its
cause or justify a threshold change. The two branches produce different
states. All gains were collected with the current frozen behavior preview;
the admitted FIT recomputes targets using its current preview each update.
Outside-crop targets also retain the existing clamped-loss limitation:
this diagnostic does not establish recovery capability.

Continue the admitted full collection and separately admitted five-epoch
policy refresh after actual full acceptance. Do not alter cost, choose a
checkpoint or launch a new grid from these prefix statistics. No new full
TEST score, convergence result or full collection acceptance is claimed.

Evidence:

- `E0_EXECUTOR_V2_P1B_SOT_MATCHED_PREFIX_20261102.json`
- `E0_EXECUTOR_V2_P1B_DROPMOE_MATCHED_PREFIX_20261102.json`
- `E0_EXECUTOR_V2_P1B_PREFIX_DIAGNOSTIC_SOURCE_REVIEW_R2_20261102.json`

The original diagnostic R1 had two engineering errors: rounded token costs
and unverified manifest bytes. Both were fixed before remote execution,
with 44 independent CPU checks passing. R1 FAIL and its exact source remain
archived; R2 is SOURCE_PASS0, UNATTESTED/same-family/provisional.

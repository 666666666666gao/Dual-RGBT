# Parent TRAIN Confidence Gate Diagnostic

## Scope And Provenance

This is a read-only analysis of existing parent21e07 TRAIN caches, not an
experiment on the protected E0 checkpoint. It reverified935bound input hashes
and reconstructed45317actually deployed sampled states from882sequences:
706fit/35740states and176component-holdout/9577states.

Source report SHA256:
`111358d02a5520997c5e15026550d92e39abdf4c81384f31dbbb771d8c421e30`.
Result: `reports/PARENT_CONFIDENCE_GATE_DIAGNOSTIC_20261031.json`,
SHA256 `d67ca7475b030b9b362588213e9aadceb0b72ef8d9a17bb1cdfeb0ac5bd87d44`.
Analysis source SHA256:
`60273324d723f74c836022c922cdce0683a1c1e016ce599419a12394caf7e187`.
No new image inference, parameter updates, test selection or policy deployment.

## Raw Results

Lost means final IoU<.1. Good localization means final IoU>=.5, not an explicit
target-identity label. A diagnostic gate triggers when confidence<threshold.
The sole fitted threshold0.5916850566864014 was selected using fit-good states
only to retain at least95percent of them. Holdout and test are not used to fit it.
The existing .65 threshold is a fixed reference, not a newly selected alternative.

| Partition / Gate | Lost Detected / Total | Lost Missed | Good Interrupted / Total | Good Retained |
|---|---:|---:|---:|---:|
| Fit / fit95 threshold | 3652/9323 (39.172%) | 5671 | 1082/21645 | 95.001% |
| Component holdout / fit95 threshold | 793/2280 (34.781%) | 1487 | 375/6319 | 94.066% |
| Fit / fixed .65 | 4116/9323 (44.149%) | 5207 | 1648/21645 | 92.386% |
| Component holdout / fixed .65 | 891/2280 (39.079%) | 1389 | 519/6319 | 91.787% |

Confidence-based lost-vs-good AUROC is0.718993737fit and0.626060225holdout.
Intermediate IoU states are excluded from AUROC only:4772fit/978holdout.
All states remain in gate-trigger counts. Exact confidence ties receive average
ranks; strict threshold ties do not trigger.

## Implication For The Redesign

1. At the fit-selected normal-localization protection point, the held component
   still misses1487/2280lost states,65.219percent. The fit95retention promise
   does not transfer exactly to holdout:375good states are interrupted and
   retention is94.066percent. Do not advertise a95percent preservation guarantee.
2. Raising the reference threshold to.65 detects98more held lost states but
   interrupts144more good states. This is not measured recovery improvement:
   a trigger does not establish that any candidate can recover the target.
3. This operating point supports testing separate failure-reliability evidence
   rather than presenting a response-threshold change as M1. It does not prove
   that every scalar gate is useless or that any new learned head will improve
   E0. Do not transfer the parent threshold into protected-E0 deployment.
4. New E0 data, explicit ROI/target evidence and matched closed-loop outcomes
   remain necessary. The paused full-TRAIN collector still requires its own
   independent source review; this analysis does not grant that admission.

## Verification And Limits

Five focused tests pass locally and on the actual26CPU-scoring environment:
perfect/reversed/tied ranking, intermediate-IoU exclusion, strict threshold
ties, fit-good retention, and holdout isolation. An alternative direct-pairwise
calculation across14407320held lost/good pairs reproduces AUROC0.6260602249412104.
This is a deterministic arithmetic cross-check, not independent scientific review.

The upstream parent has seen the component holdout. These are sampled, correlated
TRAIN states, not every frame, unseen data, benchmark PR/NPR/SR or verified
identity classes. Differences between fit/holdout AUROC alone do not establish
overfitting or a causal explanation. No independent result-audit acceptance is
claimed. Existing GPU tasks and original model/tracker sources are unchanged.

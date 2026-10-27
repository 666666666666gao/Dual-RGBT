# Backend evaluation binding review R1

**FAIL / SOURCE_FAIL — two bounded findings.** Requested gpt-6-astra / max; actual model and effort UNATTESTED. Same-family, provisional. This was a source review, with no NN, SSH or weight reads.

1. `tools/joint_candidate_results.py:85` directly joins annotation hash keys containing backslashes. All 211 RGBT210 and 469 RGBT234 original25 keys have this form; on POSIX they are single filename components. Normalize separators at this hash lookup; preserve manifest bytes, expected digests and metric math.
2. `dual_rgbt_global/tracker.py:10,29` imports/calls unsealed `lib/test/utils/hann.py`; `dual_rgbt_global/candidate_tracker.py:31` uses the resulting window in inference. The actual local package chain also contains six unsealed files listed in `R1_review.json`. Add existing hashes to evaluation seals only; preserve active training source.

The provider plumbing is correct. Original8 is a subset of26;24 and8 configuration tuples are disjoint,32 total. Source counts77/80/80; all89 protected file hashes match. Source guard records were stale/missing; no actual evaluation is authorized by registration.

R1 was communicated before revision. This artifact was serialized from the verified `before_r2` archive after R2; it does not relabel either failure. Exact R1 hashes are in `R1_review.json`.

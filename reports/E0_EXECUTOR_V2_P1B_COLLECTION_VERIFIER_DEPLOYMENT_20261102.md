# P1B Collection Verifier Deployment

The additive CPU verifier and its source review were deployed to both active
collection snapshots. Remote SHA256 checks matched, and all 98 original
runtime source hashes remained unchanged on each host.

- P25: `/home/gaob/Dual-RGBT/snapshots/p1b_dropmoe_p25_r1_20261102`
- P26: `/data2/gb/Dual-RGBT/snapshots/p1b_direct_r3_20261102`
- Verifier: `tools/verify_e0_policy_refresh_collection.py`
- Verifier SHA256: `cb1d6e8cb4ef38c4ac375e63ae897ceb0ec25e6f11418e71687430c5a8d801b2`
- Review: `E0_EXECUTOR_V2_P1B_COLLECTION_ACCEPTANCE_TOOL_SOURCE_REVIEW_R1_20261102.json`
- Review SHA256: `435a038a65e5499eab7e5c3a9a9f5f1925d9b1221a55353111e28eca3bb14059`
- Review result: SOURCE_PASS, zero blockers; UNATTESTED, same-family, provisional.

The verifier has not been executed against either full collection. Each
branch must first finish its four actual native collector tasks with
COMPLETE/exit0. Run the verifier from the corresponding snapshot cwd with
its own interpreter and four ordered plans; write a new report outside
collector runs and original sources.

The disabled full-FIT preparer and tests remain local. This deployment
does not establish actual full collection acceptance or admit full FIT,
round2, or benchmark execution.

At 2026-11-03 01:31:40 UTC all eight wrappers and eight children were live.
P25 had completed 57 sequences, 24,833 tracked frames and 4,900 records;
P26 had completed 174 sequences, 83,472 frames and 16,581 records.
Each branch targets 882 TRAIN sequences, 458,553 frames and 91,091 records.
P25 GPU temperatures were 79/69/68/63 C with all four power limits at 250 W.

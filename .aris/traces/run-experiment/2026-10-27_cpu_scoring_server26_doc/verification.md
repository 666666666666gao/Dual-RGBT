# Fresh documented invocation verification

Verdict: PASS. Native process exit code: 0. The independent verification command from `docs/CPU_SCORING_SERVER26_20261027.md` was executed once, verbatim, by `exec_command` in Windows PowerShell (`powershell.exe`), without a capture shell wrapper, retry, repair, or build invocation.

The native output contains `CPU_SCORING_WITNESS_PASS`. No divergence was observed between the documented expected exit/sentinel and the actual output.

Observed witness facts:

- Prefix: `/data2/gb/Dual-RGBT/conda-envs/cpu-scoring`.
- Python 3.10.14, NumPy 1.26.4, Matplotlib 3.8.4.
- Annotation identity: LasHeR 245 sequences / 220703 frames; RGBT210 210 / 104706; RGBT234 234 / 116650.
- Seven synthetic metric fixtures: PR, PR_LasHeR, NPR, SR, SR_LasHeR, MPR, MSR.
- CPU kernel SHA256: `2ffd58662c9b886e5fd79aff72ee4bd4ed051301ad882563d851dc5bad5faed7`.
- Source SHA256: `0642d7bd7b200acac0b66dcef58ee9596ae54593bf769a21c720fbf2de4b5d3b`.
- NN runs: 0; checkpoint reads: 0; model-prediction reads: 0.
- The output explicitly limits this witness to real annotation identities and synthetic metric fixtures; synthetic values are not tracking accuracy.

The documented command wrote the designated new remote report path: `/data2/gb/Dual-RGBT/reports/cpu_scoring_doc_verification_server26_20261027.json`. No extra readback command was run. No training/GPU progress query, source modification, ledger update, or HANDOFF edit was performed.

The ledger was observed in its pending state (`INSTALLING_PENDING_PARENT_AND_FRESH_DOCUMENT_WITNESS`); final ledger disposition remains the parent's responsibility. Bundle/package provenance and implementation details were not separately inspected under this task's document-only scope.

Requested reviewer: `gpt-6-astra`, effort `max`. Actual model and effort: UNATTESTED. Review independence: same-family. Acceptance status: provisional.

`native-tool-result.json` preserves the complete native tool response, including its exact output string and exit code. `invocation.ps1.txt` preserves the command. `review.json` contains the structured verdict and evidence limits.


# P25 DropMoE FiLM Source-Plan Recovery Audit

Observed 2026-11-04 04:37 UTC. This was a read-only lineage-recovery check.
No source, collection plan, receipt, model, or active task was changed.

## Finding

The two original reviewed source-plan byte sequences were not recoverable from
the local project or the reachable GitHub `main` history:

- s0 expected SHA256:
  `68a188a1e46374668d4cdf1f2f915fe4fe2bdef27aa4814d44677931669c0c46`
- s2 expected SHA256:
  `987e46551ec3c843a46f8c5d6ca8a6f6f239d4d2e0ad16b5a15aff6b9a856e1e`

The existing P25 runtime archive contains exact runtime task-plan bytes,
not the original reviewed source-plan bytes. Its s0 and s2 plan hashes are
`5e4ce47af18e1c5a11841b4c499446034c35acecaf3b3e2035b1375633e2254c` and
`a8dbca765a47223b5b392efd105ce894aacb0411ea2a23615a29981e4bb494ad`.
The source-snapshot `canonical_plan.json` files match those runtime hashes,
not the missing expected source hashes.

Read-only recovery checks:

- Searched 2,859 local project JSON files for either exact SHA256; no match.
- Checked 235 Git blob objects in the relevant 14-18 KB size interval; no
  matching raw-byte SHA256.
- Checked the reachable GitHub `origin/main` history. Local `HEAD` and
  `origin/main` both resolve to `2178c9e35eb9fdfb0ee05c36a63d997e3d806af4`;
  neither target plan path is present in that history.
- The prior remote search recorded in the runtime archive report did not find
  the exact source plans under the expected P25 isolated-root paths.

## Admission Consequence

The launcher validates the source pin against the review embedded in each
remote collection plan, including exact source-plan path, raw SHA256,
canonical binding, task, and GPU. The collector command and plan SHA are also
bound into the finalized run receipt. An outer-plan-only successor cannot
replace that embedded review. Editing a completed shard plan would invalidate
its existing config/receipt binding; treating an archived runtime plan as the
original source draft would misstate provenance.

Accordingly, the current P25 FiLM replay remains blocked. Do not launch from
the disabled plan or weaken the launcher/verifier checks. A future route must
preserve the original evidence distinctly and provide a valid, fully rebound
plan/config/receipt chain; no such chain is currently present. The archived
metadata alone does not contain the 882 payloads or checkpoint/manifest
bytes. Replay acceptance, if eventually admitted, remains TRAIN replay
verification only, not policy FIT or benchmark evidence.

## Live Tasks at Observation

- P25 zero replay verifier remained native `RUNNING`, wrapper/child
  `1806777/1806778` on GPU1. Child cumulative `rchar` was
  109,916,286,000 bytes and `read_bytes` 11,408,334,848 bytes. GPU1 was
  66 C / 206.96 W under the 250 W / 85 C cap. GPUs0/2/3 had no compute
  process.
- P26 zero replay R6 remained native `RUNNING`, wrapper/child
  `2322989/2322990` on GPU3. P26 FiLM shard0 remained native `RUNNING`,
  wrapper/child `2360743/2360744` on GPU1; its cumulative `rchar` had
  increased from 27,074,346,585 to 27,488,713,930 bytes between checks.
  GPU1 was 60 C / 172.60 W; GPU3 was 73 C / 176.92 W, both under limits.
  GPUs0/2 had no compute process.

No new GPU task was launched. Once P26 FiLM shard0 reaches native COMPLETE
with exit 0 and all four receipts are checked, the next candidate is a
versioned P26 FiLM replay plan followed by exact source review and canonical
runtime admission; GPU2 is the candidate card, not pre-authorized work.
P25 idle cards remain unused until its lineage blocker is resolved.

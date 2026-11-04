# C2 Policy FIT Draft Preparation

`work/prepare_c2_policy_refresh_fit.py` prepares a disabled local draft after
a matching C2 TRAIN replay has actually passed. It does not connect to a
host, deploy sources, inspect current GPUs or launch training.

Inputs must be unchanged archived bytes from the real full replay:

- Acceptance report: PASS, full 882 sequences / 458,553 frames / 91,091
  records, matching C2 executor, four matching collection plans and evidence
  rows, CUDA replay, and no benchmark or FIT acceptance claim.
- Native status: COMPLETE/exit0, with the exact verifier command, four
  collection plan paths and output path corresponding to that report.
- Explicit remote receipt/status paths, host Python, distinct FIT source
  root, run parent, fresh task ID, GPU and future exact-review path.

Use `E:/Anaconda/python.exe -B work/prepare_c2_policy_refresh_fit.py --help`
for the required arguments. `--acceptance` and `--native` are local raw
archives; `--acceptance-path` and `--native-path` are their original remote
paths. The output directory must not exist.

The output contains `plan.json` and `binding.json`. It preserves the
matching collection seed, cost, stride, executor binding and original-host
source hashes, adding the C2 FIT entry hash. It fixes the already-used
policy recipe: five epochs, batch 512, four workers, preview LR 4e-5,
quality/gain LR 4e-4, AdamW weight decay 1e-4 and gradient clipping 1.0.
The training entry retains the last five epochs and strict-reloads them.
No active training recipe is changed.

The draft has `launch_allowed=false` and a PENDING review SHA. A canonical
binding is a review target, not an admitted execution. No valid real draft
has been produced while all three replay tasks remain RUNNING.

Next steps after a real PASS: archive the acceptance and native terminal
bytes once, prepare the matching draft, create a separate FIT source root
from the exact corresponding host's pinned sources, and obtain exact
plan/source/input review plus fresh runtime admission. Two local E0 source
files differ from the collection snapshot pins; do not deploy the current
dirty local tree to satisfy the inherited source map. Preserve original
snapshots and active replay tasks.

Tests: the preparer has nine CPU contract cases; combined with the existing
replay launcher contracts, 27 tests pass. These use fixtures, not real
replay receipts or an end-to-end GPU FIT. Source review is separate from
later plan admission. TEST/dev remain disabled and targets remain unmet.

Exact source review:
`reports/P2_C2_POLICY_REFRESH_FIT_PREPARER_SOURCE_REVIEW_R1_20261104.json`,
SHA256 `63180b2ddf1a17cea6518edf61b33b998126ffe9be140513281ad6fbd20a26ea`.
It is SOURCE_PASS/0, UNATTESTED, same-family/provisional, with no scientific
acceptance, admitted plan bindings or execution bindings. The reviewer
independently reran the nine preparer tests. Later runtime admission must
authenticate the pinned native status separately: the FIT trainer does not
consume the additional `collection_acceptance_native` field itself.

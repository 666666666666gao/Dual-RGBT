# Context Advantage Execution Preparation

The original relative-head experiment failed every TRAIN qualification gate.
The next candidate is the context advantage proposal documented in
`E0_M2_CONTEXT_ADVANTAGE_IMPLEMENTATION_20261102.md`, using the same accepted
TRAIN split, candidate set, frozen epoch9 appearance head and original gates.

Core/data source review is now SOURCE_PASS/zero blocking issues:
`E0_M2_CONTEXT_ADVANTAGE_CORE_DATA_SOURCE_REVIEW_R1_20261102.json`.
Its exact three-file hashes match current local core/data/tests. Attribution
is UNATTESTED, same-family, provisional; no scientific acceptance is claimed.

New execution files provide:

- CPU export on the original P26 producer root, after accepted collection
  verification and exact parent/source/core-review bindings.
- Exported tensor and receipt hashes transferred to a dedicated P25 snapshot.
- Native command, interpreter, hostname, GPU and wrapper-parent binding.
- A five-update GPU smoke with finite gradients, changed parameters, exact
  reload, all-KEEP initialization and unchanged fit-only normalization.
- A separate full run starting from the same zero-output head, after smoke
  artifact verification and CPU decision replay. The smoke weights are not
  the full run's initialization.
- Fixed 12 epochs/348 updates, seed2027, batch512, AdamW LR3e-4/WD1e-4;
  latest, last five and a qualifying best retained.
- Terminal verification of native/config/history/receipt, normalization,
  retained/latest/best CPU decisions, original gates and checkpoint content.

Nine local CPU behavioral tests pass, including actual five-update execution
and reload on synthetic TRAIN-shaped fields. These are not actual cache
export, GPU smoke, full TRAIN or benchmark evidence.

R1 execution review discovered a terminal-verifier defect: comparing byte
hashes of separately saved latest and epoch12 files can reject identical
checkpoints because serialization includes their file names. R1 exact source
and draft bytes are preserved in `source_snapshots/context_execution_r1_20261102`.
The revised verifier preserves per-file SHA verification and compares
checkpoint metadata, state keys and every weight tensor for exact equality.
A real save-to-two-paths test also rejects metadata and tensor tampering.

R2 disabled drafts are under `e0_context_advantage_r2_20261102`:

| Phase | SHA256 |
|---|---|
| export |3bf431a8433edb4ec9adb99cedfd36bc5b3ad085d8fcb98de120934b299afe09|
| smoke |24a0fb48bb0b8f07934a9f6549ab5586ac1774b8f5304f7af2946b0384299a3b|
| full |6a45ce9880afb839cba9355b405f2a29c43ac7fb07bed3614e1cf4b75b41056d|

All bind the same 163-source map: 154 sealed original producer sources plus
nine new context implementation/test files. The target is P25 GPU2 at
`/home/gaob/Dual-RGBT/snapshots/context_advantage_r2_20261102`; export root is
`/data2/gb/Dual-RGBT`. Existing P25 evaluation files must not be overwritten.
Warm-reuse the accepted P25 Python3.7.16/Torch1.12.1+cu113 environment without
package changes; its independent provider ledger remains authoritative.

R2 execution review is pending. No context source deployment, actual tensor
export, GPU smoke or new full TRAIN has occurred. After SOURCE_PASS with exact
bindings: deploy additive sources to P26, run remote CPU tests and export;
copy the reviewed snapshot and data to P25; verify 163 sources, receipt/data
bytes and capacity; run one smoke; verify its actual terminal/native evidence;
bind one full run. CPU tensor export and cached-head training are not
end-to-end E0 training or closed-loop tracking.

No gate changes, TEST calibration, seed sweeps or rejected-head promotion.
Without a qualifying head, retain protected E0. A future context runtime
needs a separate source review and actual paired TRAIN trajectory acceptance.

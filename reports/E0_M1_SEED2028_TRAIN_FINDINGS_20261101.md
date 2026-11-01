# M1 Seed2028 TRAIN Replication

Host26 task `train_protected_e0_m1_seed2028_r1_p26_20261101` ran onGPU2
with wrapper1886623/child1886627 and completed with exit0 in30.31seconds.
The unchanged trainer performed12 epochs/216 optimizer updates. All9
receipt artifact hashes passed; best/latest and numbered epochs8-12 remain.

Only seed2027->2028 changed in the training recipe. The six-source map,
architecture,882 TRAIN sequences,706fit/176component-holdout split,
fit-only normalization, batch4096, LR.001 and AdamW weight decay1e-4
remain unchanged. Different seeds affect initial head parameters and shuffle.

| TRAIN Component-Holdout Quantity | Original Seed2027 | Seed2028 |
| --- | ---: | ---: |
| Selected epoch | 5 | 3 |
| Macro recall (%) | 53.112501 | 53.531759 |
| Lost recall (%) | 42.384271 | 40.965655 |
| Reliable interruption (%) | 13.189174 | 11.778605 |

Selection is solely highest TRAIN component-holdout macro recall, as
implemented by the unchanged trainer. Improved macro recall accompanies
lower lost recall; this is not a uniform improvement or justification to
promote a replacement based on TEST. Upstream E0 saw component-holdout
sequences; these are not independent generalization or tracking metrics.

Fixed selected head SHA256:
`d747dc25f9f8acf0abfe137b268b4a6766a3d8785ea0193865297e116351c20e`.
Final plan SHA:
`19033de1d17a5bdfa239e1992782c798946a939e3c37301552b7e19591bdbfba`.
Review SHA:
`d49eb4af4893a46ca1c4b6aa9b4fd338d21bb0951b2dfb05f07b1cca0b183a90`.
The review is same-family/UNATTESTED/provisional, scientific acceptance false.

Exact config/history/COMPLETE/native evidence is archived under
`reports/e0_m1_seed2028_runtime_20261101/`. The training plan was proposed
after original TEST outcomes were observed, explicitly disclosed in its
evidence; it is not preregistered.

Separate three-benchmark exploratory replication drafts and a versioned
entry are prepared. New evaluation admission remains pending a fresh review
of the entry, fixed head and all five smoke/full draft hashes. The entry
changes the imported evaluator's HEAD_SHA for its own process only; original
evaluator disk bytes and ongoing original-head evaluations remain intact.
This deliberate head rebinding requires its own review and new TRAIN smoke.
Report all three future replication outcomes, without TEST-based model or
epoch selection, threshold changes or independent-final claims.

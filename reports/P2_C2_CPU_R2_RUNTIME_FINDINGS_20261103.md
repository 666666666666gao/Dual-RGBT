# C2 CPU R2 Runtime

The actual isolated CPU run on P25 exited zero and produced a PASS receipt.
Result SHA256: `0c31fedbd22b43f34c9f9311f9b17ffaf3952ec7d5b78370fe8fff82261e4a64`.
Exact sources, invocation, source review, log and receipt are archived in
`reports/c2_executor_cpu_r2_20261103/`.

All four variants matched the original executor's nested tensors on both
modalities and all five actions at initialization, including `predict`.
Parameter counts and unique, complete optimizer coverage passed.
The routed variant then performed two actual CPU optimizer updates:

| Update | TRAIN Loss | Gradient Norm |
| --- | --- | --- |
| 1 | 0.7030457854 | 14.9956617355 |
| 2 | 0.6193130612 | 14.7255268097 |

Active route banks received finite nonzero B gradients; the inactive local
bank had zero current gradient. Missing embeddings and the active FiLM
received gradients. Frozen policy tensors remained equal. Post-update strict
state reload and output equality passed. CUDA remained uninitialized.

The R2 diagnostic disables DropPath only for this single-pair connectivity
check. The original training recipe retains stochastic depth. Failed R1
evidence remains intact. Two diagnostic updates are not a convergence result,
a serialized C2 training checkpoint, or a benchmark improvement.

The new local training entry and eight disabled TRAIN smoke drafts still need
their own exact review and GPU execution. Its full-timeline causal sampler
requires a fresh matched zero/no-LoRA control. All four variants use the same
recipe and data sampling, preserving the ability to separate sampling effects
from module effects. Existing frozen-executor P1B FIT continues unchanged.

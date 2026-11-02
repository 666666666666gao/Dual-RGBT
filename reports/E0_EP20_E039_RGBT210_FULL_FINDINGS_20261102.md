# E0 Epoch39 RGBT210 Full Result

The requested ep20-initialized E0 executor fine-tune completed40epochs.
Its fixed epoch39 RGBT210 evaluation now completed all210sequences and
104,706frames on26GPU1, with native COMPLETE/exit0 at2026-11-02.
The unchanged official scorer ran once, followed by complete trace-weighted
compute aggregation. These are complete exploratory TEST results,
test_eligible=false; no checkpoint, method, seed or threshold was selected.

| Fixed Checkpoint | PR | SR | Full-Run FPS | Parameters | Mean Tracked GMAC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Epoch39 | 89.018952 | 65.402349 | 10.606679 | 103,374,215 | 53.613616 |
| Epoch40, already reported | 88.368511 | 65.080225 | 11.686902 | 103,374,215 | 53.697104 |

Epoch39 descriptively exceeds the requested RGBT210 PR/SR88.7/62.9
by0.318952/2.502349points. This does not establish the LasHeR or RGBT234
goals, three-module innovation, or an independent final validation.
All fixed lastfive evaluations continue and all checkpoint dependencies
remain protected. The TEST result does not change the TRAIN recipe.

Timing includes decoding and sequence initialization under concurrent
GPU/system load; mean94.280218ms/P95142.757454ms. FPS differences between
checkpoints cannot be interpreted as isolated speed improvements.

MACs use the actual epoch39 full trace route/token counts and the existing
matching-source, fixed-shape CPU-executed arithmetic profile
`E0_EP20_E040_P26_ROUTE_MACS_20261102.json`. That profile was executed with
epoch40weights; the shared architecture and route shapes determine its
arithmetic counts. No new epoch39 profiling inference was performed.
104,496trackedframes exclude210initialization frames. Standard matrix/
convolution/attention MACs exclude initialization arithmetic, nonlinearities,
normalization, ROI/decode/memory operations. One MAC equals two profiler FLOPs.

Artifacts under `reports/e0_ep20_p26_evaluation_r1_20261102/`:

- Score `eval_e0_ep20_e039_rgbt210_r1_p26_20261102.json`,
  SHA256 `464e80ef9dc330f3ee9640418cb55b53ace861dc83f87d715d8dc0efa77b0568`.
- Curves `.npz`,
  SHA256 `6b0f3bdbcb7093eb36041da4b36fffc1b2b7413754213fb12df236805ee035ac`.
- Compute `_compute.json`,
  SHA256 `1d67b2fe4940f4a2a7a97960423d24a50b20e5fe1e806afebfcdaa7cb85d9c8e`.
- Native status in `runtime/eval_e0_ep20_e039_rgbt210_r1_p26_20261102/`,
  SHA256 `1edbdf338c13208683647e35e59f2fe3d04b8bcffa1254fcc2b483ca06cb1694`.
- COMPLETE receipt SHA256
  `92734a82ac2e64d9025d198b5393ac95c07f0febc8f065392e67985a4c5b8b60`.

Exact score/curves/compute/config/receipt/native bytes were fetched and
their endpoint hashes checked. The original epoch39 checkpoint remains
SHA256 `4405b608e961ff3bd07b6cf1c26c6201e1f6b82b7d429f9f1d9d53b99707d3ae`.

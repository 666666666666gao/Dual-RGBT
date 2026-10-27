# CADTrack baseline latency result audit — 2026-10-27

**PASS — zero blocking issues.** Same-family/provisional; requested gpt-6-astra/max, actual model and effort UNATTESTED. Local raw-file audit only; no SSH, model execution or weight load.

The600 CSV rows independently reproduce the reported results:

| Timing scope | FPS | Mean latency | P95 latency |
| --- | ---: | ---: | ---: |
| Service, excluding image decode |49.901808|20.039354 ms|26.821299 ms|
| End-to-end, including RGB/TIR decode |20.878209|47.896828 ms|68.888811 ms|

All three prespecified TRAIN sequences contain exactly200 unique measured frames21–220. The hash-bound executed loop and metadata establish20 warm frames per sequence. Every duration is finite and positive; decode+service equals end-to-end for every row with zero numeric residual. FPS was recalculated as600/summed seconds; P95 uses linear percentile interpolation. Initializations are separate.

The observed total is **103,138,437 parameters (103.138437 M)**. Native status is COMPLETE/exit0; its envelope, COMPLETE file and final run-log JSON agree. The log records the original SOTep14 load, and the reviewed executed source checks checkpoint SHA `cbcc96ba3a0cf1bfb997ee164174c2264cabf17ad013f8249b7f31a43815d731`, strictly loads original CADTrack, counts all parameters and executes CUDA-synchronized online tracking. Plan SHA `d0b019ea9b948d7c84998dca652a79832827cfebe94a44d3b4b6c04a5b694a37` and all112 local source hashes match. No checkpoint payload was reopened in this audit.

Recorded hardware/configuration: RTX3090 GPU0, FP32 with matmul/cuDNN TF32 disabled,256 search and two128 paired templates. Service timing includes crop, preprocessing, forward/head and template update. No accuracy or trainable-parameter evaluation was performed.

Both own GPU0 training groups resumed after45.250452 seconds, with wrapper statesS and child statesR. Exact identities persist in the post-resume receipt; updates advanced3000→3300 and2975→3275, both RUNNING. Last loss fields are independently finite, and the native receipt reports all logged losses finite. Other GPU/CPU jobs continued. The initial import failure remains archived as FAILED with no FPS.

The supported result is a **600-frame TRAIN timing sample under the recorded host load**. It is not full LasHeR test FPS or evidence of a paired speedup over Ours. The legacy head's deprecation warning does not prevent the successful exit; no source change was made.

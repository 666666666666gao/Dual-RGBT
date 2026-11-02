# E0 Epoch40 Full LasHeR Result

The fixed epoch40 checkpoint completed all 245 LasHeR TEST sequences and
220703 frames with native exit0. The official scorer ran once, followed by
the existing compute summarizer. Exact artifacts and native/config/receipt
bytes are archived under `e0_ep20_p26_evaluation_r1_20261102`.

| Metric | Result | Target | Result minus target |
|---|---:|---:|---:|
|PR|75.5873846215|75.8|-.2126153785|
|NPR|71.6144248298|71.9|-.2855751702|
|SR|60.0960017977|60.2|-.1039982023|

Checkpoint SHA256:
`eb63d329d382df1e8fc5bc69e73c41a11d1c263bd4ad7f4a8af7db1107b3026f`.
Score SHA256:
`c9be4623bc71549d61a5080d93d338c3fb219bc75a5b8fc7bd2734c51f92a127`.
Compute SHA256:
`22666c7dd209e8d47c546752e88bc64b224d1450c0cbe2faf945376d23a2381a`.

Parameters: 103374215. Full-run FPS: 7.7998765858 under concurrent load,
including initialization. Trace-weighted mean tracked-frame GMAC:
54.3548102732, excluding initialization arithmetic and other operations
listed in the compute artifact. The matching source profile uses an earlier
TRAIN smoke checkpoint, not a separate epoch40 profiling forward.

This is exploratory TEST reporting (`test_eligible=false`), not checkpoint
selection or independent final validation. The LasHeR targets are still
unmet. Other fixed epochs continue to their complete results.

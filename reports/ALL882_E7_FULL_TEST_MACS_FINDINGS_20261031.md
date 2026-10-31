# E7 Full-Test MAC Accounting

## Result

- Actual CPU execution on server26 completed against the full E7 exploratory
  LasHeR test trajectory: 245 sequences, 220703 frames, 220458 tracking frames.
- Parameters: 103218887.
- Estimated convolution/matrix MACs per tracking frame: 41429530772.30043
  (41.429531 GMAC).
- Estimated per-frame MACs including reused parent initialization measurements:
  41383540300.7653 (41.383540 GMAC).
- These are analytic shape-weighted estimates, not fresh GPU measurements.
  Existing concurrent FPS 8.615936 is not an isolated throughput benchmark.
  Official E7 PR/NPR/SR remain 67.543368/62.650040/51.093086.

## Evidence And Limits

The old 40.240538 GMAC report describes the parent DEV route mixture, not the
E7 full-test mixture. E7 contains 59 route shapes; 23 have existing parent
measurements, while 36 require analytic shape estimates and cover 3356 frames.
Neither the 36 shapes nor the other full-test shapes were newly GPU-profiled.

The estimator verifies all 37 Python core sources plus the architecture YAML
against both reports and actual server26 files. It checks the scored/native
completion, configuration, progress, curves and all 245 trace hashes and exact
frame indices. Gain-overlay weight changes do not change the verified core
shape accounting. Each global candidate adds two 955->64->1 head evaluations:
122368 linear MACs. Applying this increment to action-at-K8 measurements
reproduces all 23 original profiled cases exactly.

MACs cover profiler-supported convolution/matrix operations only. Normalization,
softmax, sampling, decoding, pointwise arithmetic and other unsupported
operations are excluded. Initialization reuses the parent profile.
The formula is not proof that every unmeasured shape has been independently
profiled. No scientific acceptance or accuracy improvement is claimed.

Five focused CPU tests pass locally and on server26, including actual head
dimensions, exact measured-case reproduction, missing-shape labeling and
rejection of inconsistent measurements or a changed head. Initial execution
found a missing copied fixture, a summary-derived source-count mistake and
an incorrect assumption that per_sequence metrics contain frame counts.
The actual progress.json now supplies checked sequence lengths. These attempts
failed before output creation; no fallback or model change was introduced.

Result: `reports/ALL882_E7_FULL_TEST_MACS_ESTIMATE_20261031.json`

SHA256: `ee54d8dab92d37ab44d1b4d16920ec4591f60964b240e2ee3460a424f1bb7f24`

Estimator SHA256:
`b5220e55a248e7ab41d1b2d9e5b0c87b7c3ffbfe300c4c581f3088ee2b567db0`

## Mainline Status At 02:20 UTC

All 17 inference children on eight GPUs are live with trace writes within
10.7 seconds. No new full score, crash or idle GPU was observed. Placement
remains server25 1/2/2/1 and server26 3/3/3/2. Matched-parent CPU scorer3756936
is also live. Do not restart or duplicate these runs.

An independent 451-second observation interval suggests server25 refreshedE12
needs approximately 53 more minutes, its utility jobs 69-104 minutes;
server26 jobs approximately 230-369 minutes. These are buffered concurrent
progress estimates, not formal FPS or guaranteed completion times.
E12 has 175 completed sequences and 181525 observed frames; matched parent
has 63 completed sequences and 53490 observed frames. A long current sequence
is still actively writing.

Protected-E0 full TRAIN collection remains unlaunched: the new collector has
no fresh independent source review, and the old observer review does not
cover it. No review-chat message was authorized or sent. No new dev,
gain-only grid, size-tail expansion or speculative weight deletion occurred.
The active target and higher pasted target remain distinct and unmet.

# P1B SOT RGBT210 Full Score

- Fixed policy epoch005/cost0.05; full210 sequences/104706 frames.
  Native COMPLETE/exit0, exact58-source/checkpoint/review/config/precision/
  GT/trace coverage checks passed before official CPU scoring ONCE.
- PR90.00212504165442/SR65.52158802216051. Fixed pre-P1B P1A was
  PR87.8488497279/SR64.2868049169: gains approximately2.1532753138/
  1.2347831053 percentage points. Corresponding published CADTrack-SOT
  88.7/62.9 is exceeded on this benchmark by1.3021250417/2.6215880222
  points. LasHeR remains running; this does not establish overall goal success.
- Raw archive: reports/p1b_rgbt_completed_scores_r1_20261104/sot/rgbt210/.
  audit.json binds eight raw SHA256 values. Never repeat scorer/archive.
- FPS12.171477019684513/mean82.15929737884188ms/
  P95111.95975200098474ms/103374215 total parameters. Decode/init included,
  model loading excluded; concurrent load, not isolated throughput.
- TEST exploratory_test=true/test_eligible=false. No training, tuning,
  threshold, checkpoint, epoch or branch selection from these results.
- Completion releases GPU1 for the separately reviewed SOT234R2 plan.
  New TRAIN smoke and fresh runtime admission precede its full evaluation.

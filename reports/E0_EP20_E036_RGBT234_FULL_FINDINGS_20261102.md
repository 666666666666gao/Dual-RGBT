# Fixed epoch36 full RGBT234 result

P25 epoch36 completed with native exit0,234sequences116650frames.
Existing CPU controller ran the official scorer once and compute passed.
MPR90.28648504355246/MSR67.40897985197704, FPS15.970845228879156,
mean62.61409372321497ms/P95 84.81606699992881ms,
103374215parameters/mean tracked53.411774789453546GMAC.
MPR is0.6135149564 below target90.9; MSR is1.8089798520 above65.6.

Score SHA256:
fca832429371b961e1e282abab4c5357bd261dcfb100a0234e9df46111c0f8c9

Compute SHA256:
76e52de5044f587190356ba7c85c075bd5eba70245889e94a6191e74217ec163

Exact raw score/curves/compute/native/config/receipt archived under
e0_ep20_p25_full_scores_r1_20261102. TEST exploratory=true,
test_eligible=false; no checkpoint promotion, tuning or independent
final-validation claim. No fine-tuned LasHeR full score yet.

FPS reflects concurrent load. The49-source profile708d3aaafb486a5f21a28bf134207a3ae2bf59e0c61607336e9a6addd4a8989e
used epoch1 TRAIN smoke weights with matching shapes/routes; compute
uses actual epoch36 route counts/initMACexcluded, not a new per-epoch
profiling forward.

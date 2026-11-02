# Fixed epoch36 full RGBT210 result

P25 epoch36 completed with native exit0,210sequences104706frames.
The existing CPU controller scored once and produced
COMPLETE_FULL_RUN_COMPUTE_SUMMARY. Exact score/curves/compute/native/
config/receipt bytes are archived in e0_ep20_p25_full_scores_r1_20261102.

PR89.07139359182142, SR65.35587917512396, FPS14.803747473924732,
mean67.55046327028994ms, P95 93.85338175013658ms,
103374215parameters, mean tracked-frame53.63411786461859GMAC.
Descriptive target differences are +0.3713935918PR/+2.4558791751SR.

Score SHA256:
e800e2a4bd67896ee1ddb87ec9a3a1349ce8181931020b9183ebffac8d8eebe2

Compute SHA256:
ba0c067d7350fbfb1cf27e26ca68480fb7e1591e560198360b89fc52d553f12c

This fixed TEST result is exploratory_test=true/test_eligible=false,
not checkpoint selection, recipe tuning or independent final validation.
No fine-tuned LasHeR full score was available at this check.

Concurrent FPS is not an isolated speed comparison. Matching49-source
profile708d3aaafb486a5f21a28bf134207a3ae2bf59e0c61607336e9a6addd4a8989e
used epoch1 TRAIN smoke weights and matching model shapes/routes.
Compute uses actual epoch36 route counts and excludes initialization
MACs; no epoch36 profile forward pass is claimed.

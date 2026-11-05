# P25 DropMoE-FiLM Full TRAIN Replay Complete

At 2026-11-05 02:56:12.599 UTC, native task
`verify_c2_policy_dropmoe_film_full_r3_p25_20261104` reached COMPLETE/exit0
on P25 GPU2. The acceptance is PASS for all 882 TRAIN sequences,
458,553 tracked frames and 91,091 records. The task used the four reviewed
collection plans and the original P25 source map.

The one-time terminal evidence is in
`reports/c2_policy_replay_completed_r1_20261104/dropmoe/film/`. Its
acceptance, native status, observation and run log match the four entries
in `SHA256SUMS`. The acceptance explicitly sets
`benchmark_evidence=false` and `policy_fit_admitted=false`. This is a
collection/replay milestone only, not a FIT, benchmark result or target
success. Do not rerun the replay or archive it again.

P25 GPUs were released after completion. A separate exact FIT plan/source
review and fresh canonical runtime admission are still required before any
FIT launch. Keep the fixed recipe and existing artifacts; do not use dev or
TEST for selection. Full TEST remains exploratory and ineligible.

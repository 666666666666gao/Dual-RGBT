# DropMoE Full P1B Collection Accepted

All four original P25 full TRAIN collectors completed natively with exit0.
Final s2 contains 220 sequences, 128,483 tracked frames and 25,541 records;
s3 contains 220 sequences, 127,696 tracked frames and 25,398 records.
Both terminal shards' exact raw/native/count/precision/cache bytes have
been archived under `reports/p1b_completed_shards_r1_20261102/dropmoe/`.
The previously completed s0/s1 evidence remains unchanged.

The original canonical CPU verifier then accepted all 882 sequences,
458,553 tracked frames and 91,091 records, validating receipts, cache
contents, ordered frame indices, original source/input bindings and native
ownership. CUDA remained uninitialized. Exact acceptance SHA256:
`c11a6d7fada115424f47d5ac62b5d9f5495102edefde61da70a5028c91628df8`.
Raw acceptance, four native statuses and verifier output are archived in
`reports/p1b_full_fit_r1_20261102/dropmoe/`.
No verifier rerun is needed.

The existing local preparer generated the disabled concrete plan:
task `fit_p1b_dropmoe_full_b512_r1_p25_20261102`, GPU3, batch512,
four workers, five epochs/890 updates, previewLR4e-5,
quality/gainLR4e-4, AdamW weight_decay1e-4, clipping1.
The executor stays frozen; all five policy checkpoints must be retained and
strictly reloaded. Plan SHA256:
`b9d31099f4fc997dfea6f7aa48300442b42b2a9665c93e70d5fa1842947fd31f`;
canonical binding:
`179ab290bbeccd3139dc43bb4016ab3965828e407730a8e550c2acfa1b05448c`.

The authorized reviewer returned the separate concrete report:
`fit_p1b_dropmoe_full_b512_r1_p25_20261102_ADMISSION.json`,
stage P1B_direct_E0/SOURCE_PASS/zero blockers, exact original98/supplemental
trainer maps and admitted binding. Report SHA256:
`d4ed39cdd91fadb2a406e8bd05a268e01d9737271cf18d33f84fb194ac78e274`.
At2026-11-03 05:33:20 UTC, `work/launch_p1b_full_fit.py` passed the report,
source/input/acceptance/native checks and canonical CPU admission, with
24,247MiB free on GPU3 and sufficient output-disk capacity. It deployed
only the absent admitted plan/report/supplemental trainer and started
wrapper3820939. Actual training child3820953 was live and owned a GPU3
context at05:34UTC. Config and effective TF32-off precision were saved.
This is actual native FIT launch evidence, not a completed optimizer-step,
epoch, convergence or benchmark-improvement claim.

P26 s0/s1 are COMPLETE0 and archived; s2/s3 were still live at 201/211
sequences at the 05:18 UTC check (854/882 total). Six fixed epoch20
complete-TEST controls were still live at the last inspection. These controls
are exploratory/test-ineligible and do not select or alter this FIT recipe.

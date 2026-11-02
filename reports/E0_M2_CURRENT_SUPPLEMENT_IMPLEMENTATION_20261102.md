# Current E0 TRAIN Evidence Supplement

## Observed problem and purpose

The accepted original M2 collection contains 882 sequences and 18,711
candidate sets. Its 24 candidate descriptors and fixed anchors are
restored on host26 with original acceptance hashes. It lacks the actual
current E0 proposal's paired descriptors and the matching 914-D M1
observation. Candidate0 and the learned scalar no-match do not provide
those missing observations.

The completed frozen TRAIN replay found 326 reliable states crossing below
IoU0.5 among 2,899 component-held reliable states; only179 of800 lost
states recovered, with a positive candidate available in374. The
supplement supplies explicit current E0 evidence for a future relative
KEEP/candidate scorer. It does not rerun24 candidate queries or change the
original collection. These are TRAIN diagnostics, not online benchmark gains.

## Implemented smoke and review defects

The runner uses original build_model-before-build_tracker crop binding,
one observer on one shared frozen model, two paired trackers and fixed
initial templates. It compares every prediction with accepted original
cache predictions and checks paired routes and tracker states. Only sampled
frames receive two extra encodes and ROI pooling; there is no box-head
rerun, GT-derived decision, parameter update or online intervention.

R1 review:
`reports/E0_M2_CURRENT_SUPPLEMENT_SMOKE_SOURCE_REVIEW_R1_20261102.json`
is SOURCE_BLOCKED with three actual defects:

1. Coverage was only range-checked. A central ROI with fabricated zero or
   fractional coverage could pass. R2 reconstructs the exact float32
   3x3 grid, checks both modalities and requires zero descriptors at zero
   coverage. Partial observations retain their actual pooled descriptors.
2. Verification accepted an unrelated output directory holding copied
   artifacts. R2 binds its canonical directory to the immutable plan.run.
3. Native startup omitted task and hostname checks; postverification
   ignored hostname. R2 binds native task, actual hostname, audited plan
   hostname, absolute interpreter/script argv, exact plan argument, GPU
   and state, retaining startup parent-PID and environment checks.

Seven exact R1 sources were copied and hash-verified under
`reports/source_snapshots/e0_current_supplement_smoke_r1_20261102/`
before working sources changed. Original R1 draft/audit/review remain
available. They do not admit R2 execution.

## R2 evidence and boundary

Actual host audit:
`reports/E0_M2_CURRENT_SUPPLEMENT_HOST_AUDIT_R2_20261102.json`.
All127 original sources, five inputs and three selected original caches
match their previous hashes; all186 image paths for93 paired frames exist.
Actual hostname: ubuntu-WS-C621E-SAGE-Series. New R2 run/task paths were
absent. The three locally dirty dependencies are reviewed using their
exact original host26 snapshots, without replacing either working copy.

Disabled draft:
`reports/e0_current_supplement_smoke_r2_20261102/draft.json`,
SHA256 `7c4fdf7dfbe31fbb758fdb4f11466f510b99580beaca0c68af34d621ab44ec73`.
It binds134 sources, five inputs, three parent caches, GPU1 and
boysumbrella/boyshorts/whiteboyback,31 frames each.

`E:/anaconda/python.exe -B tests/test_e0_current_supplement.py`
passed seven CPU tests. Coverage was compared with actual CPU
roi_descriptor arithmetic on105 random/boundary boxes. Negative cases
cover impossible coverage, nonzero uncovered descriptors, irregular frame
joins, wrong native identities/argv/GPU/state and wrong output directory.
An earlier independent500-box arithmetic comparison also matched exactly.
These tests are synthetic checks, not actual GPU smoke or training.

Fresh R2 review was requested from the already authorized reviewer chat.
It must bind the R2 source map and draft exactly. Review attribution remains
UNATTESTED, same-family, provisional, scientific_result_audit_accepted=false.
No R2 file has been deployed and no supplement GPU task has run as of this
implementation record. SOURCE_PASS is still required before deployment.

After review, deploy only the seven new files plus review/draft/final plan,
verify remote hashes and run the same CPU tests/admission. Use the exact
absolute native command on GPU1 after a fresh capacity check. Only actual
COMPLETE/exit0 plus the independent postverifier admits this93-frame smoke.
Full supplement and real relative training require their own scoped plans
and acceptance; smoke alone does not grant them.

## Related local work and existing jobs

`dual_rgbt/e0_relative_identity.py` and its five previously passing CPU tests
implement explicit current E0 versus candidates, KEEP on exact ties,
KEEP-only when current coverage is unavailable, and TRAIN-only float64
relative targets. They are outside the134-source smoke review and have
not been deployed, integrated, trained or promoted.

The actual host26 progress audit
`reports/E0_CURRENT_SUPPLEMENT_P26_PROGRESS_20261102.json` confirms all four
existing native tasks and their actual children live with fresh traces:
epoch40 LasHeR105 completed sequences/97,426 frames; epoch39 LasHeR25/18,295;
epoch39 RGBT21069/25,980; original E0+M2 LasHeR210/205,914.
No task is terminal, so no scorer or next evaluation was started.
Existing fixed evaluation queue max3 onGPU1 remains unchanged.

Only host26 GPU0/1 are available to this work. GPU2/3 belong to other
workloads. Host25 GPU0 still has Xid79/fallen-off-bus/all-ff PCI access;
other three cards query individually. Root is100% with6.5G free; physical
cause and disk causality are unproven. No reset, reboot, kill, speculative
weight deletion or new host25 workload was performed.

Goal remains unmet. All TEST scores remain exploratory_test=true and
test_eligible=false, with no TEST-based training or selection.

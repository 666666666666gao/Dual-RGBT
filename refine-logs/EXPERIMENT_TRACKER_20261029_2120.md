# Experiment execution tracker

| Run | Evidence | Status |
|---|---|---|
| B8/B64 probes | Four updates, two changed tensors, strict reload | SANITY_PASS |
| B384 probe | Allocated10.14/reserved12.66GiB,363.02s | SANITY_PASS |
| B576 probe | Allocated15.01/reserved18.77GiB,409.65s | SANITY_PASS |
| Parent11-frame control | Prediction/action/modality equality | COMPLETE/exit0 |
| Parent full-dev control | GPU1 wrapper1100268,scorer1100867 | RUNNING;30seq complete at01:03UTC |
| B384 full-dev diagnostic | GPU2 wrapper1106853,scorer1107101 | RUNNING;23seq complete at01:03UTC |
| B576 full-dev diagnostic | GPU0 wrapper1126673,scorer1128119 | RUNNING;1seq complete at01:03UTC |
| Resumable B8 epochs1/2 | Native exits0,optimizer steps4 then8 | Update/reload PASS |
| Resume numerical witness |265network tensors and optimizer load exact; small final moment differences also in repeat control | AUDIT RECORDED |
| Epoch controller |6Linux synthetic workflow tests | PASS; no accuracy claim |
| Formal pilot | B576/LR7.2e-5,9216pairs per epoch,up to10 | NOT ADMITTED; full-dev review pending |
| Last-five dev selection/cleanup | Implemented and synthetic workflow tested | Real formal execution pending |

At01:03UTC all eight recovery GPU slots and both queue/scorer PIDs were alive.
26:3completed/4active/5queued;25:0completed/4active/8queued. Recovery gainpair
cost0.15 scored71.9562/66.9082/53.0161(PR/NPR/SR), below cost0.05 on all three.
No newly adapted full-dev accuracy result is available yet.

## Update at2026-10-30 01:17UTC

| Run | Evidence | Status |
|---|---|---|
| Parent failure diagnostic |97artifacts checked;1215bad writes of7608 reconstructed | COMPLETE; correlation only |
| Diagnostic regression | Episode boundaries and query counterfactual distinction |2tests PASS |
| Static-template31-frame smoke |6mutations reverted;265frozen tensors;native exit0 | COMPLETE |
| Static-template full97dev |25GPU3 wrapper1155267;scorer1155571 | RUNNING; no full metrics yet |
| Parent/B384/B576 full97dev |41/34/30complete sequences at01:16:30UTC | RUNNING |
| Recovery queues |25:0done/4active/8queued;26:3done/4active/5queued |0failures;180s automatic refill |

All8cards have active evaluations. Additional25evaluations now occupy each
of its4cards alongside recovery work. Concurrent aggregate speed improvement
is unmeasured; keep formal size-tail pilot blocked on the existing score review.

## Update at2026-10-30 01:20UTC

Training-label audit COMPLETE:182182updates/379571labels with native/config
provenance, joins, counts and stable input hashes verified. Chosen-modality
sampled recoveries1393,including1069onSTOP; global random queries also highly
destructive. Two regression tests PASS. No final-parent dev recall claim.
At01:19:18all12evaluation wrapper PIDs are alive; parent/B384/B576/static
completed43/41/30/1sequences. Recovery queues remain25:0/4/8and26:3/4/5
(completed/active/queued),0failed. Full-score gates remain pending.

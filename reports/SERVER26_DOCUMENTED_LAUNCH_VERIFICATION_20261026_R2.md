# Server26 documented launch verification — R2 PASS

The corrected HANDOFF invocation ran verbatim once at **2026-10-26 11:21:44 UTC**, without further edits. Its original native **exec session23028 remains active**. The first failed invocation and diagnostic evidence are preserved separately in [R1](SERVER26_DOCUMENTED_LAUNCH_VERIFICATION_20261026_R1.json).

| Actual handle | Identity |
|---|---|
| PowerShell / Python launcher / controller | **20328 → 41216 → 30652** |
| Queue controller status PID | **30652**, exact CIM command and parent chain matched |
| Completed GPU0 smoke | run_task **433739** → observed child **433751**, native **COMPLETE / exit0** |
| Active first full dev, cost0.0 | GPU0 run_task **439618** → child **439647** |
| Next existing queue node | **2026-10-26 11:54:50.239 UTC / 19:54:50.239 Beijing** |

At the single scheduled observation after the 180-second smoke node, the native receipt confirmed **11 real train-split frames, strict reload PASS**, including bit-equal box/route and templates/history. It uses the original final checkpoint SHA `527a3bc0…28753`, **103,218,887 total parameters**, and an RTX3090. This is fixed-weight tracker inference: **0 backward calls / 0 optimizer updates**. The historical C882 native-exit warning is retained.

The same controller then started cost0.0 full97dev. Actual `/proc` commands, PPid, project cwd and child `CUDA_VISIBLE_DEVICES=0` matched the native task and local queue state. Cost0.05 and cost0.15 have not started in this observation. The one-NN-at-a-time queue remains responsible for all three complete dev runs and scoring; it does not launch test.

No complete-dev accuracy or formal full-dev FPS is available. Raw smoke timing is retained only as smoke evidence and must not be quoted as the formal online performance result.

R1 failed because the hidden PS5 child could not resolve `Get-FileHash`. The only correction was the parent-authored documentation change to invoke the same reviewed scheduler directly from existing PowerShell7. Source, plans and weights were unchanged; this verifier performed no extra correction or spontaneous retry.

Next: allow the existing controller to reach its 30-minute first-dev node. Do not terminate original session23028 or create a second controller. This verification does not wait for full dev.

Evidence: [JSON report](SERVER26_DOCUMENTED_LAUNCH_VERIFICATION_20261026_R2.json) and [recorded process/receipt snapshot](../.aris/traces/run-experiment/2026-10-26_server26_documented_launch_R2/remote_smoke_node.json). Fresh same-family verification; actual model/effort route remains UNATTESTED.

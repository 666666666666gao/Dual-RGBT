# Server26 documented launch verification — FAIL

The documented command was executed verbatim once at 2026-10-26 11:16:16 UTC. It returned hidden launcher PID **30616**, but no controller or remote neural task started. No retry, source change, or plan change was made.

- Preconditions passed: retained runtime **9,367 files / 3,337,088,228 bytes**, both SOURCE_PASS reviews, exact plan hashes, all 48/50 local source hashes. GPU0/1 used **26/63 MiB**, below the 500 MiB admission threshold.
- Windows PowerShell events identify the exact documented `-File` command and engine 5.1.26100.9444: Available at **11:16:17.092065 UTC**, Stopped at **11:16:17.499352 UTC**.
- At **11:16:27.401698 UTC**, CIM found neither PID30616 nor its children. Subsequent CIM found no matching scheduler/controller. The queue status, scheduler log directory, and native exit receipt were absent. The caller's exit0 is **not** the child launcher's exit code; the latter remains unknown.
- At **2026-10-26T11:18:25.116Z**, SSH2026 process inspection found no matching run_task/run_rankstop_dev process. All four current smoke/dev native status files were absent.
- A read-only reproduction using the same hidden PS5 Start-Process route, with only `Get-FileHash` on the plan, exited **1** (diagnostic PID18008). Its stderr reports **CommandNotFoundException: Get-FileHash**. Original launch stderr was not captured, so the exact original failing line cannot be proven; this reproduced missing command is a concrete blocker consistent with the early failure.
- `C:/Program Files/PowerShell/7/pwsh.exe` is absent. The current installed PowerShell7 executable is `C:/Program Files/WindowsApps/Microsoft.PowerShell_7.6.9.0_x64__8wekyb3d8bbwe/pwsh.exe`.
- An earlier EncodedCommand read-only diagnostic (PID45176) exited -1 with empty streams; it ran no queue and does not establish the failure cause.

**Result:** the documentation does not work without correction in this launch environment. Full tracker smoke/strict reload was not executed. No full dev metrics, measured full-dev FPS, or test results are claimed. The prior small seeded kernel witness remains separate.

There is **no running queue handle or next observation node**. Parent must correct the documented launcher environment before a separately controlled fresh attempt. This verifier has stopped at diagnosis and did not retry.

The complete command, process/event evidence, read-only native receipts, and unchanged plan/source checks are retained under [.aris/traces/run-experiment/2026-10-26_server26_documented_launch](../.aris/traces/run-experiment/2026-10-26_server26_documented_launch/launch_invocation.json). Machine-readable report: [SERVER26_DOCUMENTED_LAUNCH_VERIFICATION_20261026.json](SERVER26_DOCUMENTED_LAUNCH_VERIFICATION_20261026.json).

Verification is from a fresh same-family agent; actual model/effort routing is UNATTESTED.


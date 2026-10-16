# Scheduled adoption R1 source review

**SOURCE_FAIL — 1 blocking issue.** This is a source/admission review, not deployment or current-liveness proof.

- Plan SHA256: `6f0fc2fb656d03ba50d60f4d81d118862e254aae5ce9b9ec0330fb8d1eb6511c`.
- Exact 49-entry source map and 14-entry local input map are included in the JSON review.
- Requested routing: `gpt-6-astra/max`. Actual model and effort: `UNATTESTED`; same-family, provisional.

## Blocking finding

**ADOPT-R1-F01 — missing `time` import.** `scripts/resume_candidate_benchmarks.py:111` calls `time.time()`, but imports at lines2-11 omit `time`. The original controller imports it at `scripts/complete_candidate_benchmarks.py:8`. Imported helper functions keep the original module globals; the copied nested `finish()` resolves globals in the new module.

An isolated CPU evaluation of the exact line111 expression raised `NameError: name 'time' is not defined`. After training completes, lines121-123 may launch the first dev pair, then line125 fails while entering `finish()`. Waiting, collection, scoring, freezing and all three full benchmarks would not complete. Add the missing import to the new controller and review a newly bound plan/source revision; no training rerun is required.

## Scope and evidence

- The saved second diagnostic records 26,100 updates; the third records 46,300 of 182,182, with 486 sequences having logged updates. The third saved native/proc evidence still identifies task `train_global_candidate_trajectory_formal_20261015`, PID827849, physical GPU1, exact argv and config SHA `f7def88cb821de9ad0614d228376a946c04071be340461a460c538e5ab34a5da`.
- The three saved local states are byte-identical between those observations. Earlier CIM receipts match worker PIDs33196/41568/42732, their creation times, exact commands and distinct launcher PIDs29556/43612/18212. The third saved CIM stdout is an empty array. These files establish the dated observation; current liveness and loss cause were not queried.
- `scripts/resume_candidate_benchmarks.py:37` binds native task/PID/GPU/command, current `/proc` argv, config SHA, training plan/source/input maps, evaluation source/input hashes and the original CADTrack/SOT checkpoint. The adoption prefix contains no training launcher, model load, optimizer reset or NN call.
- The suffix beginning `    trained = remote(` is byte-exact with the original through EOF: 5,766 bytes, SHA `8040ce4b28b2b69146a6ba9694f783a94033782d041cae9084ba0a5d4901d63d`. Completion integrity, 97-sequence dev costs0/0.05/0.15, freeze-before-test and full245/210/234 protocols remain unchanged. The copied suffix still requires its module-level time dependency.
- Original64 and scale68 source maps match actual local bytes. All49 adoption sources and14 local inputs match the plan. The prior NN M0 audit remains `M0_RESULT_AUDIT_WARN`, SHA `fe433658b5a64488a94ca064e7768377133bcce7f0de4f17c5add292cb221921`; no new M0 was performed.
- Both NUL-parser layers produce delimiter codepoint0 and the exact15-part wrapper argv. Windows replacements use one backslash and produce forward-slash SCP paths. RUNNING native records have no exit_code; terminal code is required only after completion.
- New status/private/metric/log paths are distinct and unconsumed; directory/task guards reject existing destinations. `wait_owned`, `collect`, `checked_cpu` and `save` remain the original helpers. The first observation delay changes from7200 to14400 seconds only.

## Downstream and scheduler source

`scripts/complete_reference_benchmarks_adopted.py:58` and `scripts/complete_candidate_latency_adopted.py:117` contain the sole substitutions from their original files: the exact prior controller entry name. Reference still requires full E1 and every owned native exit0 before GPU work; latency still requires full E0+E1 and every native exit0. No baseline accuracy rerun is added. Future downstream plan/PID bindings have not been admitted.

`scripts/run_scheduled_controller.ps1:9` and the registration helper at lines9-28 require the reviewed plan, keep duplicate guards, request current-user Interactive/Limited execution, hide both windows, omit triggers/retry parameters, and request zero execution-time limit. The wrapper waits for its actual launcher exit and writes `launcher_pid` and `exit_code`; the controller state separately writes its true `os.getpid()`. No global policy/network/package/credential modification or automatic NN restart is present. Actual OS registration, XML defaults, worker ancestry, task state and terminal launcher receipt remain unexecuted.

The inherited plan `scope` text at `reports/candidate_completion_adopted_plan_20261016.json:84` still describes waiting for the old owner and starting C training. This nonblocking documentation inconsistency should be corrected in the next revision to describe adoption of PID827849.

## Actual CPU receipts

| Receipt | Actual exit | Result |
|---|---:|---|
| probe_01 | 0 | NUL, Windows separators and import/name inspection completed. |
| source_01 | 0 | 68 checks completed:67 pass,1 real source failure. 73 Python files parsed with3.7 grammar;125 sealed inputs unchanged. |
| powershell_01 | 1 | Preserved harness failure: Get-FileHash did not resolve under inherited PowerShell7-first module paths. |
| powershell_02 | 0 | Checker-only .NET I/O/hash serialization;2 PowerShell ASTs and8 static constraints pass. No scheduler command executed. |

Every listed checker process was joined before this review. The first PowerShell failure is retained with its exact script, stdout/stderr and transport. A separate executor receipt `e65324` (exit0, decoded tool stdout) confirms that the normal WindowsPowerShell5.1 environment resolves Get-FileHash, ConvertFrom-Json and Get-ScheduledTask; this does not prove a scheduled deployment. No production fallback or environment modification was made.

All private scripts, output, raw checker stdout/stderr, joined terminal receipts, R1 control-file copies and final unchanged seal are under `C:/Users/gb/.codex_tmp/dual_rgbt_scheduled_adoption_source_review_20261016/`.

No SSH, GPU/NN execution, live CIM/process query, controller main call, scheduler change, experiment-input edit, commit or push occurred. The existing NN specification is reused; no canonical GT or full-test metric audit is repeated. Source admission remains denied until a fresh, exactly bound revision closes ADOPT-R1-F01.

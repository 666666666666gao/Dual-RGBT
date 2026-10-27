# VRAM18 phase6 resume source review

**Status: SOURCE_PASS. Blocking issues: 0.**

Fresh experiment-audit review; same-family / provisional. Requested reviewer gpt-6-astra with max reasoning; actual model and effort UNATTESTED.

The audited resume source is `tools/resume_vram18_serial_training_wave.py`, SHA-256 `2a9ecff4a1ca324228ace9bebb8208bbbe4b25cc966cef059847e578c34e59f6`. This is source admission only, not result certification.

## exact_interrupted_state — PASS

Exact byte identity with the saved interrupted execution is required. Phase6, eight registered cases and exactly three existing formal receipts are required. Local verification confirms complete equality with the phase6 index, task/run/GPU identities, and prior controller hash.

Evidence: `tools/resume_vram18_serial_training_wave.py:109`, `tools/resume_vram18_serial_training_wave.py:112`, `tools/resume_vram18_serial_training_wave.py:117`.

## reuse_eight_native0_m0_receipts — PASS

All eight saved M0 receipts contain native COMPLETE/exit0, four updates, frozen251/changed14 and strict reload PASS. The resume path re-reads and compares each entire receipt before reattaching or launching. It has no executed sanity launch path. This review inspected saved local receipts only.

Evidence: `tools/resume_vram18_serial_training_wave.py:165`, `tools/resume_vram18_serial_training_wave.py:171`.

## preserve_existing_nineteen — PASS

Original8 plus phase4 eight are ownership entries only. The existing phase6 three are validated with unchanged startup identity checks and appended as CUDA-ready owners. The saved diagnosis contains exactly these19 matching native PIDs. Existing formal entries are skipped; no stop/restart or duplicate launch branch exists.

Evidence: `tools/resume_vram18_serial_training_wave.py:96`, `tools/resume_vram18_serial_training_wave.py:105`, `tools/resume_vram18_serial_training_wave.py:175`, `tools/resume_vram18_serial_training_wave.py:180`, `tools/resume_vram18_serial_training_wave.py:184`.

## exact_remaining_five — PASS

Exact schedule AST under inert I/O launches only five missing formal entries, on GPUs3/0/1/2/3, with the corresponding existing M0 COMPLETE hash. Every launch receipt is persisted immediately and each new formal receives180 seconds before startup validation and the next launch.

Evidence: `tools/resume_vram18_serial_training_wave.py:169`, `tools/resume_vram18_serial_training_wave.py:182`, `tools/resume_vram18_serial_training_wave.py:186`, `tools/resume_vram18_serial_training_wave.py:188`.

## ownership_and_host_memory — PASS_SOURCE_ONLY

launch, admit, startup, measurement and score_health ASTs match the accepted serial controller exactly. Native owner state/PID, child command identity, required and actual CUDA PID sets, per-card limit6, GPU capacity, sampled compact-peak/inflight host-memory reservation and >8GiB remaining MemAvailable remain mandatory. Existing task/run output directories prohibit relaunch. No source pass certifies actual available resources at execution time.

Evidence: `tools/resume_vram18_serial_training_wave.py:139`, `tools/resume_vram18_serial_training_wave.py:150`, `tools/resume_vram18_serial_training_wave.py:155`, `tools/resume_vram18_serial_training_wave.py:158`, `tools/resume_vram18_serial_training_wave.py:159`.

## immutable_training_and_plans — PASS

The independent resume review must bind the exact resume source hash. Existing index, serial controller, V1 controller,58 source entries,32 training plans,24 case guards and6 prior review reports retain their sealed hashes. Resume contains no SCP source mutation or model/plan/weight/environment rewrite; only the local execution receipt is updated.

Evidence: `tools/resume_vram18_serial_training_wave.py:73`, `tools/resume_vram18_serial_training_wave.py:77`, `tools/resume_vram18_serial_training_wave.py:119`, `tools/resume_vram18_serial_training_wave.py:123`.

## prior_health_and_profile_gates — PASS_SOURCE_ONLY

The existing host-profile native0/equality/reduction checks and hash-bound phase4 ADMIT_NEXT_WAVE gate are retained. The supplied phase4 execution/health binding matches locally. These gates and the unchanged future wave-health scoring do not certify steady-state capacity,24/32 completed trainings, accuracy or onlineFPS.

Evidence: `tools/resume_vram18_serial_training_wave.py:80`, `tools/resume_vram18_serial_training_wave.py:87`, `tools/resume_vram18_serial_training_wave.py:102`, `tools/resume_vram18_serial_training_wave.py:103`.

## failure_lifecycle_and_syntax — PASS

Local compilation passed for all three controllers;21 generated resume programs parse. Exact schedule fixtures pass for success, M0 receipt mismatch, existing-job reattach failure, first new startup failure, and third new launch capacity failure. Failures start no subsequent task and preserve all successful launch identities. Fixtures neither execute remote programs nor wait in wall time.

Evidence: `tools/resume_vram18_serial_training_wave.py:174`, `tools/resume_vram18_serial_training_wave.py:178`, `tools/resume_vram18_serial_training_wave.py:187`, `tools/resume_vram18_serial_training_wave.py:189`.

## Resolved metadata observation

At the executor request, the resume-only state metadata now sets `startup_seconds_per_new_formal=180` beside its source identity, matching the actual180-second delay. Original active sources and plans remain unchanged. Hashes, compilation,21 generated programs and the five inert lifecycle fixtures passed again after this one-line edit.

## Scope and remaining runtime evidence

The source preserves16 prior formal jobs plus3 existing phase6 jobs, reuses8 native0 M0 receipts, and can launch only the remaining5. No remote query, process launch, weight read or protected source/plan modification was performed by this review. The verification consists of local hashes, compilation, saved receipt consistency and five inert lifecycle fixtures; it is not runtime/model evidence.

Each live launch still requires the source gates to pass. Actual wave6 health, training completion, accuracy and complete onlineFPS remain unverified by this review.

Verification: `.aris/traces/experiment-audit/2026-10-27_vram18_resume_source_review/verification.json`.

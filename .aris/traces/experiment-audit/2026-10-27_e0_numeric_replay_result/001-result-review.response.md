# E0 numeric replay result audit

Status: **RESULT_PASS**. Blockers: **0**. Issues: none.

Actual reviewer model/effort: **UNATTESTED**. Requested route: gpt-6-astra / max. Independence: **same-family**. Acceptance: **provisional**.

For the three fixed historical-E0 LasHeR DEV prefixes (frames 0..10), this replay with matmul TF32 disabled reproduces the archived reference boxes when cuDNN TF32 is disabled and the historical boxes when it is enabled, exactly at six decimal places; each setting repeats with identical raw serialized boxes and routes.

Exact plan SHA-256: `20b65baa4e19feecedcdaa3b8de2c58ccfaaa2bcd9f5bbf28155b11b83bbc223`.  
COMPLETE SHA-256: `4273ec4ce1943632d346a59ee42cb59a50f13bdc6a6c9aa246641a68b5d7ad0f`.  
Replay source SHA-256: `f4d13c941a29e9e763a04ccd07aafef56dc388dfbbf4b654c8bbe3b98f6e1aff`.

The exact 120-entry source map and all artifact hashes are retained in `RESULT_REVIEW.json`.

## Recomputed prefix comparisons

| Phase | cuDNN TF32 | Historical exact boxes | Reference exact boxes | Online predictions reproduced |
|---|---|---:|---:|---|
| 0 | false | 3/33 | 33/33 | 30/30 reference |
| 1 | true | 33/33 | 3/33 | 30/30 historical |
| 2 | false | 3/33 | 33/33 | 30/30 reference |
| 3 | true | 33/33 | 3/33 | 30/30 historical |

Each 33-box phase includes three supplied initialization boxes. Total: 12 rows, 132 boxes, 12 initializations and 120 online predictions. All 24 recorded archive comparisons recompute exactly. All six same-setting sequence-repeat pairs have identical raw serialized boxes and all route fields.

| Sequence | Maximum six-decimal box difference across settings |
|---|---:|
| boywalkinginsnow3 | 0.057640 |
| leftdrillmasterstanding | 0.020447 |
| leftgirlunderthelamp | 0.079078 |

Across settings only frame 0 is equal. For these ten online frames per sequence, action, modality and token counts stay equal; confidence differs. This does not establish behavior beyond the prefixes.

## Checks

### native identity and exit: PASS

Native wrapper task, exact server25-project Python/script/plan command, GPU 0 and positive wrapper PID match the sealed plan. Native state COMPLETE and exit_code 0 agree with the companion transport exit 0. The runtime source checks that this PID is the direct parent before inference. This audit reads the persisted records, not a live process.

Evidence: `runs/pipeline_tasks/diagnose_e0_numeric_replay_20261027/status.json:2`; `runs/pipeline_tasks/diagnose_e0_numeric_replay_20261027/status.json:4`; `runs/pipeline_tasks/diagnose_e0_numeric_replay_20261027/status.json:10`; `reports/E0_NUMERIC_REPLAY_NATIVE_20261027.json:2`; `tools/replay_e0_numeric_settings.py:33`; `scripts/run_task.py:24`.

### plan source and reference binding: PASS

Exact plan SHA matches SOURCE_PASS, whose source map equals the plan and whose bytes equal the original review trace. All 120 current local source hashes match, with all 119 inherited entries unchanged. Prefix JSON and six primary archive hashes/arrays remain exact. COMPLETE checkpoint identity and scope match the plan.

Evidence: `reports/E0_NUMERIC_REPLAY_PLAN_20261027.json:8`; `reports/E0_NUMERIC_REPLAY_PLAN_20261027.json:130`; `tools/replay_e0_numeric_settings.py:26`; `tools/replay_e0_numeric_settings.py:79`; `reports/E0_NUMERIC_REPLAY_PREFIXES_20261027.json:418`.

### raw prediction counts: PASS

Exactly 12 rows in planned phase/sequence order; each has 11 finite positive-size boxes and 10 valid routes. Thus 132 boxes comprise 12 dataset initialization boxes and 120 online predictions. This is 30 distinct online frame positions replayed in four phases.

Evidence: `runs/e0_numeric_replay_20261027/COMPLETE.json#/rows`; `tools/replay_e0_numeric_settings.py:55`; `tools/replay_e0_numeric_settings.py:61`.

### six decimal recomputation: PASS

Every reported exact-frame count and maximum absolute coordinate difference was recomputed independently from raw JSON predictions using six-decimal formatting. All 24 reference-comparison records match exactly. Each False phase reproduces all 33 reference boxes including three initialization boxes; each True phase reproduces all 33 historical boxes. The opposite archive matches only initialization (3/33 boxes, 0/30 online predictions). No accuracy metric or prediction-derived normalization is involved.

Evidence: `runs/e0_numeric_replay_20261027/COMPLETE.json#/rows`; `tools/replay_e0_numeric_settings.py:68`; `runs/pipeline_tasks/diagnose_e0_numeric_replay_20261027/run.log:11`.

### same setting repeats: PASS

For each sequence, phase 0 versus 2 and phase 1 versus 3 have exactly equal raw serialized prediction values and complete route dictionaries. All six repeat pairs have zero raw coordinate difference. This is equality of recorded JSON values, not an independent GPU tensor-bit attestation.

Evidence: `runs/e0_numeric_replay_20261027/COMPLETE.json#/rows`; `reports/E0_NUMERIC_REPLAY_PLAN_20261027.json:142`.

### between setting observations: PASS

Across False versus True, only frame 0 matches. All ten online frames per sequence differ in boxes and confidence; modality, action, primary-token counts and auxiliary-token counts stay equal for these short prefixes. Maximum six-decimal coordinate differences are 0.057640, 0.020447 and 0.079078 for the three planned sequences respectively.

Evidence: `runs/e0_numeric_replay_20261027/COMPLETE.json#/rows`; `reports/E0_NUMERIC_REPLAY_PREFIXES_20261027.json:3`; `reports/E0_NUMERIC_REPLAY_PREFIXES_20261027.json:141`; `reports/E0_NUMERIC_REPLAY_PREFIXES_20261027.json:279`.

### log and metadata: PASS

The sole JSON summary in run.log equals the COMPLETE rows with predictions/routes omitted. Log warnings are PyTorch floordiv deprecations; native status records successful completion. COMPLETE reports checkpoint ee60704..., PyTorch 1.12.1+cu113, RTX 3090, 103374215 parameters, zero optimizer updates, and initial flags matmul False / cuDNN True / benchmark False. These metadata are recorded, not independently measured again.

Evidence: `runs/pipeline_tasks/diagnose_e0_numeric_replay_20261027/run.log:1`; `runs/pipeline_tasks/diagnose_e0_numeric_replay_20261027/run.log:11`; `runs/e0_numeric_replay_20261027/COMPLETE.json#/checkpoint_sha256`; `runs/e0_numeric_replay_20261027/COMPLETE.json#/defaults_at_process_start`; `runs/e0_numeric_replay_20261027/COMPLETE.json#/torch_version`; `runs/e0_numeric_replay_20261027/COMPLETE.json#/total_parameters`.

### scope and evaluation type: PASS

The archived trajectories are model-output reproduction references, explicitly not dataset GT or tuning labels. The accepted claim is exact six-decimal prefix reproduction under the two settings in this one controlled replay, with identical in-process repetitions. No full-97-DEV causal attribution, PR/NPR/SR improvement, test accuracy or FPS claim is supported.

Evidence: `reports/E0_NUMERIC_REPLAY_PLAN_20261027.json:149`; `runs/e0_numeric_replay_20261027/COMPLETE.json#/scope`.

## Native completion

Task `diagnose_e0_numeric_replay_20261027`, wrapper PID `613681`, physical GPU `0`, native and companion exit codes `0`. Native start: 2026-10-27T15:57:48.943058+00:00; completion: 2026-10-27T15:58:17.970942+00:00. These are lifecycle records, not an FPS result.

## Limits

- The evidence covers only three 11-box prefixes: 30 distinct online frame positions, four phases, and two repeats per setting. It cannot establish behavior over all 97 DEV sequences or full sequence lengths.
- Historical effective numerical flags were unrecorded. Prefix reproduction does not prove the original full-run configuration or that cuDNN TF32 explains the entire historical/reference DEV metric discrepancy.
- The comparisons use archived model predictions as reproduction references, not GT. No PR, NPR, SR, accuracy improvement, TF32 model-selection recommendation, or change to the current precision protocol follows.
- Initialization boxes count in 11/11 comparisons; each row contains ten actual online predictions and one supplied initialization box.
- Repeat equality is within one process and compares recorded JSON values. No cross-process, cross-device, or GPU tensor-bit reproducibility claim is made.
- The raw source checkpoints and original data/manifest.json are not local audit inputs. Runtime source asserts their hashes before inference, and native exit 0 is consistent with those checks passing; this reviewer did not independently rehash remote bytes or images.
- COMPLETE lacks its own plan/native-task identity fields and is accepted only as part of this hash-bound artifact set with the reviewed plan, source review, status.json, run.log and companion native receipt.
- Native elapsed time is lifecycle evidence only, not a latency/FPS benchmark. Parameter count and device/software fields are recorded metadata, not fresh measurements.
- No additional NN execution, remote queries, training observation, target-code edits, training-plan edits, or pause actions were performed for this audit.

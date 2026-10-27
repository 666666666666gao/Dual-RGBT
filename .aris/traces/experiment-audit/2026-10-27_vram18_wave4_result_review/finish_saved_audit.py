"""Finish a local saved-evidence audit. No remote, torch, model, or weight access."""
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
TRACE = Path(__file__).resolve().parent
hashes = {}


def raw(name):
    name = str(name).replace('\\', '/')
    assert not name.endswith(('.pth', '.pth.tar', '.pt'))
    value = (ROOT / name).read_bytes()
    hashes[name] = hashlib.sha256(value).hexdigest()
    return value


def read(name):
    return json.loads(raw(name))


def rep(stem):
    return read('reports/' + stem + '_20261027.json')


def cite(name, needle, occurrence=0):
    lines = raw(name).decode('utf-8-sig').splitlines()
    matches = [i for i, value in enumerate(lines, 1) if needle in value]
    return name + ':' + str(matches[occurrence])


manifest_path = 'reports/VRAM18_COMPLETED_ADMISSION_METADATA_COLLECTION_20261027.json'
manifest = read(manifest_path)
metadata = []
for name, expected in manifest['files'].items():
    value = raw(name)
    assert hashes[name] == expected['sha256'] and len(value) == expected['bytes']
    value.decode('utf-8')
    metadata.append(dict(path=name, sha256=hashes[name], bytes=len(value), exact=True))
assert len(metadata) == 88

admissions = []
for item in metadata:
    name = item['path']
    if not name.endswith('/config.json'):
        continue
    run = name.rsplit('/', 1)[0]
    cfg, complete, sanity = [read(run + '/' + part) for part in ('config.json', 'COMPLETE.json', 'SANITY.json')]
    native_path = 'runs/pipeline_tasks/train_' + run.rsplit('/', 1)[1] + '/status.json'
    native = read(native_path)
    plan = read(cfg['plan'])
    assert cfg['plan_sha256'] == hashes[cfg['plan']]
    assert cfg['source_sha256'] == plan['source_sha256'] and cfg['input_sha256'] == plan['input_sha256']
    assert plan['seed'] == 2027
    rows = list(csv.DictReader(raw(run + '/trajectory_steps.csv').decode().splitlines()))
    assert [int(row['step']) for row in rows] == [1, 2, 3, 4]
    assert all(math.isfinite(float(row[k])) for row in rows for k in row if k == 'loss' or k.endswith('_loss'))
    assert (complete['updates'], complete['tracked_frames'], complete['initializes']) == (4, 20, 1)
    assert complete['seen_train_sequences'] == ['11runthree']
    assert complete['frozen_state_tensors'] == 251 and len(complete['changed_state_tensors']) == 14
    assert complete['selected_execution_bit_equal'] and complete['teacher_preserves_tracker_state']
    assert sanity['status'] == 'PASS' and sanity['strict_reload_prediction_bit_equal'] and sanity['four_request_legacy_bit_equal']
    r3 = run == 'runs/host_allocator_default_20261027_sanity'
    assert native['state'] == ('RUNNING' if r3 else 'COMPLETE')
    if not r3:
        assert native['exit_code'] == 0
    admissions.append(dict(run=run, native_state=native['state'], native_exit=native.get('exit_code'),
        disposition='FAILED_R3_KILLED_WRAPPER' if r3 else 'NATIVE0_FOUR_UPDATE_ADMISSION',
        updates=4, tracked_frames=20, sequence='11runthree', seed=2027,
        labelled_queries=complete['labelled_queries'], frozen_state_tensors=251, changed_state_tensors=14,
        evidence=[cite(native_path, '"state"'), cite(run + '/COMPLETE.json', '"updates"'),
                  cite(run + '/SANITY.json', '"strict_reload_prediction_bit_equal"'), run + '/trajectory_steps.csv:1']))
assert len(admissions) == 13 and sum(a['native_exit'] == 0 for a in admissions) == 12

execution_path = 'reports/VRAM18_WAVE4_EXECUTION_20261027.json'
node_path = 'reports/VRAM18_WAVE4_HEALTH_NODE_20261027.json'
health_path = 'reports/VRAM18_WAVE4_HEALTH_20261027.json'
execution, end, health = map(read, (execution_path, node_path, health_path))
index = rep('VRAM18_TRAINING_PLAN_INDEX')
node_plan = rep('VRAM18_WAVE4_HEALTH_NODE_PLAN')
start = execution['measurement_start']
before = {t['native']['task']: t for t in start['tasks']}
after = {t['native']['task']: t for t in end['tasks']}
registered = {t['task']: t for t in node_plan['tasks']}
assert len(execution['cases']) == 8 and len(before) == len(after) == 16
assert set(before) == set(after) == set(registered)
assert health['execution_sha256'] == hashes[execution_path] and health['raw_node_sha256'] == hashes[node_path]
assert health['index_sha256'] == hashes['reports/VRAM18_TRAINING_PLAN_INDEX_20261027.json']
for case in execution['cases']:
    run = 'runs/' + case['runs']['sanity']
    admission = case['sanity_admission']
    assert admission['complete'] == read(run + '/COMPLETE.json')
    assert admission['probe'] == read(run + '/SANITY.json')
    assert admission['native'] == read('runs/pipeline_tasks/' + case['tasks']['sanity'] + '/status.json')
    assert admission['complete_sha256'] == hashes[run + '/COMPLETE.json']
    command = case['formal']['command']
    assert command[command.index('--sanity-complete-sha256') + 1] == admission['complete_sha256']

older = read(index['baseline_previous'])
reference = read(index['baseline_reference'])
assert hashes[index['baseline_previous']] == index['baseline_previous_sha256']
assert hashes[index['baseline_reference']] == index['baseline_reference_sha256']
old_tasks = {t['native']['task']: t for t in older['tasks']}
ref_tasks = {t['task']: t for t in reference['tasks']}
interval = end['observed'] - start['observed']
reference_interval = reference['observed'] - older['observed']
rates, controls, retention, task_rows = {}, {}, {}, []
for task, a in before.items():
    b, registered_task = after[task], registered[task]
    assert a['native']['pid'] == b['native']['pid'] == registered_task['pid']
    assert a['child_pid'] == b['child_pid'] == registered_task['child_pid']
    assert a['native']['command'] == b['native']['command'] == registered_task['command']
    assert a['native']['physical_gpu'] == b['native']['physical_gpu'] == registered_task['gpu']
    assert a['native']['state'] == b['native']['state'] == 'RUNNING'
    assert a['finite_logged_losses'] and b['finite_logged_losses']
    delta = b['logged_updates'] - a['logged_updates']
    assert delta > 0
    rates[task] = delta / interval
    if task in ref_tasks:
        old, ref = old_tasks[task], ref_tasks[task]
        assert old['native']['pid'] == ref['pid'] == a['native']['pid']
        assert old['child_pid'] == ref['child_pid'] == a['child_pid']
        controls[task] = (ref['updates'] - old['logged_updates']) / reference_interval
        retention[task] = rates[task] / controls[task]
    task_rows.append(dict(task=task, gpu=a['native']['physical_gpu'], pid=a['native']['pid'], child_pid=a['child_pid'],
        start_updates=a['logged_updates'], end_updates=b['logged_updates'], delta=delta,
        updates_per_second=rates[task], reference_updates_per_second=controls.get(task), original_retention=retention.get(task)))
swap_start = sum(int(t['host_memory']['Swap'].split()[0]) for t in before.values())
swap_end = sum(int(t['host_memory']['Swap'].split()[0]) for t in after.values())
derived = dict(interval_seconds=interval, aggregate_updates_per_second=sum(rates.values()),
    previous_wave_reference_updates_per_second=sum(controls.values()), aggregate_ratio=sum(rates.values()) / sum(controls.values()),
    min_original_rate_retention=min(retention.values()), swap_growth_kib=swap_end - swap_start)
for key, value in derived.items():
    assert math.isclose(health[key], value, rel_tol=1e-13, abs_tol=1e-12)
assert rates == health['rates'] and retention == health['original_eight_rate_retention']
assert derived['aggregate_ratio'] > 1 and derived['min_original_rate_retention'] >= .4
assert derived['swap_growth_kib'] <= 65536 and health['mem_available_kib'] >= 8 * 1024**2
assert health['status'] == 'ADMIT_NEXT_WAVE'
wave4 = dict(**derived, start_observed=start['observed'], end_observed=end['observed'],
    total_update_delta=sum(t['delta'] for t in task_rows), original_reference_interval_seconds=reference_interval,
    original_reference_delta=sum(ref_tasks[n]['updates'] - old_tasks[n]['logged_updates'] for n in ref_tasks),
    task_count=16, tasks_per_gpu=dict(Counter(t['gpu'] for t in task_rows)),
    tasks=task_rows, swap_start_kib=swap_start, swap_end_kib=swap_end,
    mem_available_reported_kib=health['mem_available_kib'], mem_available_exact_endpoint_timestamp=None,
    mem_available_independent_raw_endpoint_receipt=False, gpu_memory_start_csv=start['gpu_csv'],
    interpretation='Historical staged-capacity comparison across different training intervals and evolution states; not paired training speedup, current status, completed training, or online FPS.')

memory = []
for stem in ['HOST_ALLOCATOR_R4_DEFAULT_SAMPLES', 'HOST_ALLOCATOR_R4_COMPACT_SAMPLES',
             'HOST_ALLOCATOR_TRIM_COMPACT_SAMPLES', 'HOST_ALLOCATOR_WARM_COMPACT_SAMPLES']:
    path = 'reports/' + stem + '_20261027.json'
    data = read(path)
    samples = data['samples']
    peak, last = max(s['pss_kib'] for s in samples), samples[-1]['pss_kib']
    assert peak == data['peak_sampled_pss_kib'] and last == data['last_sample_pss_kib']
    assert data['native']['state'] == 'COMPLETE' and data['native']['exit_code'] == 0
    memory.append(dict(report=path, samples=len(samples), sampled_peak_pss_kib=peak, last_sample_pss_kib=last,
        sampled_peak_pss_gib=peak / 1024**2, last_sample_pss_gib=last / 1024**2,
        sample_interval_seconds_min=min(b['observed'] - a['observed'] for a, b in zip(samples, samples[1:])),
        sample_interval_seconds_max=max(b['observed'] - a['observed'] for a, b in zip(samples, samples[1:])),
        native_elapsed_seconds=data['native']['completed'] - data['native']['started'],
        evidence=[cite(path, '"samples"'), cite(path, '"peak_sampled_pss_kib"'), cite(path, '"scope"', -1)]))
releases = []
for stem in ('HOST_ALLOCATOR_TRIM_STARTUP_MEMORY', 'HOST_ALLOCATOR_WARM_MEMORY_RELEASE'):
    path = 'reports/' + stem + '_20261027.json'
    rec = read(path)['record']
    releases.append(dict(report=path, malloc_trim_return=rec['malloc_trim_return'],
        pss_released_kib=int(rec['before']['Pss'].split()[0]) - int(rec['after']['Pss'].split()[0]),
        rss_released_kib=int(rec['before']['Rss'].split()[0]) - int(rec['after']['Rss'].split()[0]),
        evidence=[cite(path, '"Pss"'), cite(path, '"Pss"', 1), cite(path, '"malloc_trim_return"')]))

cpu_path = str((TRACE / 'completed_cpu_verification.json').relative_to(ROOT)).replace('\\', '/')
receipt_path = str((TRACE / 'completed_cpu_ssh_receipt.json').relative_to(ROOT)).replace('\\', '/')
cpu, receipt = read(cpu_path), read(receipt_path)
assert receipt['exit_code'] == 0 and json.loads(receipt['stdout']) == cpu
assert len(cpu['probes']) == 4 and len(cpu['comparisons']) == 3
for probe in cpu['probes']:
    run = 'runs/' + probe['run']
    complete = read(run + '/COMPLETE.json')
    assert probe['native'] == read('runs/pipeline_tasks/train_' + probe['run'] + '/status.json')
    assert probe['weight_sha256'] == complete['checkpoint_sha256']
    for name, expected in probe['metadata_sha256'].items():
        if name == 'query_labels.csv':
            assert expected == complete['private_labels_sha256']
        else:
            raw(run + '/' + name)
            assert hashes[run + '/' + name] == expected
for comparison in cpu['comparisons']:
    assert comparison['state_tensors_bit_equal'] == 265
    opt = comparison['optimizer']
    assert opt['tensor_count'] == 42 and not opt['bit_equal'] and opt['tolerance_pass']
    assert opt['atol'] == 1e-8 and opt['rtol'] == 1e-6
    assert all(comparison['csv_bit_equal'].values())
trim, warm = rep('HOST_ALLOCATOR_PAIRED_TRIM_PROFILE'), rep('HOST_ALLOCATOR_PAIRED_WARM_PROFILE')
for data, comparison in zip((trim, warm), cpu['comparisons'][1:]):
    assert data['equality']['optimizer_max_abs'] == comparison['optimizer']['max_abs']
    assert data['cases'][0]['reused_completed_reference']
assert execution['host_profile_sha256'] == hashes[index['host_profile']]
r3 = rep('HOST_ALLOCATOR_R3_RUNTIME_FAILURE_DIAGNOSIS')
failure = r3['files']['reports/train_host_allocator_default_20261027_sanity.profile_failure.json']
assert failure['wrapper_exit_code'] == -9 and failure['killed_owned_profile_group']
assert rep('HOST_ALLOCATOR_DEFAULT_SSH')['ssh_exit_code'] == 1
r4 = rep('HOST_ALLOCATOR_R4_COMPARISON_DIAGNOSIS')
assert json.loads(r4['stdout'])['optimizer_differences']
saved_local = read(str((TRACE / 'local_verification.json').relative_to(ROOT)))
assert saved_local['status'] == 'PASS' and not saved_local['failures']
source_bindings = {(item['path'], item['expected']) for item in saved_local['source_hash_checks_detail']}
for name, expected in source_bindings:
    raw(name)
    assert hashes[name] == expected
assert len({name for name, expected in source_bindings}) == 63

checks = {
    'gt_provenance': dict(status='PASS', details='TRAIN admission uses data/manifest.json train boxes; current-frame boxes enter label generation after track advances. Saved admissions are one sequence (11runthree), four updates and twenty tracked frames. No benchmark accuracy is inferred; dataset pixels/annotations were not reloaded in this finishing audit.',
        evidence=['scripts/train_candidate_joint_quality_trajectory.py:186', 'scripts/train_candidate_joint_quality_trajectory.py:297', 'scripts/train_candidate_joint_gainpair_trajectory.py:182', 'scripts/train_candidate_joint_gainpair_trajectory.py:294', admissions[0]['evidence'][1]]),
    'score_normalization': dict(status='PASS', details='Throughput is logged update delta divided by saved elapsed time; capacity ratio uses the earlier eight-task aggregate. PSS max is a sampled resource maximum, not a prediction-derived score denominator. Numerical comparison is explicitly differential verification, not ground-truth accuracy.',
        evidence=['tools/launch_vram18_training_wave.py:39', 'tools/launch_vram18_training_wave.py:46', 'tools/launch_vram18_training_wave.py:50', 'tools/profile_host_allocator_r4.py:72']),
    'result_existence': dict(status='PASS', details='All 88 collected raw metadata files match SHA-256 and byte counts (270491 bytes total). Eight wave4 admissions and four later memory probes have native COMPLETE/0. The R3 run remains failed despite its child COMPLETE/SANITY files. Embedded wave4 admission receipts and formal admission seals match raw files. All wave4 rates, retention and swap growth recompute.',
        evidence=[manifest_path + ':2', manifest_path + ':357', health_path + ':8', 'reports/HOST_ALLOCATOR_DEFAULT_SSH_20261027.json:3', 'reports/HOST_ALLOCATOR_R3_RUNTIME_FAILURE_DIAGNOSIS_20261027.json:109']),
    'dead_code': dict(status='PASS', details='Saved native commands, emitted startup/warm memory records and saved numerical-comparison output demonstrate use of the audited paths. This audit inspects those records and source; it does not re-execute training or a weight comparison.',
        evidence=['scripts/train_candidate_host_compact.py:17', 'scripts/train_candidate_host_compact.py:24', 'scripts/train_candidate_host_warm_compact.py:17', 'scripts/train_candidate_host_warm_compact.py:26', 'tools/profile_host_allocator_trim.py:116', 'tools/profile_host_allocator_warm.py:116', cpu_path + ':153']),
    'scope': dict(status='WARN', details='Single-seed four-update TRAIN admissions and historical capacity samples only. Numerical tolerance is not optimizer bit equality or full-training equivalence. Reused default controls, sequential profiling, evolving training states, two-second PSS samples and reload-clone peaks preclude clean-room aggregate, steady-state peak or malloc_trim benefit claims. MemAvailable is an extra score-time read with no exact endpoint timestamp.',
        evidence=[health_path + ':43', 'tools/launch_vram18_training_wave.py:58', 'tools/observe_vram18_training.py:55', 'reports/HOST_ALLOCATOR_TRIM_PROFILE_PLAN_20261027.json:233', 'reports/HOST_ALLOCATOR_WARM_PROFILE_PLAN_20261027.json:339', 'tools/profile_host_allocator_r4.py:72']),
    'evaluation_type': dict(status='PASS', classification='real_gt', details='Dataset-GT-backed TRAIN sanity plus engineering resource instrumentation and saved numerical regression checks; no completed benchmark, accuracy evaluation, online FPS measurement, or paper novelty evaluation.',
        evidence=['scripts/train_candidate_joint_quality_trajectory.py:187', 'scripts/train_candidate_joint_quality_trajectory.py:332', health_path + ':43'])}

claims = [
    dict(id='C1', claim='Eight wave4 candidates passed native0 four-update admission.', impact='supported', scope='One TRAIN sequence, seed2027, 20 tracked frames each; not formal completion.'),
    dict(id='C2', claim='Wave4 saved interval has 16 identified progressing tasks and 15.700198288483193 aggregate updates/s.', impact='supported', scope='15875 updates / 1011.1337263584137 seconds; historical snapshot only.'),
    dict(id='C3', claim='Historical aggregate/reference ratio is 1.5837446760914262.', impact='needs_qualifier', scope='Earlier 8-task reference: 168725 / 17019.99653530121 = 9.91333926831578 updates/s. Different workload counts and evolution intervals; no clean-room or paired speedup claim.'),
    dict(id='C4', claim='Four memory probes have matching 265 network tensors and CSV bytes after four updates.', impact='supported_by_saved_cpu_audit', scope='Finishing reviewer only read saved CPU receipts; optimizer is not bit equal.'),
    dict(id='C5', claim='Compact, trim and warm optimizer states pass atol1e-8/rtol1e-6.', impact='supported_by_saved_cpu_audit', scope='Maximum absolute differences 4.656612873077393e-10, 9.313225746154785e-10, 4.656612873077393e-10; no full-training equivalence.'),
    dict(id='C6', claim='Allocator and wrapper samples support broad malloc_trim memory benefit or speedup.', impact='unsupported', scope='Immediate trim releases only 82/116 KiB PSS; sampled endpoint/peak differences are not causal trim benefit or steady-state performance.'),
    dict(id='C7', claim='All four GPUs reached about18GiB at wave4.', impact='unsupported', scope='Saved measurement-start GPU use is 9447/9484/9439/9598 MiB, roughly9.2-9.4GiB, with four tasks per GPU.'),
    dict(id='C8', claim='These records establish accuracy, complete fulltrain equivalence, or onlineFPS.', impact='unsupported', scope='No such evaluation belongs to this evidence scope.'),
    dict(id='C9', claim='51165932 KiB MemAvailable was measured at the exact health-node endpoint.', impact='unsupported', scope='Value is saved from an additional score-time read; no exact timestamp or independent raw endpoint receipt is retained.'),
]

timestamp = datetime.now(timezone.utc).isoformat()
report = dict(audit_skill='experiment-audit', date='2026-10-27', generated_at=timestamp,
    verdict='WARN', overall_verdict='warn', integrity_status='warn', reason_code='bounded_operational_evidence_with_scope_limits',
    summary='Saved completed admissions and historical wave4 arithmetic are internally consistent. Preserve R3 runtime failure and R4 exact-optimizer failure. Accept only scoped operational and four-update numerical-tolerance conclusions.',
    deterministic_verdict='PASS', blocking_issues=0, blockers=[], review_independence='same-family', acceptance_status='provisional',
    reviewer_model_requested='gpt-6-astra', reviewer_reasoning_requested='max', reviewer_model='UNATTESTED', actual_model='UNATTESTED',
    reviewer_family='openai', model_attestation='UNATTESTED',
    agent_id='/root/finish_vram18_result_audit_20261027', verdict_id='/root/finish_vram18_result_audit_20261027',
    trace_path=str(TRACE.relative_to(ROOT)).replace('\\', '/'),
    access_scope=dict(local_saved_evidence_only=True, remote_calls=0, weight_reads=0, active_run_reads=0, new_experiments=0,
        completed_cpu_audit_attribution='Existing saved evidence from the preceding auditor; not executed or rerun by this finishing reviewer.',
        excluded=['live processes and remote state', 'active NN logs or weights', 'new training/evaluation', 'full-training equivalence', 'benchmark accuracy/onlineFPS', 'paper novelty']),
    checks=checks, claims=claims, metadata_count=len(metadata), metadata_bytes=sum(x['bytes'] for x in metadata),
    metadata_verification=metadata, admissions=admissions, wave4=wave4, memory_samples=memory, immediate_releases=releases,
    saved_cpu_verification=dict(status=cpu['status'], receipt_exit_code=receipt['exit_code'], receipt_stdout_matches=True,
        cuda_visible_devices=cpu['cuda_visible_devices'], threads=cpu['threads'], interop_threads=cpu['interop_threads'],
        comparisons=cpu['comparisons'], evidence=[cpu_path + ':2', cpu_path + ':153', cpu_path + ':295', receipt_path + ':12']),
    saved_predecessor_local_verification=dict(status=saved_local['status'], source_hash_checks=saved_local['source_hash_checks'],
        source_unique_files=saved_local['source_unique_files'], source_hash_mismatches=saved_local['source_hash_mismatches'],
        attribution='Saved deterministic check result; this finishing audit freshly rechecked raw88 metadata, admission seals, wave4 arithmetic, PSS reductions, and CPU receipt consistency.'),
    native_failures_preserved=[dict(run='host_allocator_default_20261027_sanity', failure='R3 observer ProcessLookupError; SSH1, wrapper reaped -9, native RUNNING stale',
        evidence=['reports/HOST_ALLOCATOR_DEFAULT_SSH_20261027.json:3', 'reports/HOST_ALLOCATOR_R3_RUNTIME_FAILURE_DIAGNOSIS_20261027.json:109']),
        dict(run='host_allocator_r4_compact_20261027_sanity', failure='Native0 execution succeeded, original exact optimizer comparison failed; later tolerant probes do not rewrite this failure.',
        evidence=['tools/profile_host_allocator_r4.py:102', 'reports/HOST_ALLOCATOR_R4_COMPARISON_DIAGNOSIS_20261027.json:2', 'reports/HOST_ALLOCATOR_TRIM_PROFILE_PLAN_20261027.json:233'])],
    action_items=['Keep the scope qualifiers and preserved failures attached when reporting these numbers.',
        'Label MemAvailable as score-time extra read without exact endpoint timestamp; do not present it as an independently timestamped endpoint sample.'],
    fresh_unique_source_hash_checks=63,
    audited_input_hashes=hashes)

lines = ['# VRAM18 wave4 and host-memory result audit', '',
    '**Verdict: WARN. Deterministic saved-evidence checks: PASS. No blocking issues.**', '',
    'Date: 2026-10-27. Fresh finishing reviewer: same-family, provisional; requested gpt-6-astra/max, actual model UNATTESTED. This audit reads local saved evidence only: no remote calls, active-run access, weights, or new experiments. The existing CPU audit is cited evidence; this reviewer did not execute it.', '',
    'The records support eight native0 wave4 admissions and a historical staged-capacity decision. They do not establish completed formal training, full-training equivalence, accuracy, online FPS, a controlled aggregate speedup, or material malloc_trim benefit.', '']
for key, check in checks.items():
    lines += ['## ' + key.replace('_', ' ').title() + ': ' + check['status'], '', check['details'], '',
              'Evidence: ' + '; '.join('`' + e + '`' for e in check['evidence']) + '.', '']
lines += ['## Recomputed wave4', '',
    'All 16 saved task identities match between the start and end (wrapper PID, child PID, GPU and command), with four per GPU, positive update deltas and finite logged losses. These are historical observations, not a current live-state assertion.', '',
    '| Quantity | Recomputed value |', '|---|---:|',
    '| Interval | 1011.1337263584137 s |', '| Total update delta | 15,875 |',
    '| Aggregate | 15.700198288483193 updates/s |',
    '| Earlier eight-task reference | 168,725 / 17019.99653530121 = 9.91333926831578 updates/s |',
    '| Historical aggregate/reference ratio | 1.5837446760914262 |',
    '| Minimum original-task rate retention | 0.8754537034556423 (87.5454%) |',
    '| Sum of trainer Swap, start / end | 6,932,716 / 6,959,732 KiB |',
    '| Swap increase | 27,016 KiB |',
    '| Extra score-time MemAvailable | 51,165,932 KiB; exact timestamp unavailable |', '',
    'The operational gate recomputes to ADMIT_NEXT_WAVE using the recorded MemAvailable. That value came from a separate read during scoring, not the timed raw health node. The exact read time and an independent raw endpoint receipt are absent. This is a reporting limitation, not an invented blocker.', '',
    'Evidence: `reports/VRAM18_WAVE4_EXECUTION_20261027.json:1934`; `reports/VRAM18_WAVE4_HEALTH_NODE_20261027.json:2`; `reports/VRAM18_WAVE4_HEALTH_20261027.json:8`; `tools/launch_vram18_training_wave.py:35`; `tools/launch_vram18_training_wave.py:58`.', '',
    'Measurement-start GPU use was 9447, 9484, 9439 and 9598 MiB. Wave4 does not establish that each card reached about 18 GiB. The 1.5837 ratio compares different intervals and counts of evolving training jobs. It is an operational aggregate ratio, not a paired clean-room speedup.', '',
    '## Completed raw admission evidence', '',
    'All 88 collected metadata files match the manifest SHA-256 and byte length, totaling 270,491 bytes. Eight wave4 admissions and four later memory probes have native COMPLETE/0. Each records four optimizer updates, twenty tracked frames, one TRAIN sequence (11runthree), seed 2027, 251 frozen and 14 changed tensors, plus strict reload/replay checks. The eight wave4 admission seals match the formal launch arguments. All 63 unique source files in the saved source-check inventory still match their expected hashes. This does not claim their formal runs completed.', '',
    'R3 is explicitly excluded from successful admissions: its child wrote COMPLETE and SANITY, but the observer encountered ProcessLookupError, SSH returned 1, its owned wrapper was reaped with -9, and native RUNNING remained stale. Later probes do not erase that failure.', '',
    'Evidence: `reports/VRAM18_COMPLETED_ADMISSION_METADATA_COLLECTION_20261027.json:2`; `reports/VRAM18_COMPLETED_ADMISSION_METADATA_COLLECTION_20261027.json:357`; `reports/HOST_ALLOCATOR_DEFAULT_SSH_20261027.json:3`; `reports/HOST_ALLOCATOR_R3_RUNTIME_FAILURE_DIAGNOSIS_20261027.json:109`. Per-run raw evidence and all 88 hashes are in the JSON report.', '',
    '## Memory samples and numerical comparison', '',
    '| Probe | Samples | Sampled peak PSS (GiB) | Last sampled PSS (GiB) | Native elapsed (s) |', '|---|---:|---:|---:|---:|']
for label, item in zip(['R4 default', 'R4 compact', 'Post-load trim', 'Post-update warm trim'], memory):
    lines.append('| %s | %d | %.6f | %.6f | %.6f |' % (label, item['samples'], item['sampled_peak_pss_gib'], item['last_sample_pss_gib'], item['native_elapsed_seconds']))
lines += ['', 'Samples are about two seconds apart and include load/reload clones. They are neither continuous peaks nor formal steady-state measurements. The later probes explicitly reuse the completed R4 default as their reference; they do not provide a new contemporaneous control. Elapsed times are recorded probe durations, not training speedup measurements.', '',
    'Immediate post-load malloc_trim released only 82 KiB PSS/96 KiB RSS; after four updates it released 116 KiB of each. Return value 1 alone does not establish useful memory savings. Differences in last sampled or peak PSS cannot be attributed to a material malloc_trim benefit.', '',
    'The saved CPU audit reports all 265 network tensors and both trajectory/query-label CSV files bit equal against R4 default. The optimizer state (42 tensors checked) is **not bit equal**: maximum absolute differences are 4.656612873077393e-10 (compact), 9.313225746154785e-10 (post-load trim), and 4.656612873077393e-10 (warm trim). All pass atol 1e-8/rtol 1e-6. These are four-update numerical checks, not full-training equivalence. The original R4 exact-optimizer requirement remains failed.', '',
    'The saved CPU SSH receipt exits 0 and its parsed stdout exactly matches completed_cpu_verification.json. Its native statuses and metadata hashes match the collected raw records, and its stored weight/label hashes match the COMPLETE records. This finishing audit checked that receipt consistency without loading any weights or private query-label files. Stored state-element count includes buffers and is not substituted for the model parameter count.', '']
for item in memory + releases:
    lines.append('Evidence: ' + '; '.join('`' + e + '`' for e in item['evidence']) + '.')
lines += ['', 'Additional evidence: `' + cpu_path + ':153`; `' + cpu_path + ':295`; `tools/profile_host_allocator_r4.py:102`; `reports/HOST_ALLOCATOR_R4_COMPARISON_DIAGNOSIS_20261027.json:2`; `reports/HOST_ALLOCATOR_TRIM_PROFILE_PLAN_20261027.json:233`; `reports/HOST_ALLOCATOR_WARM_PROFILE_PLAN_20261027.json:339`.', '',
    '## Claim handling', '',
    '- Supported: completed four-update admissions, preserved native failures, exact raw metadata seals, recomputed historical wave4 counts/rates, and the explicitly saved numerical-tolerance results.',
    '- Qualified: the historical 1.5837 aggregate/reference ratio, sampled PSS values, and score-time MemAvailable without an exact timestamp.',
    '- Unsupported by this evidence: full-training equivalence, optimizer bit equality, material malloc_trim benefit, controlled aggregate speedup, approximately 18 GiB on all cards at wave4, benchmark accuracy, or online FPS.', '',
    'No new run or approval is required to finish this saved-evidence audit. Retain these qualifications whenever the numbers are reused.', '']

# Read cited source files into the sealed input inventory and validate every exact line reference.
for check in checks.values():
    for evidence in check['evidence']:
        name, number = evidence.rsplit(':', 1)
        assert 1 <= int(number) <= len(raw(name).decode('utf-8-sig').splitlines())
raw(str(Path(__file__).relative_to(ROOT)))
output = ROOT / 'reports/VRAM18_WAVE4_AND_MEMORY_RESULT_AUDIT_20261027'
output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
output.with_suffix('.md').write_text('\n'.join(lines), encoding='utf-8')
(TRACE / 'finish_saved_verification.json').write_text(json.dumps(dict(status='PASS', metadata_count=88,
    metadata_bytes=270491, unique_source_hash_checks=63, successful_admissions=12, preserved_failed_admissions=1, wave4=wave4,
    memory_samples=memory, immediate_releases=releases, cpu_receipt_stdout_matches=True,
    no_remote_calls=True, no_weight_reads=True), indent=2) + '\n', encoding='utf-8')
(TRACE / '002-finish-review.response.md').write_text('\n'.join(lines), encoding='utf-8')
(TRACE / '002-finish-review.meta.json').write_text(json.dumps(dict(timestamp=timestamp,
    agent_id=report['agent_id'], requested_model='gpt-6-astra', requested_reasoning='max', actual_model='UNATTESTED',
    review_independence='same-family', acceptance_status='provisional', status='ok', verdict='WARN',
    report='reports/VRAM18_WAVE4_AND_MEMORY_RESULT_AUDIT_20261027.json'), indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(verdict='WARN', deterministic_verdict='PASS', blockers=0,
    metadata_count=88, metadata_bytes=270491, successful_admissions=12, preserved_failed_admissions=1,
    wave4_aggregate=wave4['aggregate_updates_per_second'], report=str(output))))

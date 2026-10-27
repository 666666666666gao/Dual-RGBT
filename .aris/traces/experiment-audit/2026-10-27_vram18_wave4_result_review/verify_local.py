"""Deterministic verification of saved local receipts; never contacts a server."""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
TRACE = Path(__file__).resolve().parent
hashes, parsed, source_checks, input_checks = {}, {}, [], []
failures = []

def sha(name):
    path = ROOT / name
    value = hashlib.sha256(path.read_bytes()).hexdigest()
    hashes[str(name).replace('\\', '/')] = value
    return value

def read(name):
    name = str(name).replace('\\', '/')
    if name not in parsed:
        sha(name)
        parsed[name] = json.loads((ROOT / name).read_bytes())
    return parsed[name]

def check(condition, label):
    if not condition:
        failures.append(label)

def report(stem):
    return read('reports/' + stem + '_20261027.json')

manifest = report('VRAM18_COMPLETED_ADMISSION_METADATA_COLLECTION')
metadata = []
for name, expected in manifest['files'].items():
    path = ROOT / name
    actual = sha(name)
    check(actual == expected['sha256'], 'metadata SHA: ' + name)
    check(path.stat().st_size == expected['bytes'], 'metadata bytes: ' + name)
    if path.suffix == '.json':
        read(name)
    elif path.suffix in ('.log', '.csv'):
        path.read_text(encoding='utf-8')
    metadata.append(dict(path=name, sha256=actual, bytes=path.stat().st_size, exact=actual == expected['sha256']))
check(len(metadata) == 88, '88 metadata count')

initial = [
    'VRAM18_PRE_EXPANSION_BASELINE', 'VRAM18_WAVE4_EXECUTION', 'VRAM18_WAVE4_HEALTH_NODE_PLAN',
    'VRAM18_WAVE4_HEALTH_NODE', 'VRAM18_WAVE4_HEALTH', 'VRAM18_TRAINING_PLAN_INDEX',
    'VRAM18_TRAINING_SOURCE_REVIEW', 'VRAM18_TRAINING_GRID',
    'HOST_ALLOCATOR_R3_RUNTIME_FAILURE_DIAGNOSIS', 'HOST_ALLOCATOR_DEFAULT_SSH',
    'HOST_ALLOCATOR_PROFILE_PLAN', 'HOST_ALLOCATOR_PROFILE_SOURCE_REVIEW',
    'HOST_ALLOCATOR_PROFILE_R4_PLAN', 'HOST_ALLOCATOR_PROFILE_R4_SOURCE_REVIEW',
    'HOST_ALLOCATOR_R4_DEFAULT_SAMPLES', 'HOST_ALLOCATOR_R4_COMPACT_SAMPLES',
    'HOST_ALLOCATOR_R4_COMPARISON_DIAGNOSIS', 'HOST_ALLOCATOR_R4_DEFAULT_SSH',
    'HOST_ALLOCATOR_R4_COMPACT_SSH', 'HOST_ALLOCATOR_PAIRED_TRIM_PROFILE',
    'HOST_ALLOCATOR_TRIM_STARTUP_MEMORY', 'HOST_ALLOCATOR_TRIM_PROFILE_PLAN',
    'HOST_ALLOCATOR_TRIM_PROFILE_SOURCE_REVIEW', 'HOST_ALLOCATOR_TRIM_COMPACT_SAMPLES',
    'HOST_ALLOCATOR_TRIM_COMPACT_SSH', 'HOST_ALLOCATOR_PAIRED_WARM_PROFILE',
    'HOST_ALLOCATOR_WARM_MEMORY_RELEASE', 'HOST_ALLOCATOR_WARM_PROFILE_PLAN',
    'HOST_ALLOCATOR_WARM_PROFILE_SOURCE_REVIEW', 'HOST_ALLOCATOR_WARM_COMPACT_SAMPLES',
    'HOST_ALLOCATOR_WARM_COMPACT_SSH']
for stem in initial:
    report(stem)
read('reports/FOUR_GPU_EIGHT_TRAIN_FIRST_NODE_20261026.json')
execution = report('VRAM18_WAVE4_EXECUTION')
index = report('VRAM18_TRAINING_PLAN_INDEX')
for case in execution['cases']:
    plan = read(case['plan'])
    read(plan['source_review'])
for stem in ['HOST_ALLOCATOR_PROFILE_PLAN', 'HOST_ALLOCATOR_PROFILE_R4_PLAN',
             'HOST_ALLOCATOR_TRIM_PROFILE_PLAN', 'HOST_ALLOCATOR_WARM_PROFILE_PLAN']:
    for case in report(stem)['cases']:
        plan = read(case['plan'])
        read(plan['source_review'])

checked_docs = set()
while set(parsed) - checked_docs:
    name = next(iter(set(parsed) - checked_docs))
    doc = parsed[name]
    checked_docs.add(name)
    if not isinstance(doc, dict):
        continue
    for src, expected in doc.get('source_sha256', {}).items():
        actual = sha(src)
        source_checks.append(dict(owner=name, path=src, expected=expected, actual=actual, exact=actual == expected))
    for inp, expected in doc.get('input_sha256', {}).items():
        if inp.endswith(('.pth', '.pth.tar')):
            input_checks.append(dict(owner=name, path=inp, expected=expected, status='weight_not_read_scope'))
        elif (ROOT / inp).is_file():
            actual = sha(inp)
            input_checks.append(dict(owner=name, path=inp, expected=expected, actual=actual, status='exact' if actual == expected else 'mismatch'))
    for key in ['plan', 'source_review']:
        value = doc.get(key)
        if isinstance(value, str) and value.endswith('.json') and (ROOT / value).is_file():
            read(value)

admissions = []
configs = [n for n in manifest['files'] if n.endswith('/config.json')]
for name in configs:
    run = str(Path(name).parent).replace('\\', '/')
    cfg = read(name)
    plan = read(cfg['plan'])
    complete = read(run + '/COMPLETE.json')
    probe = read(run + '/SANITY.json')
    native = read('runs/pipeline_tasks/train_' + run.split('/')[-1] + '/status.json')
    rows = list(csv.DictReader((ROOT / run / 'trajectory_steps.csv').read_text().splitlines()))
    check(cfg['plan_sha256'] == sha(cfg['plan']), 'plan SHA: ' + run)
    check(cfg['source_sha256'] == plan['source_sha256'], 'config source map: ' + run)
    check(cfg['input_sha256'] == plan['input_sha256'], 'config input map: ' + run)
    check((complete['updates'], complete['tracked_frames'], complete['initializes']) == (4, 20, 1), 'sanity counts: ' + run)
    check(len(rows) == 4 and [int(r['step']) for r in rows] == [1,2,3,4], 'CSV steps: ' + run)
    check(all(math.isfinite(float(row[k])) for row in rows for k in row if k == 'loss' or k.endswith('_loss')), 'finite loss: ' + run)
    check(complete['seen_train_sequences'] == ['11runthree'], 'one genuine TRAIN sequence: ' + run)
    check(sum(complete['route_frames'].values()) == 20 and sum(complete['sampled_family_counts'].values()) == 8, 'route family counts: ' + run)
    check(complete['sampled_query_labels'] == 8 and complete['labelled_queries'] == 8 + complete['extra_selected_query_labels'], 'label counts: ' + run)
    check(complete['frozen_state_tensors'] == 251 and len(complete['changed_state_tensors']) == 14, 'frozen changed: ' + run)
    check(probe['status'] == 'PASS' and probe['strict_reload_prediction_bit_equal'] and probe['four_request_legacy_bit_equal'], 'reload probe: ' + run)
    check(complete['selected_execution_bit_equal'] and complete['teacher_preserves_tracker_state'], 'teacher replay: ' + run)
    is_r3 = run == 'runs/host_allocator_default_20261027_sanity'
    check(native['state'] == ('RUNNING' if is_r3 else 'COMPLETE'), 'native state: ' + run)
    if not is_r3:
        check(native['exit_code'] == 0, 'native 0: ' + run)
    admissions.append(dict(run=run, plan=cfg['plan'], native_state=native['state'], native_exit=native.get('exit_code'),
                           audit_admission='FAILED_R3_KILLED_WRAPPER' if is_r3 else 'NATIVE0_4STEP',
                           updates=len(rows), frames=complete['tracked_frames'], sequence=complete['seen_train_sequences'],
                           labelled_queries=complete['labelled_queries'], frozen=complete['frozen_state_tensors'],
                           changed=len(complete['changed_state_tensors']), strict_reload=probe['strict_reload_prediction_bit_equal'],
                           complete_sha256=sha(run + '/COMPLETE.json'), weight_sha256=complete['checkpoint_sha256']))

for case in execution['cases']:
    run = 'runs/' + case['runs']['sanity']
    adm = case['sanity_admission']
    check(adm['complete'] == read(run + '/COMPLETE.json'), 'embedded COMPLETE: ' + run)
    check(adm['probe'] == read(run + '/SANITY.json'), 'embedded SANITY: ' + run)
    check(adm['native'] == read('runs/pipeline_tasks/' + case['tasks']['sanity'] + '/status.json'), 'embedded native: ' + run)
    check(adm['complete_sha256'] == sha(run + '/COMPLETE.json'), 'sanity seal: ' + run)
    command = case['formal']['command']
    check(command[command.index('--sanity-complete-sha256') + 1] == sha(run + '/COMPLETE.json'), 'formal admission seal: ' + run)

start, end = execution['measurement_start'], report('VRAM18_WAVE4_HEALTH_NODE')
old, pre = read('reports/FOUR_GPU_EIGHT_TRAIN_FIRST_NODE_20261026.json'), report('VRAM18_PRE_EXPANSION_BASELINE')
health = report('VRAM18_WAVE4_HEALTH')
node = report('VRAM18_WAVE4_HEALTH_NODE_PLAN')
before = {v['native']['task']: v for v in start['tasks']}
after = {v['native']['task']: v for v in end['tasks']}
older = {v['native']['task']: v for v in old['tasks']}
reference = {v['task']: v for v in pre['tasks']}
node_tasks = {v['task']: v for v in node['tasks']}
check(len(before) == len(after) == 16 and set(before) == set(after) == set(node_tasks), '16 identities')
dt, old_dt = end['observed'] - start['observed'], pre['observed'] - old['observed']
rates, controls, retention, per_task = {}, {}, {}, []
for task, a in before.items():
    b, n = after[task], node_tasks[task]
    check(a['native']['pid'] == b['native']['pid'] == n['pid'], 'same wrapper: ' + task)
    check(a['child_pid'] == b['child_pid'] == n['child_pid'], 'same child: ' + task)
    check(a['native']['command'] == b['native']['command'] == n['command'], 'same command: ' + task)
    check(a['native']['physical_gpu'] == b['native']['physical_gpu'] == n['gpu'], 'same GPU: ' + task)
    check(a['native']['state'] == b['native']['state'] == 'RUNNING', 'historical RUNNING: ' + task)
    check(a['finite_logged_losses'] and b['finite_logged_losses'], 'historical finite losses: ' + task)
    delta = b['logged_updates'] - a['logged_updates']
    check(delta > 0, 'positive update delta: ' + task)
    rates[task] = delta / dt
    if task in reference:
        prior, ref = older[task], reference[task]
        check(prior['native']['pid'] == ref['pid'] == a['native']['pid'], 'reference wrapper: ' + task)
        check(prior['child_pid'] == ref['child_pid'] == a['child_pid'], 'reference child: ' + task)
        controls[task] = (ref['updates'] - prior['logged_updates']) / old_dt
        retention[task] = rates[task] / controls[task]
    per_task.append(dict(task=task, pid=a['native']['pid'], child_pid=a['child_pid'], gpu=n['gpu'],
                         start_updates=a['logged_updates'], end_updates=b['logged_updates'], delta=delta,
                         updates_per_second=rates[task], reference_updates_per_second=controls.get(task), retention=retention.get(task)))
swap_start = sum(int(v['host_memory']['Swap'].split()[0]) for v in before.values())
swap_end = sum(int(v['host_memory']['Swap'].split()[0]) for v in after.values())
derived = dict(interval_seconds=dt, aggregate_updates_per_second=sum(rates.values()),
               previous_wave_reference_updates_per_second=sum(controls.values()),
               aggregate_ratio=sum(rates.values())/sum(controls.values()),
               min_original_rate_retention=min(retention.values()), swap_growth_kib=swap_end-swap_start)
for key, value in derived.items():
    check(math.isclose(value, health[key], rel_tol=1e-13, abs_tol=1e-12), 'health derived ' + key)
check(rates == health['rates'] and retention == health['original_eight_rate_retention'], 'all health rate dictionaries')
check(health['execution_sha256'] == sha('reports/VRAM18_WAVE4_EXECUTION_20261027.json'), 'execution seal')
check(health['raw_node_sha256'] == sha('reports/VRAM18_WAVE4_HEALTH_NODE_20261027.json'), 'node seal')
check(health['index_sha256'] == sha('reports/VRAM18_TRAINING_PLAN_INDEX_20261027.json'), 'index seal')
check(node['observer_sha256'] == sha(node['observer_source']), 'observer seal')
check(execution['host_profile_sha256'] == sha(index['host_profile']), 'host profile seal')

memory = []
for stem in ['HOST_ALLOCATOR_R4_DEFAULT_SAMPLES', 'HOST_ALLOCATOR_R4_COMPACT_SAMPLES',
             'HOST_ALLOCATOR_TRIM_COMPACT_SAMPLES', 'HOST_ALLOCATOR_WARM_COMPACT_SAMPLES']:
    data = report(stem)
    samples = data['samples']
    peak, last = max(v['pss_kib'] for v in samples), samples[-1]['pss_kib']
    check(peak == data['peak_sampled_pss_kib'] and last == data['last_sample_pss_kib'], 'sample peak/last: ' + stem)
    check(data['native']['state'] == 'COMPLETE' and data['native']['exit_code'] == 0, 'sample native0: ' + stem)
    memory.append(dict(report=stem, n=len(samples), peak_sampled_pss_kib=peak, last_sample_pss_kib=last,
                       peak_gib=peak/1024**2, last_gib=last/1024**2,
                       interval_min_seconds=min(b['observed']-a['observed'] for a,b in zip(samples,samples[1:])),
                       interval_max_seconds=max(b['observed']-a['observed'] for a,b in zip(samples,samples[1:])),
                       native_elapsed=data['native']['completed']-data['native']['started'],
                       native_start=data['native']['started'], native_end=data['native']['completed']))
trim = {}
for stem in ['HOST_ALLOCATOR_TRIM_STARTUP_MEMORY', 'HOST_ALLOCATOR_WARM_MEMORY_RELEASE']:
    data = report(stem)['record']
    trim[stem] = dict(pss_released_kib=int(data['before']['Pss'].split()[0])-int(data['after']['Pss'].split()[0]),
                      rss_released_kib=int(data['before']['Rss'].split()[0])-int(data['after']['Rss'].split()[0]),
                      malloc_trim_return=data['malloc_trim_return'])

out = dict(status='PASS' if not failures else 'FAIL', failures=failures, metadata_count=len(metadata), metadata=metadata,
           admissions=admissions, source_hash_checks=len(source_checks), source_unique_files=len({v['path'] for v in source_checks}),
           source_hash_mismatches=[v for v in source_checks if not v['exact']], source_hash_checks_detail=source_checks,
           input_checks=input_checks, audited_input_hashes=hashes,
           wave4=dict(**derived, original_reference_seconds=old_dt, original_reference_delta=sum(reference[n]['updates']-older[n]['logged_updates'] for n in reference),
                      total_update_delta=sum(v['delta'] for v in per_task), tasks=per_task, swap_start_kib=swap_start,
                      swap_end_kib=swap_end, mem_available_reported_kib=health['mem_available_kib'],
                      mem_available_independent_raw_receipt=False,
                      gpu_memory_at_start_csv=start['gpu_csv']), memory=memory, trim=trim)
(TRACE/'local_verification.json').write_text(json.dumps(out, indent=2)+'\n', encoding='utf-8')
print(json.dumps({k:out[k] for k in ['status','failures','metadata_count','source_hash_checks','source_unique_files','source_hash_mismatches','memory','trim']},separators=(',',':')))
print(json.dumps(dict(wave4=out['wave4']),separators=(',',':')))

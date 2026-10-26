"""Independent stdlib-only recomputation; never imports the experimental code."""
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import pstats
import statistics
import struct

ROOT = Path(__file__).resolve().parents[4]
TRACE = Path(__file__).resolve().parent
PLAN_PATH = ROOT / 'reports/RANKSTOP_INFERENCE_SCALING_PLAN_20261026.json'
plan = json.loads(PLAN_PATH.read_bytes())
OUT = ROOT / 'profile_output' / plan['run']
checks = []
hashes = {}
sample_sum_anomalies = []


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def load(path):
    path = Path(path)
    hashes[str(path.relative_to(ROOT)).replace('\\', '/')] = sha(path)
    return json.loads(path.read_bytes())


def check(name, passed):
    checks.append(dict(check=name, passed=bool(passed)))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest_json(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sample_summary(rows, gpu):
    values = [row['gpus'][gpu] for row in rows]
    return dict(samples=len(values), used_min_mib=min(row[1] for row in values),
                used_max_mib=max(row[1] for row in values),
                free_min_mib=min(row[2] for row in values),
                utilization_min=min(row[3] for row in values),
                utilization_max=max(row[3] for row in values),
                utilization_mean=sum(row[3] for row in values)/len(values))


plan = load(PLAN_PATH)
review = load(ROOT / plan['source_review'])
launch = load(ROOT / 'reports/RANKSTOP_INFERENCE_SCALING_LAUNCH_20261026.json')
startup = load(ROOT / 'reports/RANKSTOP_INFERENCE_SCALING_STARTUP_20261026.json')
node = load(ROOT / 'reports/RANKSTOP_INFERENCE_SCALING_NODE_20261026.json')
config = load(OUT / 'config.json')
complete = load(OUT / 'COMPLETE.json')
native = load(ROOT / 'runs/pipeline_tasks' / plan['task'] / 'status.json')
dev_plan = load(ROOT / 'reports/rankstop_dev_plan_server26_20261026.json')
check('frozen_plan_matches_runtime_config', config['plan'] == plan)
check('plan_hash_matches_review_launch_config', sha(PLAN_PATH) == review['plan_sha256'] == launch['plan_sha256'] == config['plan_sha256'])
check('native_snapshot_identity', config['native'] == startup['native'])
check('native_snapshot_to_terminal', {k: native[k] for k in config['native'] if k != 'state'} == {k: v for k, v in config['native'].items() if k != 'state'})
check('native_terminal_zero', native['state'] == 'COMPLETE' and native['exit_code'] == 0 and native['physical_gpu'] == '0')
check('native_launch_owner', native['pid'] == launch['task']['pid'] == startup['native']['pid'])
check('node_snapshots_exact', node['native'] == native and node['complete'] == complete)
check('complete_and_best_profile_exclusion', complete['status'] == 'COMPLETE' and len(complete['arms']) == 10 and len(complete['profile_arms']) == 2 and all(not x['profile'] for x in complete['arms']) and all(x['profile'] for x in complete['profile_arms']))
check('inherited48_source_bindings', len(dev_plan['source_sha256']) == 48 and all(plan['source_sha256'].get(k) == v for k, v in dev_plan['source_sha256'].items()))
check('source_review_binding', plan['source_sha256'] == review['source_sha256'] and review['status'] == 'SOURCE_PASS' and review['blocking_issues'] == 0)
sources = []
for name, expected in plan['source_sha256'].items():
    path = ROOT / name
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    actual = sha(path)
    hashes[name] = actual
    check('source_sha256:' + name, actual == expected)
    sources.append(dict(path=name, sha256=actual, ast_nodes=sum(1 for _ in ast.walk(tree))))
inputs = []
for name, expected in plan['input_sha256'].items():
    path = ROOT / name
    if path.is_file():
        actual = sha(path)
        hashes[name] = actual
        check('local_input_sha256:' + name, actual == expected)
        inputs.append(dict(path=name, status='MATCH' if actual == expected else 'MISMATCH', sha256=actual))
    else:
        inputs.append(dict(path=name, status='REMOTE_ONLY'))

log_path = ROOT / 'runs/pipeline_tasks' / plan['task'] / 'run.log'
hashes[str(log_path.relative_to(ROOT)).replace('\\', '/')] = sha(log_path)
log_lines = log_path.read_text(encoding='utf-8').splitlines()
log_json = [(i+1, json.loads(line)) for i, line in enumerate(log_lines) if line.startswith('{')]
check('run_log_exact12arms_terminal', len(log_json) == 13 and log_json[-1][1]['status'] == 'COMPLETE')
check('run_log_no_traceback', not any('Traceback (most recent call last)' in line for line in log_lines))
arm_rows, all_workers, sequence_reference, per_sequence = [], [], None, []
for index in range(12):
    profile = index >= 10
    item = complete['profile_arms'][index-10] if profile else complete['arms'][index]
    cv, count = item['opencv_threads'], item['processes']
    directory = OUT / ('cprofile_cv%d' % cv if profile else 'arm%d_n%d_cv%d' % (index, count, cv))
    local_complete = load(directory / 'COMPLETE.json')
    check(directory.name + ':complete_exact', local_complete == {k:v for k,v in item.items() if k != 'serial_prediction_route_state_equal'})
    check(directory.name + ':log_summary_exact', log_json[index][1] == {k:v for k,v in local_complete.items() if k != 'samples'})
    if not profile:
        check(directory.name + ':planned_setting', plan['arms'][index] == dict(processes=count, opencv_threads=cv))
        check(directory.name + ':top_parity_flag', item['serial_prediction_route_state_equal'] is True)
    workers = [load(directory / ('worker%d.json' % i)) for i in range(count)]
    all_workers.extend(workers)
    selected = plan['sequence_names'][:1] if profile else plan['sequence_names']
    names, results, worker_details = [], {}, []
    for i, worker in enumerate(workers):
        assigned = selected[i::count]
        check(directory.name + ':worker%d_assignment' % i, [row['sequence'] for row in worker['results']] == assigned)
        check(directory.name + ':worker%d_frames' % i, worker['frames'] == len(assigned)*128)
        check(directory.name + ':worker%d_settings' % i, worker['profile'] == profile and worker['cv2_threads'] == cv and worker['torch_threads'] == 1 and worker['total_parameters'] == 103218887 and worker['gpu_name'] == 'NVIDIA GeForce RTX 3090')
        check(directory.name + ':worker%d_runtime_assertions' % i, worker['model_state_bit_equal'] is True and worker['no_backward_or_optimizer'] is True)
        check(directory.name + ':worker%d_time' % i, native['started'] < worker['started'] < worker['finished'] < native['completed'])
        for row in worker['results']:
            name = row['sequence']
            names.append(name)
            results[name] = row
            check(directory.name + ':' + name + ':counts', len(row['predictions']) == 128 and len(row['routes']) == 127)
            check(directory.name + ':' + name + ':finite_boxes', all(len(box) == 4 and all(math.isfinite(v) for v in box) and min(box[2:]) > 0 for box in row['predictions']))
            check(directory.name + ':' + name + ':route_fields', all(set(route) == {'modality','action','request_index','candidate_rank','candidate_count','confidence','primary_tokens','auxiliary_tokens'} for route in row['routes']))
            check(directory.name + ':' + name + ':route_values', all(route['modality'] in (0,1) and route['action'] in range(5) and 0 <= route['candidate_count'] <= 8 and math.isfinite(route['confidence']) and route['primary_tokens'] == 385 and route['auxiliary_tokens'] in (0,193,385) for route in row['routes']))
            check(directory.name + ':' + name + ':sha256_form', len(row['final_state_sha256']) == 64 and all(c in '0123456789abcdef' for c in row['final_state_sha256']))
        worker_details.append(dict(index=i, pid=worker['pid'], sequences=assigned, frames=worker['frames'],
            measured_seconds=worker['finished']-worker['started'], block_fps=worker['frames']/(worker['finished']-worker['started']),
            peak_allocated_bytes=worker['peak_allocated_bytes'], peak_reserved_bytes=worker['peak_reserved_bytes']))
    check(directory.name + ':complete_unique_work', len(names) == len(set(names)) == len(selected) and set(names) == set(selected))
    check(directory.name + ':pid_exit_ownership', [w['pid'] for w in workers] == item['worker_pids'] and sorted(x['pid'] for x in item['admissions']) == sorted(item['worker_pids']) and item['worker_exit_codes'] == [0]*count)
    if sequence_reference is None:
        sequence_reference = results
    check(directory.name + ':canonical_predictions_routes_state_parity', canonical(results) == canonical({name:sequence_reference[name] for name in selected}))
    start = min(w['started'] for w in workers)
    end = max(w['finished'] for w in workers)
    frames = sum(len(row['predictions']) for row in results.values())
    timed = [row for row in item['samples'] if start <= row['epoch'] <= end]
    samples = item['samples']
    check(directory.name + ':samples_increasing', all(a['epoch'] < b['epoch'] for a,b in zip(samples, samples[1:])))
    check(directory.name + ':sample_fields', all(len(row['gpus']) == 2 and [g[0] for g in row['gpus']] == [0,1] and all(len(g)==4 and all(isinstance(v,int) for v in g) and g[1] >= 0 and g[2] >= 0 and 0 <= g[3] <= 100 for g in row['gpus']) for row in samples))
    sample_sum_anomalies.extend(dict(arm=directory.name, sample_index=i, epoch=row['epoch'], gpu=g[0],
        used_mib=g[1], free_mib=g[2], used_plus_free_mib=g[1]+g[2], in_measured_window=start<=row['epoch']<=end)
        for i,row in enumerate(samples) for g in row['gpus'] if g[1]+g[2] != 24127)
    recomputed = dict(started=start, finished=end, total_frames=frames,
        inference_wall_seconds=end-start, aggregate_fps=frames/(end-start),
        end_to_end_fps=frames/item['end_to_end_seconds'],
        gpu0_peak_used_mib=max(row['gpus'][0][1] for row in samples),
        gpu0_mean_utilization=sum(row['gpus'][0][3] for row in timed)/len(timed))
    for key, value in recomputed.items():
        check(directory.name + ':' + key + '_exact', item[key] == value)
    check(directory.name + ':ordered_timing_intervals', item['end_to_end_seconds'] > item['parent_makespan_seconds'] >= item['inference_wall_seconds'])
    arm_rows.append(dict(arm=directory.name, profile=profile, processes=count, opencv_threads=cv,
        log_line=log_json[index][0], **recomputed, initializations=len(selected), tracking_calls=frames-len(selected),
        end_to_end_seconds=item['end_to_end_seconds'], parent_makespan_seconds=item['parent_makespan_seconds'],
        cpu_children_seconds=item['cpu_children_seconds'], full_gpu0=sample_summary(samples,0), timed_gpu0=sample_summary(timed,0),
        full_gpu1=sample_summary(samples,1), timed_gpu1=sample_summary(timed,1),
        sample_interval_mean_seconds=statistics.mean(b['epoch']-a['epoch'] for a,b in zip(samples,samples[1:])),
        worker_start_spread_seconds=max(w['started'] for w in workers)-start,
        worker_blocks=worker_details, predictions_routes_state_canonical_sha256=digest_json(results)))

for name in plan['sequence_names']:
    row = sequence_reference[name]
    routes = row['routes']
    since_query = 0
    for route in routes:
        since_query = 0 if route['action'] else since_query+1
    per_sequence.append(dict(sequence=name, prediction_count=len(row['predictions']), route_count=len(routes),
        initial_box=row['predictions'][0], final_box=row['predictions'][-1],
        final_state_sha256=row['final_state_sha256'], reconstructed_final_frame=127, reconstructed_since_query=since_query,
        action_counts=dict(sorted(Counter(route['action'] for route in routes).items())),
        modality_counts=dict(sorted(Counter(route['modality'] for route in routes).items())),
        primary_tokens=sum(route['primary_tokens'] for route in routes), auxiliary_tokens=sum(route['auxiliary_tokens'] for route in routes),
        predictions_sha256=digest_json(row['predictions']), routes_sha256=digest_json(routes)))

remote = load(TRACE / 'REMOTE_READONLY_PROVENANCE.json')
check('remote_plan_hash', remote['plan_sha256'] == sha(PLAN_PATH))
check('remote_all26_input_binding', set(remote['inputs']) == set(plan['input_sha256']))
for name, row in remote['inputs'].items():
    check('remote_input:' + name, row['sha256'] == row['expected'] == plan['input_sha256'][name])
for name, expected in remote['artifacts'].items():
    actual = sha(ROOT / name)
    hashes[name] = actual
    check('remote_archived_copy:' + name, actual == expected)
check('TRAIN882_DEV97_scope', remote['train_sequences'] == 882 and remote['dev_sequences'] == 97 and not remote['selected_dev_overlap'])
check('TRAIN_selected_order', [r['name'] for r in remote['selected_train']] == plan['sequence_names'])
for row in remote['selected_train']:
    name = row['name']
    initial = [struct.unpack('f',struct.pack('f',v))[0] for v in row['installed_initial_box']]
    check('installed_initial_box:' + name, initial == row['manifest_initial_box'] == sequence_reference[name]['predictions'][0])
    check('installed_frame_paths:' + name, row['image_paths_present'] and row['rgb_frames'] == row['tir_frames'] == row['boxes_frames'] and row['rgb_frames'] >= 128 and len(row['rgb_prefix']) == len(set(row['rgb_prefix'])) == len(row['tir_prefix']) == len(set(row['tir_prefix'])) == 128)

best = max(arm_rows[:10], key=lambda row:row['aggregate_fps'])
check('best_argmax_excludes_profiles', (best['processes'],best['opencv_threads']) == (complete['best_processes'],complete['best_opencv_threads']))
check('unique_worker_pids', len(set(w['pid'] for w in all_workers)) == len(all_workers) == 34)
check('sequential_arm_intervals', all(a['finished'] < b['started'] for a,b in zip(arm_rows,arm_rows[1:])))
profile_summaries=[]
for index, cv in enumerate((80,1)):
    path=OUT / ('cprofile_cv%d' % cv) / 'worker0.prof'
    hashes[str(path.relative_to(ROOT)).replace('\\','/')] = sha(path)
    stats=pstats.Stats(str(path))
    functions=[dict(file=name[0],line=name[1],function=name[2],calls=value[1],self_seconds=value[2],cumulative_seconds=value[3]) for name,value in sorted(stats.stats.items(),key=lambda x:x[1][3],reverse=True)[:40]]
    check('cprofile%d_top40_exact' % cv, complete['hotspots'][index] == dict(opencv_threads=cv,functions=functions))
    interesting=[dict(file=k[0],line=k[1],function=k[2],calls=v[1],self_seconds=v[2],cumulative_seconds=v[3]) for k,v in stats.stats.items() if k[2] in ('trajectory','track','initialize','read_pair','candidate_queries') or 'imread' in k[2] or 'synchronize' in k[2]]
    check('cprofile%d_actual_track_callcount' % cv, sum(v[1] for k,v in stats.stats.items() if k[2]=='track') == 127)
    check('cprofile%d_actual_init_callcount' % cv, sum(v[1] for k,v in stats.stats.items() if k[2]=='initialize') == 1)
    check('cprofile%d_actual_read_pair_callcount' % cv, sum(v[1] for k,v in stats.stats.items() if k[2]=='read_pair') == 128)
    profile_summaries.append(dict(opencv_threads=cv, total_calls=stats.total_calls, primitive_calls=stats.prim_calls,
        total_self_seconds=stats.total_tt, interesting_calls=interesting, top10=functions[:10]))

result=dict(generated_at=datetime.now(timezone.utc).isoformat(), deterministic_checks=len(checks),
    failed=[c for c in checks if not c['passed']], checks=checks, source_hashes=sources,
    local_inputs=inputs, plan_sha256=sha(PLAN_PATH), audited_artifact_hashes=hashes, native=native,
    command_wall_seconds=native['completed']-native['started'], arms=arm_rows,
    per_sequence=per_sequence, profiles=profile_summaries,
    repeated_baseline=[dict(opencv_threads=cv, first_fps=arm_rows[first]['aggregate_fps'], repeat_fps=arm_rows[last]['aggregate_fps'],
        repeat_over_first=arm_rows[last]['aggregate_fps']/arm_rows[first]['aggregate_fps'],
        delta_percent=100*(arm_rows[last]['aggregate_fps']/arm_rows[first]['aggregate_fps']-1)) for cv,first,last in ((80,0,4),(1,5,9))],
    best=best['arm'], all_timed_frames=sum(a['total_frames'] for a in arm_rows),
    all_timed_initializations=sum(a['initializations'] for a in arm_rows),
    all_timed_tracking_calls=sum(a['tracking_calls'] for a in arm_rows),
    excluded_warmup_frames=5*len(all_workers), excluded_warmup_initializations=len(all_workers),
    excluded_warmup_tracking_calls=4*len(all_workers), worker_count=len(all_workers),
    remote_input_bindings=len(remote['inputs']), remote_archived_copies=len(remote['artifacts']),
    sample_sum_anomalies=sample_sum_anomalies,
    first_verifier_revision='Initial verifier asserted used+free was exactly24127MiB at every integer sample. One pre-timing sample sums to24128MiB. This is preserved as an observation, not a validity invariant; raw data and all peak/utilization/rate calculations unchanged. Initial script and output retained.',
    workload_action_counts=dict(sorted(Counter(route['action'] for row in sequence_reference.values() for route in row['routes']).items())),
    workload_candidate_count_distribution=dict(sorted(Counter(route['candidate_count'] for row in sequence_reference.values() for route in row['routes']).items())))
(TRACE / 'DETERMINISTIC_RECOMPUTATION.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:result[k] for k in ('deterministic_checks','failed','command_wall_seconds','repeated_baseline','best','all_timed_frames','all_timed_initializations','all_timed_tracking_calls','excluded_warmup_frames','worker_count')},indent=2))
for arm in arm_rows:
    print(json.dumps({k:arm[k] for k in ('arm','total_frames','inference_wall_seconds','aggregate_fps','end_to_end_fps','gpu0_peak_used_mib','gpu0_mean_utilization','timed_gpu1','worker_start_spread_seconds')}))

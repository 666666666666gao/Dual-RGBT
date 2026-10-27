"""Local stdlib-only checks of archived latency receipts. No model imports."""
import base64
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
PLAN = 'reports/BASELINE_E0_PAIRED_LATENCY_PLAN_20261027.json'
REVIEW = 'reports/BASELINE_E0_PAIRED_LATENCY_SOURCE_REVIEW_20261027.json'
RUN = 'runs/baseline_e0_paired_latency_20261027/'
TASK = 'runs/pipeline_tasks/profile_baseline_e0_pair_latency_20261027/'
INPUTS = [
    'tools/profile_baseline_e0_pair.py', PLAN, REVIEW,
    'reports/BASELINE_E0_PAIRED_LATENCY_LAUNCH_20261027.json',
    'reports/BASELINE_E0_PAIRED_LATENCY_FIRST_NODE_20261027.json',
    'reports/BASELINE_E0_PAIRED_LATENCY_NATIVE_20261027.json',
    'reports/PAIRED_LATENCY_E0_RESTORE_20261027.json',
    RUN + 'COMPLETE.json', RUN + 'frame_seconds.csv', RUN + 'resume.json',
    RUN + 'ownership_before_pause.json', TASK + 'status.json', TASK + 'run.log',
    'reports/SERVER26_MODEL_TOTAL_PARAMETERS_20261026.json',
    'reports/model_parameter_counts.json',
    'runs/cadtrack_baseline_latency_r2_20261027/COMPLETE.json',
    'reports/BASELINE_E0_PAIRED_LATENCY_COMPARISON_20261027.json',
]

def read(name):
    return json.loads((ROOT / name).read_bytes())

def sha(name):
    return hashlib.sha256((ROOT / name).read_bytes()).hexdigest()

def p95(values):
    ordered = sorted(values)
    index = (len(ordered) - 1) * .95
    lower = math.floor(index)
    return ordered[lower] + (ordered[lower + 1] - ordered[lower]) * (index - lower)

def stats(rows):
    result = {'frames': len(rows)}
    for label in ('decode', 'service', 'end_to_end'):
        values = [r[label + '_seconds'] for r in rows]
        total = math.fsum(values)
        result[label] = {'sum_seconds': total, 'fps': len(values) / total,
                         'mean_ms': total * 1000 / len(values), 'p95_ms': p95(values) * 1000}
    return result

plan, review = read(PLAN), read(REVIEW)
complete = read(RUN + 'COMPLETE.json')
status = read(TASK + 'status.json')
launch = read('reports/BASELINE_E0_PAIRED_LATENCY_LAUNCH_20261027.json')
first = read('reports/BASELINE_E0_PAIRED_LATENCY_FIRST_NODE_20261027.json')
native = read('reports/BASELINE_E0_PAIRED_LATENCY_NATIVE_20261027.json')
restore = read('reports/PAIRED_LATENCY_E0_RESTORE_20261027.json')
before = read(RUN + 'ownership_before_pause.json')
resume = read(RUN + 'resume.json')
counts = read('reports/model_parameter_counts.json')
current_counts = read('reports/SERVER26_MODEL_TOTAL_PARAMETERS_20261026.json')
previous_baseline = read('runs/cadtrack_baseline_latency_r2_20261027/COMPLETE.json')
comparison_report = read('reports/BASELINE_E0_PAIRED_LATENCY_COMPARISON_20261027.json')
assert plan['task'] == status['task'] == 'profile_baseline_e0_pair_latency_20261027'
assert complete['status'] == 'PAIRED_BASELINE_E0_LATENCY_COMPLETE'
assert review['status'] == 'SOURCE_PASS' and review['blocking_issues'] == 0
assert review['plan_sha256'] == complete['plan_sha256'] == sha(PLAN)
assert review['source_sha256'] == plan['source_sha256']
source_hashes = {name: {'expected': expected, 'actual': sha(name)}
                 for name, expected in plan['source_sha256'].items()}
assert len(source_hashes) == 119
assert all(v['expected'] == v['actual'] for v in source_hashes.values())
assert complete['frame_csv_sha256'] == sha(RUN + 'frame_seconds.csv')
assert plan['e0_checkpoint'] == restore['checkpoint'] == 'runs/stage_c_all882_20261013/latest.pth'
assert plan['input_sha256'][plan['e0_checkpoint']] == restore['sha256'] == 'ee60704e7fe1e281584430332dea7807a335bd4d723a769e4783217b0ac6f56c'
assert plan['input_sha256'][plan['baseline_checkpoint']] == previous_baseline['checkpoint_sha256'] == counts['baseline_before_sha256']
noncheckpoint_inputs = {}
for name, expected in plan['input_sha256'].items():
    if name in (plan['baseline_checkpoint'], plan['e0_checkpoint']):
        continue
    exists = (ROOT / name).is_file()
    noncheckpoint_inputs[name] = {'available_locally': exists, 'expected_sha256': expected}
    if exists:
        actual = sha(name)
        assert actual == expected
        noncheckpoint_inputs[name]['actual_sha256'] = actual

source_trace = '.aris/traces/experiment-audit/2026-10-27_baseline_e0_pair_latency/'
assert (ROOT / REVIEW).read_bytes() == (ROOT / (source_trace + 'round2.review.json')).read_bytes()
assert (ROOT / PLAN).read_bytes() == (ROOT / (source_trace + 'round2.plan.json')).read_bytes()
assert (ROOT / 'tools/profile_baseline_e0_pair.py').read_bytes() == (ROOT / (source_trace + 'round2.profile_baseline_e0_pair.py')).read_bytes()
assert (ROOT / REVIEW).read_bytes() == (ROOT / (source_trace + 'round2.source_guard_payload.json')).read_bytes()
source_trace_hashes = {str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (ROOT / source_trace).iterdir() if p.is_file()}
compile((ROOT / 'tools/profile_baseline_e0_pair.py').read_bytes(), 'profile_baseline_e0_pair.py', 'exec')

assert status == first['status'] == native['native']
assert base64.b64decode(first['status_base64']) == (ROOT / (TASK + 'status.json')).read_bytes()
assert status['state'] == 'COMPLETE' and status['exit_code'] == 0
assert status['physical_gpu'] == '0' and plan['physical_gpu'] == 0
assert status['pid'] == launch['pid'] == 526316
expected_command = ['/home/gaob/Dual-RGBT/conda-envs/dual_rgbt/bin/python',
                    '/home/gaob/Dual-RGBT/tools/profile_baseline_e0_pair.py', '--plan', PLAN]
assert status['command'] == launch['command'][7:] == expected_command
assert launch['command'][1:7] == ['/home/gaob/Dual-RGBT/scripts/run_task_four_gpu.py',
                                 '--task', plan['task'], '--gpu', '0', '--']
assert launch['launched'] <= status['started'] < status['completed'] < native['observed']
assert first['process_exists'] is False
log = (ROOT / (TASK + 'run.log')).read_text(encoding='utf-8')
assert first['log_tail'] == log
log_json = [json.loads(line) for line in log.splitlines() if line.startswith('{')]
assert log_json[:-1] == complete['phases'] and log_json[-1] == complete
assert 'Traceback' not in log

with (ROOT / (RUN + 'frame_seconds.csv')).open(newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    assert reader.fieldnames == ['phase', 'sequence', 'frame', 'decode_seconds', 'service_seconds', 'end_to_end_seconds']
    rows = list(reader)
for row in rows:
    row['frame'] = int(row['frame'])
    for label in ('decode_seconds', 'service_seconds', 'end_to_end_seconds'):
        row[label] = float(row[label])
        assert math.isfinite(row[label]) and row[label] > 0
phase_names = ['baseline_before', 'e0_cost0', 'e0_cost0p05', 'baseline_after']
seqs = ['bluegirl', 'bike2', "comingboy'shead"]
assert plan['sequence_names'] == seqs
assert plan['warmup_frames'] == 20 and plan['measured_frames'] == 200
assert [p['name'] for p in plan['phases']] == phase_names
assert [(p['model'], p['cost_weight']) for p in plan['phases']] == [('baseline', None), ('e0', 0.0), ('e0', .05), ('baseline', None)]
assert [p['name'] for p in complete['phases']] == phase_names
expected_keys = [(phase, sequence, frame) for phase in phase_names for sequence in seqs for frame in range(21, 221)]
actual_keys = [(r['phase'], r['sequence'], r['frame']) for r in rows]
assert actual_keys == expected_keys
assert len(rows) == len(set(actual_keys)) == plan['expected_frames'] == 2400
residual = max(abs(r['decode_seconds'] + r['service_seconds'] - r['end_to_end_seconds']) for r in rows)
assert residual <= 1e-12
recomputed = {}
metric_diffs = []
by_sequence = {}
for p in complete['phases']:
    selected = [r for r in rows if r['phase'] == p['name']]
    calculated = stats(selected)
    assert len(selected) == p['frames'] == 600
    assert len(p['initialization_seconds']) == 3 and all(math.isfinite(v) and v > 0 for v in p['initialization_seconds'])
    for label in ('service', 'end_to_end'):
        for metric in ('fps', 'mean_ms', 'p95_ms'):
            diff = abs(calculated[label][metric] - p[label][metric])
            metric_diffs.append(diff)
            assert math.isclose(calculated[label][metric], p[label][metric], rel_tol=1e-12, abs_tol=1e-12)
    recomputed[p['name']] = calculated
    by_sequence[p['name']] = {s: stats([r for r in selected if r['sequence'] == s]) for s in seqs}
baseline_pooled = stats([r for r in rows if r['phase'] in ('baseline_before', 'baseline_after')])
assert baseline_pooled['frames'] == 1200
bracket_drift = {label: {metric + '_after_vs_before_percent':
                         (recomputed['baseline_after'][label][metric] / recomputed['baseline_before'][label][metric] - 1) * 100
                         for metric in ('fps', 'mean_ms', 'p95_ms')}
                 for label in ('service', 'end_to_end')}
comparisons = {}
for name in ('e0_cost0', 'e0_cost0p05'):
    comparisons[name + '_vs_pooled_baseline'] = {
        label: {'baseline_fps_divided_by_e0_fps': baseline_pooled[label]['fps'] / recomputed[name][label]['fps'],
                'e0_fps_as_percent_of_baseline': 100 * recomputed[name][label]['fps'] / baseline_pooled[label]['fps'],
                'e0_mean_latency_increase_percent': (recomputed[name][label]['mean_ms'] / baseline_pooled[label]['mean_ms'] - 1) * 100}
        for label in ('service', 'end_to_end')}
comparisons['e0_cost0p05_vs_e0_cost0'] = {
    label: {'fps_increase_percent': (recomputed['e0_cost0p05'][label]['fps'] / recomputed['e0_cost0'][label]['fps'] - 1) * 100,
            'mean_latency_reduction_percent': (1 - recomputed['e0_cost0p05'][label]['mean_ms'] / recomputed['e0_cost0'][label]['mean_ms']) * 100,
            'p95_reduction_percent': (1 - recomputed['e0_cost0p05'][label]['p95_ms'] / recomputed['e0_cost0'][label]['p95_ms']) * 100}
    for label in ('service', 'end_to_end')}
comparisons['new_pooled_baseline_vs_previous_unpaired_baseline'] = {
    label: {'fps_change_percent': (baseline_pooled[label]['fps'] / previous_baseline[label]['fps'] - 1) * 100,
            'scope': 'Different acquisition windows; descriptive drift only, no optimization attribution.'}
    for label in ('service', 'end_to_end')}

assert complete['resume'] == resume
assert len(before) == len(native['resumed_owner_progress']) == 2
assert {str(p) for b in before for p in (b['native']['pid'], b['child_pid'])} == set(resume['states'])
assert all(v in ('R', 'S') for v in resume['states'].values())
assert 0 < resume['paused_seconds'] < status['completed'] - status['started']
progress = []
for b, after in zip(before, native['resumed_owner_progress']):
    assert b['native'] == after['native']
    assert b['native']['physical_gpu'] == '0' and b['native']['state'] == 'RUNNING'
    assert b['child_pid'] == after['child_pid']
    assert b['finite_logged_losses'] is True and after['finite_logged_losses'] is True
    assert int(b['last_update']['step']) == b['logged_updates']
    assert int(after['last_update']['step']) == after['logged_updates']
    assert after['logged_updates'] > b['logged_updates']
    for snapshot in (b, after):
        assert all(math.isfinite(float(v)) for k, v in snapshot['last_update'].items() if k == 'loss' or k.endswith('_loss'))
    progress.append({'task': b['native']['task'], 'wrapper_pid': b['native']['pid'], 'child_pid': b['child_pid'],
                     'before_updates': b['logged_updates'], 'after_updates': after['logged_updates'],
                     'increase': after['logged_updates'] - b['logged_updates']})

# Integer formulas from the source definitions; no torch/model import or checkpoint access.
conv = lambda i, o, k: i * o * k * k + o
linear = lambda i, o: i * o + o
dim = 768
block = 2 * (2 * dim) + linear(dim, 3 * dim) + linear(dim, dim) + linear(dim, 4 * dim) + linear(4 * dim, dim)
backbone = conv(3, dim, 16) + dim + 197 * dim + 12 * block + 2 * dim + (64 + 256 + 1) * dim
decode = conv(1536, 768, 3) + 2 * 768
head_stem = sum(conv(i, o, 3) + 2 * o for i, o in ((768, 256), (256, 128), (128, 64), (64, 32)))
head = 3 * head_stem + conv(32, 1, 1) + 2 * conv(32, 2, 1)
preview = conv(3, 16, 3) + conv(16, 32, 3) + conv(32, 32, 3)
quality = linear(74, 64) + linear(64, 1)
gain = linear(1690, 128) + linear(128, 1)
query_gain = linear(955, 64) + linear(64, 1)
components = {'backbone': backbone, 'decode': decode, 'box_head': head,
              'preview': preview, 'quality': quality, 'gain': gain}
assert components == counts['ours_components']
baseline_total = backbone + decode + head
e0_total = sum(components.values())
current_total = e0_total - gain + query_gain
assert baseline_total == counts['baseline_total_parameters'] == 103138437
assert e0_total == counts['ours_total_parameters'] == 103374215
assert current_total == current_counts['total_parameters'] == 103218887
assert current_counts['fps'] is None
for p in complete['phases']:
    assert p['total_parameters'] == (baseline_total if p['model'] == 'baseline' else e0_total)
original_source = (ROOT / 'dual_rgbt/model.py').read_bytes().replace(
    b"checkpoint=Path(__file__).resolve().parents[1] / 'checkpoints/CADTrack_ep0014.pth.tar'",
    b"checkpoint='/home/gaob/CADTrack_ep0014.pth.tar'")
assert hashlib.sha256(original_source).hexdigest() == counts['source_sha256']['dual_rgbt/model.py']

assert comparison_report['csv_sha256'] == sha(RUN + 'frame_seconds.csv')
assert comparison_report['baseline']['total_parameters'] == baseline_total
assert comparison_report['baseline']['frames'] == baseline_pooled['frames']
assert comparison_report['ours_e0_cost0'] == complete['phases'][1]
assert comparison_report['ours_e0_cost0p05'] == complete['phases'][2]
assert comparison_report['current_in_training']['total_parameters'] == current_total
assert comparison_report['current_in_training']['fps'] is None
assert comparison_report['scope'] == complete['scope']
for label in ('service', 'end_to_end'):
    for metric in ('fps', 'mean_ms', 'p95_ms'):
        assert math.isclose(comparison_report['baseline'][label][metric], baseline_pooled[label][metric], rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(comparison_report['baseline_bracket_fps_drift_percent'][label], bracket_drift[label]['fps_after_vs_before_percent'], rel_tol=1e-12, abs_tol=1e-12)

result = {
    'status': 'PASS', 'python': sys.version, 'stdlib_only': True,
    'audited_input_sha256': {name: sha(name) for name in INPUTS},
    'source_hashes': source_hashes, 'source_count': len(source_hashes),
    'source_trace_hashes': source_trace_hashes,
    'noncheckpoint_inputs': noncheckpoint_inputs,
    'checkpoint_payloads_opened': False, 'project_modules_imported': False,
    'remote_or_gpu_access': False,
    'rows': len(rows), 'groups': 12, 'group_size': 200, 'frame_range': [21, 220],
    'exact_expected_row_order': True, 'duplicate_keys': 0,
    'max_decode_plus_service_residual_seconds': residual,
    'max_recomputed_reported_metric_absolute_difference': max(metric_diffs),
    'quantile_definition': 'Sorted raw durations, linear interpolation at (N-1)*0.95, NumPy default percentile convention.',
    'phase_statistics': recomputed, 'phase_sequence_statistics': by_sequence,
    'pooled_baseline': baseline_pooled, 'bracket_drift': bracket_drift,
    'comparisons': comparisons, 'resumed_progress': progress,
    'native_duration_seconds': status['completed'] - status['started'],
    'paused_seconds': resume['paused_seconds'],
    'post_complete_observation_delay_seconds': native['observed'] - status['completed'],
    'total_parameter_source_arithmetic': {'components': components, 'backbone_block': block,
        'baseline_total': baseline_total, 'e0_total': e0_total, 'query_gain': query_gain,
        'current_architecture_total': current_total, 'e0_parameter_delta': e0_total-baseline_total,
        'e0_parameter_delta_percent': 100*(e0_total/baseline_total-1),
        'current_parameter_delta': current_total-baseline_total,
        'current_parameter_delta_percent': 100*(current_total/baseline_total-1),
        'scope': 'Source arithmetic corroborates runtime total enumerations. Buffers excluded. No trainable parameter count performed.'},
    'reviewed_source_trace_exact': True,
    'native_receipts_and_complete_log_exact': True,
    'comparison_report_all_values_match_raw_csv': True,
}
(OUT / 'deterministic_result_checks.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps({k: result[k] for k in ('status', 'rows', 'source_count', 'max_decode_plus_service_residual_seconds',
    'max_recomputed_reported_metric_absolute_difference', 'pooled_baseline', 'bracket_drift', 'comparisons',
    'resumed_progress', 'total_parameter_source_arithmetic', 'noncheckpoint_inputs')}, indent=2))

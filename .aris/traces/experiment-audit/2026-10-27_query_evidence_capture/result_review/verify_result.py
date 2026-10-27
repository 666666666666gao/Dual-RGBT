"""Independent local artifact verification; standard library only, no NN or remote IO."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import struct

ROOT = Path('C:/Users/gb/projects/Dual-RGBT')
OUT = Path(__file__).resolve().parent
PLAN = 'reports/QUERY_EVIDENCE_CAPTURE_ADMISSION_PLAN_20261027.json'
RESULT = 'runs/query_evidence_capture_admission_20261027/COMPLETE.json'
STATUS = 'runs/pipeline_tasks/check_query_evidence_capture_20261027/status.json'
LOG = 'runs/pipeline_tasks/check_query_evidence_capture_20261027/run.log'
NATIVE = 'reports/QUERY_EVIDENCE_CAPTURE_NATIVE_20261027.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name):
    return json.loads((ROOT / name).read_bytes())


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def float_bytes(values):
    return b''.join(struct.pack('<f', value) for value in values)


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


plan, result, native, receipt = map(read, (PLAN, RESULT, STATUS, NATIVE))
review = read(plan['source_review'])
assert sha(ROOT / PLAN) == result['plan_sha256'] == review['plan_sha256'] == '377a5f449481e876c737b01a91e197befdbfc7ef655fbb5e0f317399b3257ba2'
assert sha(ROOT / RESULT) == 'e70542a2fc764e9781d3c5d54626734830f3aac11e6ad9f0c61cd6558e5d58d3'
assert review['status'] == 'SOURCE_PASS' and review['blocking_issues'] == 0
assert review['source_sha256'] == plan['source_sha256']
assert all(sha(ROOT / n) == h for n, h in plan['source_sha256'].items())
assert plan['checkpoint'] == 'runs/global_candidate_policy_formal_20261014/latest.pth'
assert result['checkpoint_sha256'] == plan['input_sha256'][plan['checkpoint']] == 'b84d00454717cae56048753e0fe54c1d6b92cfcd2820c01fa65296508488ccb3'
provider = read('reports/SERVER25_JOINT_EVALUATION_PROTOCOL_20261027.json')['provider']
assert provider['project'] == '/home/gaob/Dual-RGBT'
assert native['command'] == [provider['python'], provider['project'] + '/tools/check_query_evidence_capture.py', '--plan', PLAN]
assert native['task'] == plan['task'] == 'check_query_evidence_capture_20261027'
assert native['physical_gpu'] == str(plan['physical_gpu']) == '0'
assert native['state'] == 'COMPLETE' and native['exit_code'] == receipt['exit_code'] == 0
assert isinstance(native['pid'], int) and native['pid'] > 0
assert native['completed'] > native['started'] and receipt['seconds'] > 0
log_summaries = [json.loads(line) for line in (ROOT / LOG).read_text().splitlines() if line.startswith('{')]
assert log_summaries == [{k: v for k, v in result.items() if k != 'rows'}]
assert result['status'] == 'CAPTURE_REAL_TRAJECTORY_PASS'
assert result['parameters_unchanged'] is True and result['optimizer_updates'] == 0
assert result['predictions_routes_templates_history_exact'] is True

expected_keys = {(observed, cost, sequence) for observed in (False, True)
                 for cost in plan['costs'] for sequence in plan['sequence_names']}
rows = result['rows']
keys = [(r['observed'], r['cost'], r['sequence']) for r in rows]
assert len(keys) == len(set(keys)) == len(expected_keys) == 12
assert set(keys) == expected_keys
by_key = dict(zip(keys, rows))
route_keys = {'modality', 'action', 'request_index', 'candidate_rank', 'candidate_count',
              'confidence', 'primary_tokens', 'auxiliary_tokens'}
evidence_keys = {'frame', 'primary_x', 'primary_y', 'primary_w', 'primary_h',
                 'primary_confidence', 'selected_gain', 'selected_token_cost', 'selected_utility'}
summaries = []
stop_boxes = query_boxes_different = 0
counts = {False: Counter(), True: Counter()}
max_cost_error = max_utility_error = 0.
for row in rows:
    assert isinstance(row['observed'], bool)
    assert len(row['boxes']) == plan['frames_per_sequence'] == 11
    assert len(row['routes']) == 10
    assert len(row['evidence']) == (10 if row['observed'] else 0)
    assert len(row['template_sha256']) == 64 and int(row['template_sha256'], 16) >= 0
    assert len(row['history']) == 1 and len(row['history'][0]) == 8
    assert all(math.isfinite(v) and f32(v) == v for v in row['history'][0])
    for box in row['boxes']:
        assert len(box) == 4 and all(math.isfinite(v) and f32(v) == v for v in box)
        assert box[2] > 0 and box[3] > 0
    for route in row['routes']:
        assert set(route) == route_keys
        assert route['action'] in range(5) and route['modality'] in (0, 1)
        assert route['primary_tokens'] == 385
        assert route['auxiliary_tokens'] == (0 if route['action'] == 0 else 385 if route['action'] == 4 else 193)
        assert 0 <= route['candidate_count'] <= plan['max_candidates']
        assert math.isfinite(route['confidence'])
        counts[row['observed']][route['action']] += 1
    if not row['observed']:
        continue
    plain = by_key[(False, row['cost'], row['sequence'])]
    for field in ('boxes', 'routes', 'template_sha256', 'history'):
        assert row[field] == plain[field], (row['cost'], row['sequence'], field)
    assert all(float_bytes(a) == float_bytes(b) for a, b in zip(row['boxes'], plain['boxes']))
    assert float_bytes(row['history'][0]) == float_bytes(plain['history'][0])
    assert [e['frame'] for e in row['evidence']] == list(range(1, 11))
    for frame, (route, evidence) in enumerate(zip(row['routes'], row['evidence']), 1):
        assert set(evidence) == evidence_keys
        primary = [evidence['primary_' + axis] for axis in ('x', 'y', 'w', 'h')]
        assert all(math.isfinite(v) for v in evidence.values())
        assert primary[2] > 0 and primary[3] > 0
        token_cost = f32(route['auxiliary_tokens'] / 385.)
        utility = f32(evidence['selected_gain'] - f32(row['cost'] * token_cost))
        max_cost_error = max(max_cost_error, abs(evidence['selected_token_cost'] - token_cost))
        max_utility_error = max(max_utility_error, abs(evidence['selected_utility'] - utility))
        assert evidence['selected_token_cost'] == token_cost
        assert evidence['selected_utility'] == utility
        if route['action'] == 0:
            assert route['request_index'] == route['candidate_rank'] == -1
            assert evidence['selected_gain'] == evidence['selected_token_cost'] == evidence['selected_utility'] == 0
            assert float_bytes(primary) == float_bytes(row['boxes'][frame])
            assert evidence['primary_confidence'] == route['confidence']
            stop_boxes += 1
        else:
            assert route['request_index'] >= 0 and evidence['selected_utility'] > 0
            assert route['request_index'] < 2 + 2 * route['candidate_count']
            assert route['action'] == 1 and route['request_index'] == 0 and route['candidate_rank'] == -1
            query_boxes_different += (float_bytes(primary) != float_bytes(row['boxes'][frame]))
    summaries.append(dict(sequence=row['sequence'], cost=row['cost'],
        tracked_frames=10, action_counts=dict(Counter(r['action'] for r in row['routes'])),
        boxes_float32_bytes_equal=True, routes_equal=True, history_float32_bytes_equal=True,
        recorded_template_sha256_equal=True, template_sha256=row['template_sha256']))

assert counts[False] == counts[True] == Counter({0: 50, 1: 10})
assert result['stop_frames'] == stop_boxes == 50
assert result['query_frames'] == query_boxes_different == 10
assert max_cost_error == max_utility_error == 0
for sequence in plan['sequence_names']:
    initial = by_key[(False, plan['costs'][0], sequence)]['boxes'][0]
    assert all(row['boxes'][0] == initial for row in rows if row['sequence'] == sequence)

now = datetime.now(timezone.utc).isoformat()
evidence = dict(status='PASS', generated_at=now, result_sha256=sha(ROOT / RESULT),
    rows=12, paired_trajectories=6, sequences=plan['sequence_names'], costs=plan['costs'],
    frames_per_trajectory=11, boxes_compared_including_initialization=66,
    routes_compared=60, final_histories_compared=6, final_template_hash_pairs_compared=6,
    observed_evidence_records=60, plain_evidence_records=0,
    observed_stop_frames=50, observed_query_frames=10,
    plain_stop_frames=50, plain_query_frames=10,
    stop_primary_final_float32_byte_equal=50,
    query_primary_final_boxes_different=10,
    maximum_token_cost_error=0., maximum_float32_utility_error=0.,
    row_checks=summaries, native_task=native['task'], native_pid=native['pid'],
    native_command=native['command'], physical_gpu=native['physical_gpu'],
    native_state=native['state'], native_exit_code=native['exit_code'],
    native_elapsed_seconds=native['completed']-native['started'],
    wrapper_exit_code=receipt['exit_code'], wrapper_elapsed_seconds=receipt['seconds'],
    log_summary_equals_complete_summary=True, all_source_hashes_match=True,
    source_hash_count=len(plan['source_sha256']), installed_source_guard_matches=True,
    template_bytes_rehashed_by_reviewer=False, checkpoint_bytes_read_by_reviewer=False,
    reviewer_neural_network_execution=False, reviewer_remote_queries=False)
save('DETERMINISTIC_EVIDENCE.json', evidence)
inputs = [PLAN, RESULT, STATUS, LOG, NATIVE, plan['source_review'],
          'tools/check_query_evidence_capture.py', 'scripts/query_evidence_capture.py',
          'reports/SERVER25_JOINT_EVALUATION_PROTOCOL_20261027.json']
report = dict(audit_skill='experiment-audit', audit_phase='RESULT', status='RESULT_PASS',
    blocking_issues=0, issues=[], generated_at=now,
    result=RESULT, result_sha256=sha(ROOT/RESULT),
    plan=PLAN, plan_sha256=sha(ROOT/PLAN),
    checkpoint=plan['checkpoint'], checkpoint_sha256=result['checkpoint_sha256'],
    source_sha256=plan['source_sha256'], audited_input_hashes={n:sha(ROOT/n) for n in inputs},
    native_exit_code=0, deterministic_evidence='DETERMINISTIC_EVIDENCE.json',
    deterministic_evidence_sha256=sha(OUT/'DETERMINISTIC_EVIDENCE.json'),
    requested_reviewer_model='gpt-6-astra', requested_reviewer_reasoning_effort='max',
    actual_model='UNATTESTED', reviewer_model='UNATTESTED', actual_reasoning_effort='UNATTESTED',
    review_independence='same-family', acceptance_status='provisional',
    agent_task_name='/root/audit_query_evidence_capture_20261027',
    summary='Six plain/observed trajectory pairs match exactly for saved boxes/routes/history and recorded final-template hashes. All sixty observation rows validate: fifty STOP and ten QUERY frames; native exit zero and exact plan/checkpoint/source bindings pass.',
    scope='Real fixed historical candidate_b step-6250 TRAIN observation equivalence on three eleven-frame prefixes and two fixture costs only.',
    limitations=[
        'All ten QUERY frames use bluegirl at cost zero and action 1 LOCAL_MAIN; actions 2/3/4 and additional data/weights are not covered.',
        'Template tensors are not retained in COMPLETE.json: the reviewer compares the recorded SHA256 values, whose raw-byte computation is established by the bound checker source.',
        'Dataset and historical checkpoint bytes were checked by the source-bound successful native checker; the reviewer did not reread them or execute a model.',
        'No new full-model accuracy, signed-IoU benefit, FPS, full benchmark coverage, or complete formal evaluator/CSV runtime validation is established.'],
    reviewer_neural_network_execution=False, reviewer_remote_queries=False)
save('RESULT_REVIEW.json', report)
text = ('# Query evidence capture result audit\n\n'
        '**RESULT_PASS; blocking_issues = 0.**\n\n'
        'Six paired trajectories agree exactly for 66 saved boxes, 60 routes, six final histories and six recorded final-template SHA256 pairs. '
        'The 60 evidence rows contain 50 STOP and 10 QUERY frames. Every STOP primary box equals its final box at float32 byte level; '
        'all 10 QUERY primary boxes differ from final boxes. Token-cost and float32-utility errors are zero.\n\n'
        'Native task completed with exit 0 on server25 GPU0. Command, plan/checkpoint hashes, log summary and every bound source hash match. '
        'Exact result SHA256: `' + report['result_sha256'] + '`.\n\n'
        'Scope: fixed historical B6250, three real TRAIN prefixes, eleven frames each, costs 0 and 100. '
        'QUERY coverage is action 1 on bluegirl only. Templates were compared through stored SHA256 values; raw tensors were not retained. '
        'No new full-model accuracy or complete formal evaluator runtime is asserted.\n\n'
        'Requested Astra/max; actual model/effort UNATTESTED; same-family/provisional. Reviewer performed local standard-library checks only.\n')
(OUT/'RESULT_REVIEW.md').write_text(text, encoding='utf-8')
(OUT/'001-result-review.response.md').write_text(text, encoding='utf-8')
save('001-result-review.request.json', dict(purpose='result-review', tool='followup_task',
    prompt='Real admission completed native0. Independently audit local runs/query_evidence_capture_admission_20261027/COMPLETE.json, runs/pipeline_tasks/check_query_evidence_capture_20261027/{status.json,run.log}, reports/QUERY_EVIDENCE_CAPTURE_NATIVE_20261027.json, final admission plan and source. Recompute raw plain vs observed boxes/routes/templates/history equality, genuine STOP+QUERY counts, evidence primary=final onSTOP, token cost/utility and record counts; verify native command/task/plan/checkpoint hashes and sources. No NN/remote. Save concise RESULT_PASS/FAIL and deterministic evidence under .aris/traces/experiment-audit/2026-10-27_query_evidence_capture/result_review/. Scope real fixedB6250TRAIN equivalence only, not newfullmodel accuracy nor complete formal evaluator runtime. Need exact result JSON path for publication gate; samefamily/provisionalUNATTESTED.'))
save('001-result-review.meta.json', dict(purpose='result-review', generated_at=now,
    status='ok', agent_task_name=report['agent_task_name'], requested_model='gpt-6-astra',
    requested_reasoning_effort='max', actual_model='UNATTESTED',
    review_independence='same-family', acceptance_status='provisional'))
print(json.dumps({k:report[k] for k in ('status','blocking_issues','result_sha256','plan_sha256','checkpoint_sha256')}))
print(str(OUT/'RESULT_REVIEW.json'))

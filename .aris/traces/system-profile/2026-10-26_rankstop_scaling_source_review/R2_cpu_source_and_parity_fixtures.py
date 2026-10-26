"""CPU-only source binding, workload and AST-isolated trajectory checks."""
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np

ROOT = Path(r'C:\Users\gb\projects\Dual-RGBT')
OUT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


plan_path = ROOT / 'reports/RANKSTOP_INFERENCE_SCALING_PLAN_20261026.json'
plan = json.loads(plan_path.read_bytes())
old = json.loads((ROOT / 'reports/rankstop_dev_plan_server26_20261026.json').read_bytes())
script = ROOT / 'scripts/profile_rankstop_scaling.py'
assert sha(plan_path) == 'cb64752e50fea18da27eb9be6451cbcfd01c7561fdfcab6707409f6960651f6b'
assert sha(script) == '3ffa29862ec8e881bdd98b2dd0d8ee8f159160c39d4a9fcc6b90c08a99b95a56'
tree = ast.parse(script.read_bytes(), feature_version=(3, 7))
nodes = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
checks = []


def passed(name, **details):
    checks.append(dict(name=name, status='PASS', **details))


assert len(plan['source_sha256']) == 49 and len(old['source_sha256']) == 48
assert all(plan['source_sha256'][name] == digest for name, digest in old['source_sha256'].items())
assert set(plan['source_sha256']) - set(old['source_sha256']) == {'scripts/profile_rankstop_scaling.py'}
for name, digest in plan['source_sha256'].items():
    assert sha(ROOT / name) == digest, name
    ast.parse((ROOT / name).read_bytes(), feature_version=(3, 7))
passed('all49_source_hashes_and_python37_syntax', unchanged_inherited_sources=48)

input_facts = []
for name, digest in plan['input_sha256'].items():
    path = ROOT / name
    if path.exists():
        actual = sha(path)
        assert actual == digest, name
        input_facts.append(dict(path=name, status='MATCH', sha256=actual))
    else:
        input_facts.append(dict(path=name, status='NOT_PRESENT_LOCAL', planned_sha256=digest))
passed('all_locally_present_bound_inputs_match', present=sum(row['status'] == 'MATCH' for row in input_facts),
       absent=sum(row['status'] == 'NOT_PRESENT_LOCAL' for row in input_facts))
assert plan['checkpoint'] == old['checkpoint']
assert plan['checkpoint_sha256'] == old['checkpoint_sha256'] == plan['input_sha256'][plan['checkpoint']]
assert plan['factory_checkpoint'] == old['factory_checkpoint']
assert plan['input_sha256'][plan['factory_checkpoint']] == old['input_sha256'][old['factory_checkpoint']]
assert plan['input_sha256']['checkpoints/CADTrack_ep0014.pth.tar'] == old['protected_checkpoint_sha256']
passed('checkpoint_factory_pretrained_bindings_identical_to_existing_dev')

terminal = json.loads((ROOT / plan['completed_dev_status']).read_bytes())
observation = json.loads((ROOT / 'reports/RANKSTOP_COST0_TERMINAL_OBSERVATION_20261026.json').read_bytes())
assert terminal == observation['native']
assert terminal['state'] == 'COMPLETE' and terminal['exit_code'] == 0 and terminal['physical_gpu'] == '0'
assert observation['complete']['checkpoint_sha256'] == plan['checkpoint_sha256']
assert observation['complete']['plan_sha256'] == sha(ROOT / 'reports/rankstop_dev_plan_server26_20261026.json')
assert observation['complete']['frames'] == 54646 and observation['complete']['sequences'] == 97
passed('local_actual_cost0_native_receipt_matches_terminal_observation')

training = json.loads((ROOT / 'reports/global_candidate_rankstop_trajectory_plan_20261018.json').read_bytes())
dev = json.loads((ROOT / 'data/benchmarks/server26/lasher_dev.json').read_bytes())
streams = json.loads((ROOT / 'reports/RANKSTOP_SCALING_TRAIN_STREAMS_20261026.json').read_bytes())
assert [row['name'] for row in streams] == plan['sequence_names']
assert len(set(plan['sequence_names'])) == 8 and plan['frames_per_sequence'] == 128
assert all(row['frames'] >= 128 for row in streams)
assert set(plan['sequence_names']) <= set(training['sequence_names'])
assert not set(plan['sequence_names']) & {row['name'] for row in dev['sequences']}
assert len(training['sequence_names']) == 882
passed('eight_fixed_streams_in_frozen882_train_disjoint_from97_dev', minimum_stream_frames=min(row['frames'] for row in streams))
expected_arms = [dict(processes=n, opencv_threads=cv) for cv in (80, 1) for n in (1, 2, 4, 8, 1)]
assert plan['arms'] == expected_arms and plan['cost_weight'] == .05
for arm in plan['arms']:
    groups = [plan['sequence_names'][i::arm['processes']] for i in range(arm['processes'])]
    assert sorted(name for group in groups for name in group) == sorted(plan['sequence_names'])
    assert all(groups) and sum(len(group) * 128 for group in groups) == 1024
passed('ten_arms_equal1024_frames_exactly_once_with_order_preserved')


class OnlyInitialBox:
    def __init__(self):
        self.reads = []

    def __getitem__(self, key):
        assert key == 0, 'noninitial ground truth accessed'
        self.reads.append(key)
        return [1, 2, 10, 11]


class TensorCPU:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=np.float32)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class FakeTracker:
    def __init__(self, *, state_offset=0, template_offset=0, history_offset=0, seconds=1):
        self.state_offset, self.template_offset, self.history_offset = state_offset, template_offset, history_offset
        self.seconds = seconds

    def initialize(self, image, box):
        self.frame = 0
        self.state = box.copy()
        self.templates = TensorCPU([self.template_offset, 1])
        self.history = TensorCPU([self.history_offset, 2])

    def track(self, image):
        self.frame += 1
        assert image[1] == self.frame
        self.state = np.array([self.frame, self.frame + 1, 10, 11], dtype=np.float32)
        output = self.state.copy()
        self.state[0] += self.state_offset
        self.history = TensorCPU([self.history_offset, self.frame])
        return output, dict(modality=self.frame % 2, action=self.frame % 5,
            request_index=self.frame % 8 - 1, candidate_rank=self.frame % 8,
            candidate_count=8, confidence=.75, primary_tokens=385,
            auxiliary_tokens=193, seconds=self.seconds)


namespace = {'hashlib': hashlib}
exec(compile(ast.Module(body=[nodes['trajectory']], type_ignores=[]), '<reviewed-trajectory>', 'exec'), namespace)
trajectory = namespace['trajectory']
reads = []


def read_pair(sequence, frame):
    reads.append((sequence['name'], frame))
    return sequence['name'], frame


def run_sequence(name, tracker):
    return trajectory(tracker, {'name': name, 'boxes': OnlyInitialBox()}, 128, read_pair, np)


reference = {name: run_sequence(name, FakeTracker()) for name in plan['sequence_names']}
assert len(reads) == 1024
for name in plan['sequence_names']:
    assert [frame for sequence, frame in reads if sequence == name] == list(range(128))
    assert len(reference[name]['predictions']) == 128 and len(reference[name]['routes']) == 127
passed('trajectory_reads_only_initialGT_and_frames0through127_causally')
for arm in plan['arms']:
    actual = {}
    for index in range(arm['processes']):
        assigned = plan['sequence_names'][index::arm['processes']]
        tracker = FakeTracker(seconds=2)
        trajectory(tracker, {'name': assigned[0], 'boxes': OnlyInitialBox()}, 5, read_pair, np)
        for name in assigned:
            actual[name] = run_sequence(name, tracker)
    assert actual == reference
passed('trajectory_warmup_reset_and_partition_parity_all10_arms_seconds_excluded')
one = reference[plan['sequence_names'][0]]
for field in one['routes'][0]:
    changed = copy.deepcopy(one)
    changed['routes'][0][field] += 1
    assert changed != one
passed('all8_nontiming_route_fields_sensitive_to_change')
changed = copy.deepcopy(one)
changed['predictions'][10][0] += .25
assert changed != one
passed('prediction_coordinate_change_rejected')
for option in ('state_offset', 'template_offset', 'history_offset'):
    changed = run_sequence(plan['sequence_names'][0], FakeTracker(**{option: 1}))
    assert changed['predictions'] == one['predictions'] and changed['routes'] == one['routes']
    assert changed['final_state_sha256'] != one['final_state_sha256']
    passed('final_state_digest_detects_' + option)

calls = []
sample_namespace = {
    'subprocess': SimpleNamespace(check_output=lambda argv, **kwargs: (calls.append(argv) or '0, 26, 24101, 0\n1, 2328, 21799, 13\n')),
    'time': SimpleNamespace(time=lambda: 100.),
}
exec(compile(ast.Module(body=[nodes['gpu_sample']], type_ignores=[]), '<reviewed-gpu-sample>', 'exec'), sample_namespace)
sample = sample_namespace['gpu_sample']()
assert calls == [['nvidia-smi', '-i', '0,1', '--query-gpu=index,memory.used,memory.free,utilization.gpu', '--format=csv,noheader,nounits']]
assert sample['gpus'][0] == [0, 26, 24101, 0]
passed('disclosed_gpu01_memory_util_only_no_power_temperature')

projection = next(n for n in ast.walk(nodes['main']) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'projected' for t in n.targets))
admission = next(n for n in ast.walk(nodes['main']) if isinstance(n, ast.Assert) and n.msg is not None and 'Memory admission failed' in ast.unparse(n.msg))
projection_code = compile(ast.Expression(projection.value), '<reviewed-memory-projection>', 'eval')
admission_code = compile(ast.Expression(admission.test), '<reviewed-memory-admission>', 'eval')
environment = dict(arms=[{'gpu0_peak_used_mib': 2400, 'processes': 1}, {'gpu0_peak_used_mib': 4800, 'processes': 2}, {'gpu0_peak_used_mib': 9600, 'processes': 4}], count=8, memory=[0, 26, 24101, 0])
environment['projected'] = eval(projection_code, environment)
assert environment['projected'] == 19200 and eval(admission_code, environment)
for rejected in (23103, 23600):
    environment['projected'] = rejected
    assert not eval(admission_code, environment)
passed('eightworker_memory_projection_and1024MiB_headroom_boundaries', fixture_only=True)

report = dict(status='PASS', source_sha256=sha(script), plan_sha256=sha(plan_path), checks=checks,
              check_count=len(checks), input_facts=input_facts,
              limitations=['No Torch/OpenCV import, GPU/SSH/resource query, worker process or benchmark execution.',
                           'Trajectory tests use a deterministic CPU fake; actual model output equality remains a required runtime gate.',
                           'Remote-only data/weights are checked through bound source gates and existing receipts, not rehashed by this source-only review.'])
(OUT / 'R2_CPU_SOURCE_AND_PARITY_FIXTURES.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(dict(status=report['status'], checks=checks, input_counts=dict(present=sum(row['status']=='MATCH' for row in input_facts), absent=sum(row['status']=='NOT_PRESENT_LOCAL' for row in input_facts))), indent=2))

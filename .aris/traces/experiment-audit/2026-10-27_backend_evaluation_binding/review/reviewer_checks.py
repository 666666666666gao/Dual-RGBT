"""Independent local source/metadata and synthetic guard checks; no NN or weight reads."""
import ast
import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parent
TRACE = OUT.parent
TOOLKIT = Path('C:/Users/gb/Desktop/document/cadtrack/toolkit')
PROTOCOLS = [
    'reports/JOINT_CANDIDATE_EVALUATION_PROTOCOL_20261026.json',
    'reports/VRAM18_EVALUATION_PROTOCOL_20261027.json',
    'reports/SERVER25_JOINT_EVALUATION_PROTOCOL_20261027.json',
]
SOURCES = [
    'scripts/joint_evaluation_contract.py',
    'scripts/run_joint_candidate_evaluation.py',
    'tools/joint_candidate_results.py',
    'tools/register_backend_evaluation_protocols.py',
    'tools/register_joint_evaluation_protocol.py',
    'scripts/benchmark_inputs.py',
]


def sha(path):
    assert '.pth' not in str(path), 'Review must never read weights'
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def tree(path):
    source = Path(path).read_text(encoding='utf-8-sig')
    compile(source, str(path), 'exec')
    return ast.parse(source)


specs = [read(ROOT / n) for n in PROTOCOLS]
summary = dict(scope=__doc__, neural_network_runs=0, weights_read=0, remote_calls=0,
               generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               hashes={n: sha(ROOT / n) for n in PROTOCOLS + SOURCES})
summary['syntax_compilation'] = {n: bool(tree(ROOT / n)) for n in SOURCES}
summary['protocols'] = {}
for name, spec in zip(PROTOCOLS, specs):
    item = dict(provider=spec['provider'], counts={k: len(spec[k]) for k in
                ('source_sha256', 'input_sha256', 'training_plan_sha256', 'toolkit_code_sha256')})
    item['input_verification'] = {}
    for field in ('source_sha256', 'training_plan_sha256', 'input_sha256'):
        for path, expected in spec[field].items():
            p = ROOT / path
            if '.pth' in path:
                status = 'NOT_READ_WEIGHTS_PROHIBITED'
            elif not p.is_file():
                assert field == 'input_sha256'
                status = 'NOT_LOCAL_RUNTIME_GATED'
            else:
                assert sha(p) == expected, path
                status = 'PASS'
            if field == 'input_sha256':
                item['input_verification'][path] = dict(expected=expected, status=status)
    guard = ROOT / spec['source_review']
    item['source_review_path'] = spec['source_review']
    item['guard_exists'] = guard.exists()
    if guard.exists():
        g = read(guard)
        item['guard_protocol_matches'] = g['protocol_sha256'] == sha(ROOT / name)
        item['guard_source_matches'] = g['source_sha256'] == spec['source_sha256']
    summary['protocols'][name] = item

sets = [set(s['training_plan_sha256']) for s in specs]
assert [len(s) for s in sets] == [8, 24, 8]
assert sets[0] <= sets[1] and not sets[1] & sets[2]
plans = {p: read(ROOT / p) for p in sets[1] | sets[2]}
key = lambda p: (p['repair']['kind'], p['max_candidates'], p['cost_weight'], p['seed'])
keys26, keys25 = [{key(plans[p]) for p in s} for s in sets[1:]]
assert len(keys26) == 24 and len(keys25) == 8 and not keys26 & keys25
for p in plans.values():
    assert p['status'] == 'READY'
    assert len(p['sequence_names']) == len(set(p['sequence_names'])) == 882
    assert (p['seed'], p['rounds'], p['label_stride']) == (2027, 2, 5)
    assert p['expected_formal_counts'] == specs[0]['formal_counts']
index = read(ROOT / 'reports/SERVER25_REMAINING_GRID_20261027.json')
for row in index['plans']:
    assert sha(ROOT / row['path']) == row['sha256']
    assert sha(ROOT / row['original_unlaunched_plan']) == row['original_unlaunched_sha256']
    assert key(read(ROOT / row['original_unlaunched_plan'])) == key(plans[row['path']])
    assert row['original_unlaunched_plan'] not in sets[1]
summary['candidate_ownership'] = dict(original8_subset26=True, provider26=24, provider25=8,
    distinct_configuration_tuples=32, intersection=0, reassignment_pairs_verified=8)
protected = read(TRACE / 'protected_training_files.json')
assert all(sha(ROOT / p) == h for p, h in protected.items())
summary['protected_files'] = dict(count=len(protected), mismatches=[])
summary['candidate_evaluation_directory_exists'] = (ROOT / 'reports/joint_candidate_evaluation').exists()

for name in PROTOCOLS[:2]:
    previous, current = read(TRACE / 'pre_change' / name), read(ROOT / name)
    for key_name in ('input_sha256', 'toolkit_code_sha256', 'datasets', 'split_sizes',
                     'formal_counts', 'seed', 'total_parameters', 'dev_cost_candidates',
                     'reference_decision', 'initialization', 'selection_criterion'):
        assert previous[key_name] == current[key_name], (name, key_name)
summary['unchanged_protocol_rules'] = True

summary['unchanged_functions'] = {}
for name, allowed in [
    ('scripts/joint_evaluation_contract.py', {'training_metadata', 'evaluation_command', 'smoke_admission', 'verify_runs'}),
    ('scripts/run_joint_candidate_evaluation.py', {'main'}),
    ('tools/joint_candidate_results.py', {'score', 'commands'}),
]:
    before = {n.name: ast.dump(n) for n in tree(TRACE / 'pre_change' / name).body if isinstance(n, ast.FunctionDef)}
    after = {n.name: ast.dump(n) for n in tree(ROOT / name).body if isinstance(n, ast.FunctionDef)}
    assert set(before) == set(after)
    changed = {n for n in before if before[n] != after[n]}
    assert changed == allowed, (name, changed)
    summary['unchanged_functions'][name] = sorted(set(before) - changed)

# Discover local imports, including package initializer code. No imported module executes.
seen, edges = {}, []
queue = ['scripts/run_joint_candidate_evaluation.py', 'tools/joint_candidate_results.py']
def module_file(module):
    rel = module.replace('.', '/')
    for base in (ROOT, ROOT / 'scripts', ROOT / 'tools'):
        for p in (base / (rel + '.py'), base / rel / '__init__.py'):
            if p.is_file():
                return p

while queue:
    name = queue.pop()
    if name in seen:
        continue
    p = ROOT / name
    parsed = tree(p)
    seen[name] = parsed
    for node in ast.walk(parsed):
        modules = []
        if isinstance(node, ast.Import):
            modules = [a.name for a in node.names]
        if isinstance(node, ast.ImportFrom):
            if node.level:
                package = p.parent.relative_to(ROOT).parts
                module = '.'.join(package[:len(package) - node.level + 1])
                module += '.' + node.module if node.module else ''
            else:
                module = node.module or ''
            modules = [module] + [module + '.' + a.name for a in node.names]
        for module in modules:
            dep = module_file(module)
            if dep is None:
                continue
            deps = [dep]
            for parent in dep.parents:
                if parent == ROOT:
                    break
                init = parent / '__init__.py'
                if init.is_file():
                    deps.append(init)
            for dep in deps:
                rel = dep.relative_to(ROOT).as_posix()
                edges.append((name, node.lineno, rel))
                queue.append(rel)
for spec in specs:
    assert set(seen) <= set(spec['source_sha256'])
summary['local_import_graph'] = dict(file_count=len(seen), files=sorted(seen),
    edges=sorted(set(edges)), unsealed=[])

# Execute only actual pure command and metadata-guard definitions from the reviewed contract.
contract_ast = tree(ROOT / 'scripts/joint_evaluation_contract.py')
selected = [n for n in contract_ast.body if isinstance(n, ast.Assign) or
            isinstance(n, ast.FunctionDef) and n.name in ('evaluation_command', 'training_metadata')]
namespace = dict(json=json, sha=sha)
exec(compile(ast.Module(body=selected, type_ignores=[]), '<reviewed contract definitions>', 'exec'), namespace)
summary['synthetic_guard_checks'] = []
for i, spec in enumerate(specs):
    for split in ('train_smoke', 'dev', 'test'):
        cfg = dict(plan='synthetic.json', benchmark='LasHeR', split=split, task='synthetic_task',
                   run='synthetic_run', cost_weight=0.05, shard=0, shards=1,
                   smoke_frames=11 if split == 'train_smoke' else 0, selection_sha256='frozen-test-sha')
        command = namespace['evaluation_command'](cfg, spec)
        assert command[:2] == [spec['provider']['python'], spec['provider']['project'] + '/scripts/run_joint_candidate_evaluation.py']
        assert ('--selection-sha256' in command) == (split == 'test')
        summary['synthetic_guard_checks'].append(f'protocol{i}_{split}_provider_and_selection_command_PASS')
    for state, code in [('RUNNING', None), ('COMPLETE', 1)]:
        fixture = OUT / 'fixtures' / f'protocol{i}_{state}_{code}'
        fixture.mkdir(parents=True, exist_ok=True)
        plan_path = fixture / 'synthetic_plan.json'
        plan_path.write_text(json.dumps(dict(runs=dict(formal='synthetic_final'), tasks=dict(formal='synthetic_task'))))
        native = fixture / 'runs/pipeline_tasks/synthetic_task/status.json'
        native.parent.mkdir(parents=True, exist_ok=True)
        native.write_text(json.dumps(dict(state=state, exit_code=code)))
        rejected = False
        try:
            namespace['training_metadata'](fixture, dict(training_plan_sha256={'synthetic_plan.json': sha(plan_path)}), 'synthetic_plan.json')
        except AssertionError:
            rejected = True
        assert rejected and not (fixture / 'runs/synthetic_final').exists()
        summary['synthetic_guard_checks'].append(f'protocol{i}_{state}_{code}_rejected_before_complete_config_or_weight_PASS')

summary['toolkit_code_verified'] = {}
for name, expected in specs[0]['toolkit_code_sha256'].items():
    actual = sha(TOOLKIT / name)
    assert actual == expected
    assert all(s['toolkit_code_sha256'][name] == expected for s in specs)
    summary['toolkit_code_verified'][name] = actual
summary['annotation_verification'] = {}
current_score = next(n for n in tree(ROOT / 'tools/joint_candidate_results.py').body if isinstance(n, ast.FunctionDef) and n.name == 'score')
old_score = next(n for n in tree(TRACE / 'before_r2/tools/joint_candidate_results.py').body if isinstance(n, ast.FunctionDef) and n.name == 'score')
def annotation_assert(fn):
    return next(n for n in ast.walk(fn) if isinstance(n, ast.Assert) and 'annotation_file_sha256' in ast.dump(n))

for name in ('data/benchmarks/rgbt210_test.json', 'data/benchmarks/rgbt234_test.json'):
    manifest = read(ROOT / name)
    lookup = {}
    for rel, expected in manifest['annotation_file_sha256'].items():
        actual = sha(TOOLKIT / rel.replace('\\', '/'))
        assert actual == expected
        lookup[PurePosixPath('/cpu-toolkit') / rel.replace('\\', '/')] = actual
    ns = dict(sha=lookup.__getitem__, toolkit=PurePosixPath('/cpu-toolkit'), manifest=manifest)
    rejected = False
    try:
        exec(compile(ast.Module(body=[annotation_assert(old_score)], type_ignores=[]), '<R1 POSIX path lookup>', 'exec'), ns)
    except KeyError:
        rejected = True
    assert rejected
    exec(compile(ast.Module(body=[annotation_assert(current_score)], type_ignores=[]), '<R2 POSIX path lookup>', 'exec'), ns)
    summary['annotation_verification'][name] = dict(count=len(lookup), all_actual_local_hashes_match=True,
        synthetic_posix_R1='FAIL_EXPECTED', synthetic_posix_R2='PASS', manifest_sha256=sha(ROOT / name))

summary['status'] = 'LOCAL_SOURCE_METADATA_AND_SYNTHETIC_CHECKS_PASS'
(OUT / 'R2_deterministic_checks.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(status=summary['status'], protocol_counts=[x['counts'] for x in summary['protocols'].values()],
    protected_files=len(protected), import_files=len(seen), synthetic_guard_checks=len(summary['synthetic_guard_checks']),
    annotation_files=sum(x['count'] for x in summary['annotation_verification'].values()), weights_read=0, neural_network_runs=0)))

"""Write the targeted R2 source review without changing production artifacts."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

TRACE = Path(__file__).resolve().parent
ROOT = TRACE.parents[3]
PLAN_NAME = 'reports/BASELINE_E0_PAIRED_LATENCY_PLAN_20261027.json'
SOURCE_NAME = 'tools/profile_baseline_e0_pair.py'
PLAN_SHA = '6c9cf09b2d843c78bde967cb02f4c735cb5dadae6ced44a462d247292e665909'
SOURCE_SHA = '5a15d81a9575c13ac9bcfe4fa1accbe97ff13d33b9b25667efaa6edc9259f3f2'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


preserved_names = ['round1.plan.json', 'round1.profile_baseline_e0_pair.py',
                   'round1.review.json', 'round1.review.md', 'round1.source_guard_payload.json',
                   'deterministic_source_checks.json', 'historical_model_path_only.diff',
                   'pre_r2/tools/profile_baseline_e0_pair.py',
                   'pre_r2/reports/BASELINE_E0_PAIRED_LATENCY_PLAN_20261027.json']
preserved_hashes = {name: digest(TRACE / name) for name in preserved_names}
assert digest(ROOT / PLAN_NAME) == PLAN_SHA
assert digest(ROOT / SOURCE_NAME) == SOURCE_SHA
plan = json.loads((ROOT / PLAN_NAME).read_bytes())
old_plan = json.loads((TRACE / 'round1.plan.json').read_bytes())
old_review = json.loads((TRACE / 'round1.review.json').read_bytes())
checks_result = json.loads((TRACE / 'round2.deterministic_source_checks.json').read_bytes())
assert checks_result['plan_sha256'] == PLAN_SHA
assert checks_result['source_count'] == 119 and checks_result['python_source_count'] == 118
assert not checks_result['hash_mismatches']
assert not checks_result['syntax_failures']
assert not checks_result['unbound_imported_sources']
assert not checks_result['baseline_inherited_changes']

added = sorted(set(plan['source_sha256']) - set(old_plan['source_sha256']))
required = sorted(['dual_rgbt/__init__.py', 'lib/__init__.py', 'lib/models/__init__.py',
                   'lib/models/layers/__init__.py', 'lib/test/tracker/__init__.py', 'lib/utils/__init__.py'])
assert added == required
assert not set(old_plan['source_sha256']) - set(plan['source_sha256'])
changed_existing = [name for name, value in old_plan['source_sha256'].items()
                    if plan['source_sha256'][name] != value]
assert changed_existing == [SOURCE_NAME]
assert plan['task'] == 'profile_baseline_e0_pair_latency_20261027'
assert plan['scope'] == old_plan['scope'].replace('Source113', 'Source119')
assert {name: value for name, value in plan.items() if name not in ('source_sha256', 'scope', 'task')} == {
    name: value for name, value in old_plan.items() if name not in ('source_sha256', 'scope')}

old_source = (TRACE / 'round1.profile_baseline_e0_pair.py').read_text(encoding='utf-8')
new_source = (ROOT / SOURCE_NAME).read_text(encoding='utf-8')
admission = """    native = json.loads((ROOT / 'runs/pipeline_tasks' / plan['task'] / 'status.json').read_bytes())
    assert native['state'] == 'RUNNING' and native['pid'] == os.getppid()
    assert native['task'] == plan['task'] and native['physical_gpu'] == '0'
    assert native['command'] == [sys.executable, str(Path(__file__).resolve()), '--plan', args.plan]
"""
anchor = "    assert os.environ['OPENCV_FOR_THREADS_NUM'] == '1'\n"
assert new_source == old_source.replace(anchor, anchor + admission, 1)
assert new_source.index(admission) < new_source.index('os.killpg(pid, signal.SIGSTOP)')

checks = old_review['checks']


def move_evidence(value):
    if isinstance(value, list):
        return [move_evidence(item) for item in value]
    if isinstance(value, dict):
        return {key: move_evidence(item) if key == 'evidence' else item for key, item in value.items()}
    if isinstance(value, str) and value.startswith(SOURCE_NAME + ':'):
        return re.sub(r'\d+', lambda match: str(int(match.group()) + (4 if int(match.group()) >= 36 else 0)), value)
    if isinstance(value, str) and value.startswith(PLAN_NAME + ':'):
        return re.sub(r'\d+', lambda match: str(int(match.group()) + (6 if 128 <= int(match.group()) <= 160 else 0)), value)
    return value


checks = {name: move_evidence(check) for name, check in checks.items()}
checks['finally_resume']['details'] = checks['finally_resume']['details'].replace(
    ' F1 separately blocks admission of the profiler itself.',
    ' The profiler now also requires its own registered native wrapper before any stop signal.')
checks['native_profile_admission'] = dict(status='PASS', evidence=[
    'reports/BASELINE_E0_PAIRED_LATENCY_PLAN_20261027.json:167',
    'tools/profile_baseline_e0_pair.py:36-39', 'tools/profile_baseline_e0_pair.py:68-70',
    'scripts/run_task_four_gpu.py:18-31'],
    details='The plan mandates task profile_baseline_e0_pair_latency_20261027. Before input hashes, trainer admission or any stop signal, the profiler loads that task status and requires RUNNING, pid==os.getppid(), exact task, physical_gpu0, and exact command [sys.executable, resolved absolute profiler path, --plan, args.plan]. The reviewed native wrapper writes these fields before launching the subprocess and persists its actual return code. Direct/unregistered invocation cannot pass this gate with the existing native receipt. The executor must invoke the absolute profiler path under the detached registered wrapper; this review does not claim the future launch or exit has occurred.')
checks['import_dependency_coverage'] = dict(status='PASS', evidence=[
    'reports/BASELINE_E0_PAIRED_LATENCY_PLAN_20261027.json:127-132',
    'tools/profile_baseline_e0_pair.py:33',
    '.aris/traces/experiment-audit/2026-10-27_baseline_e0_pair_latency/round2.deterministic_source_checks.json'],
    details='All six omitted initializers are added with their exact current hashes. The conservative local static import closure contains63 files, including actual package initializers and the branch-local optional e1 factory imports, with zero local files outside the119-file source guard. All imported model/tracker/crop/launcher/control dependencies are hash-bound. External installed-package contents and remote deployment remain outside this local source review.')
checks['source_hash_and_compile'] = dict(status='PASS', evidence=[
    '.aris/traces/experiment-audit/2026-10-27_baseline_e0_pair_latency/round2.deterministic_source_checks.json'],
    details='All119 exact hashes match.118 Python files pass stdlib ast.parse(feature_version=(3,7)) and compile under the same installed CPython3.13.12 without project imports or pyc output. All112 inherited baseline-plan entries remain unchanged. Against R1, only the profiler hash changes and the six initializer hashes are added. The entire profiler differs by exactly the four-line native admission block. The plan additionally adds task and changes only the source-count wording from113 to119; inputs/phases/frame counts/model paths are unchanged.')
checks['e0_crop_binding']['details'] += ' R2 changes no import order, factory, tracker or crop source; the verified R1 binding remains intact.'

report = dict(old_review)
report.update(review_round='R2', status='SOURCE_PASS', verdict='PASS', blocking_issues=0, issues=[],
    generated_at=datetime.now(timezone.utc).isoformat(),
    review_scope='Targeted R2 source-readiness review. Directly read the entire revised profiler and exact plan changes; rechecked119 hashes,118 Python parse/compile checks and local import closure. Retains R1 semantic conclusions for byte-identical model/tracker/crop/timing/ownership/finally paths. No source edits, SSH, remote access, GPU, neural execution or checkpoint reads.',
    plan_sha256=PLAN_SHA, profiler_sha256=SOURCE_SHA, source_sha256=plan['source_sha256'],
    source_count=119, python_sources_compiled=118, checks=checks,
    previous_review='.aris/traces/experiment-audit/2026-10-27_baseline_e0_pair_latency/round1.review.json',
    resolved_findings=[dict(id='F1', status='RESOLVED', evidence='tools/profile_baseline_e0_pair.py:36-39',
                            details='Mandatory native task/status/parent PID/GPU/exact command binding precedes every signal.'),
                       dict(id='F2', status='RESOLVED', evidence='reports/BASELINE_E0_PAIRED_LATENCY_PLAN_20261027.json:127-132',
                            details='Six exact initializer hashes added; complete conservative local import closure is covered.')],
    round2_verification=dict(profiler_exact_four_line_admission_addition=True,
                             unchanged_prior_registered_entries=112, added_initializer_paths=added,
                             plan_other_changes=['task', 'scope Source113->Source119'],
                             local_import_closure_count=63, unbound_local_imports=[],
                             preserved_round1_artifact_sha256=preserved_hashes),
    claim_ceiling='SOURCE_PASS establishes source readiness only. After successful detached native execution and result audit, report the specified600-frame-per-phase TRAIN latency/totalP, initialization separately, all four phase results and loaded-host scope. Define any bracket-meanFPS comparison explicitly. No full-test FPS, new accuracy, isolated-host throughput or current32-model result is established.')
report['audited_input_sha256'] = dict(old_review['audited_input_sha256'])
report['audited_input_sha256'][PLAN_NAME] = PLAN_SHA
report['audited_input_sha256'][SOURCE_NAME] = SOURCE_SHA
report['audited_input_sha256']['.aris/traces/experiment-audit/2026-10-27_baseline_e0_pair_latency/round1.review.json'] = preserved_hashes['round1.review.json']
report['runtime_status'] = 'NOT_EXECUTED_BY_REVIEWER'
(TRACE / 'round2.plan.json').write_bytes((ROOT / PLAN_NAME).read_bytes())
(TRACE / 'round2.profile_baseline_e0_pair.py').write_bytes((ROOT / SOURCE_NAME).read_bytes())
for name in ['round2.review.json', 'round2.source_guard_payload.json']:
    (TRACE / name).write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
summary = '''# Baseline / historical E0 paired-latency source audit — R2

Verdict: **SOURCE_PASS — zero blockers.** This is source readiness, not a completed latency result. Requested Astra/max; actual model and effort UNATTESTED; same-family/provisional.

R1 F1 is resolved: `tools/profile_baseline_e0_pair.py:36-39` requires the plan's native task status to be RUNNING, match the wrapper parent PID, task and GPU0, and carry the exact executable/absolute profiler/plan command. It precedes the stop at line70. The existing native wrapper writes status before the child launch and persists its actual exit (`scripts/run_task_four_gpu.py:18-31`). Execute through that wrapper with the absolute profiler path and detached session.

R1 F2 is resolved: `reports/BASELINE_E0_PAIRED_LATENCY_PLAN_20261027.json:127-132` adds the six initializer hashes. All119 hashes match; all118 Python files pass offline AST/compile. The conservative local import closure has63 files and no unbound local imports. External packages and remote deployment were not inspected.

The profiler differs from R1 by exactly the four admission lines. The plan adds task, six initializers, the new profiler hash and the updated source-count wording. The112 inherited baseline source entries, model/tracker/crop paths, exact historical checkpoint hashes, four phases, frame counts, synchronized timers, ownership checks and finally resume behavior remain unchanged. E0 still binds six_channel_crop before importing its tracker.

Scope remains600 measured TRAIN frames per phase,20 warm frames per sequence, baseline-before / E0cost0 / E0cost0.05 / baseline-after, initialization separate. Report all phases and define any bracket-meanFPS arithmetic. Other GPU/CPU tasks remain active. No full-test FPS, new accuracy, current32-model result, successful future launch or resumed-training progress is asserted here.

Plan SHA256: `6c9cf09b2d843c78bde967cb02f4c735cb5dadae6ced44a462d247292e665909`

Profiler SHA256: `5a15d81a9575c13ac9bcfe4fa1accbe97ff13d33b9b25667efaa6edc9259f3f2`

The full machine guard is `round2.source_guard_payload.json`; the review is `round2.review.json`. R1 and pre_r2 artifacts are preserved with checksums. Only trace files were written; no remote, model, GPU or checkpoint operations occurred.
'''
(TRACE / 'round2.review.md').write_text(summary, encoding='utf-8')
assert preserved_hashes == {name: digest(TRACE / name) for name in preserved_names}
assert digest(ROOT / PLAN_NAME) == PLAN_SHA and digest(ROOT / SOURCE_NAME) == SOURCE_SHA
print(json.dumps(dict(status=report['status'], blocking_issues=0, plan_sha256=PLAN_SHA,
                     profiler_sha256=SOURCE_SHA, guard_sha256=digest(TRACE / 'round2.source_guard_payload.json'),
                     trace_path=report['trace_path']), indent=2))

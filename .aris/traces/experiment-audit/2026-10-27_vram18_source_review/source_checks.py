"""Read-only source/registration checks. No project imports or runtime probes."""
import ast
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path

ROOT = Path('C:/Users/gb/projects/Dual-RGBT')
OUT = ROOT / '.aris/traces/experiment-audit/2026-10-27_vram18_source_review'

def read(name):
    return json.loads((ROOT / name).read_bytes())

def sha(name):
    return hashlib.sha256((ROOT / name).read_bytes()).hexdigest()

index_path = 'reports/VRAM18_TRAINING_PLAN_INDEX_20261027.json'
protocol_path = 'reports/VRAM18_EVALUATION_PROTOCOL_20261027.json'
original_path = 'reports/JOINT_CANDIDATE_EVALUATION_PROTOCOL_20261026.json'
index, spec, original = map(read, [index_path, protocol_path, original_path])
grid = read(index['grid'])
profile_path = 'reports/HOST_ALLOCATOR_PROFILE_PLAN_20261027.json'
profile = read(profile_path)
all_plans = {name: read(name) for name in spec['training_plan_sha256']}
sources = {}
source_conflicts = []
for mapping in [spec['source_sha256'], original['source_sha256']] + [p['source_sha256'] for p in all_plans.values()]:
    for name, digest in mapping.items():
        if name in sources and sources[name] != digest:
            source_conflicts.append(name)
        sources[name] = digest
source_actual = {name: sha(name) for name in sources}
source_mismatches = {name: [digest, source_actual[name]] for name, digest in sources.items() if digest != source_actual[name]}
syntax = {}
for name in sources:
    ast.parse((ROOT / name).read_text(encoding='utf-8-sig'), filename=name)
    syntax[name] = 'PASS'

allowed_differences = {'physical_gpu','cost_weight','max_candidates','runs','tasks','source_review','sanity_native_status','execution_env','registered_grid','comparison_control','launch_wrapper','scope','source_sha256'}
plan_checks = []
for row in index['plans']:
    plan = all_plans[row['path']]
    base = read(plan['comparison_control'])
    diff = [k for k in sorted(set(plan)|set(base)) if plan.get(k) != base.get(k)]
    source_added = {k:v for k,v in plan['source_sha256'].items() if k not in base['source_sha256']}
    plan_checks.append(dict(path=row['path'],sha256=sha(row['path']),registered_sha256=row['sha256'],
        metadata_matches=all(plan[k] == row[v] for k,v in [('physical_gpu','gpu'),('cost_weight','cost_weight'),('max_candidates','max_candidates')]),
        objective_matches=plan['repair']['kind']=={'joint_quality':'joint_preview_relative_quality','joint_gainpair':'signed_gain_and_same_state_difference_regression'}[row['objective']],
        comparison_control=plan['comparison_control'], differences=diff,unexpected_differences=sorted(set(diff)-allowed_differences),
        source_added=source_added,source_removed=sorted(set(base['source_sha256'])-set(plan['source_sha256'])),
        source_changed=[k for k in base['source_sha256'] if plan['source_sha256'].get(k)!=base['source_sha256'][k]],
        original_inputs_exact=plan['input_sha256']==base['input_sha256'],sequence_order_exact=plan['sequence_names']==base['sequence_names'],
        sequences=len(plan['sequence_names']),unique_sequences=len(set(plan['sequence_names'])),seed=plan['seed'],
        rounds=plan['rounds'],label_stride=plan['label_stride'],formal_counts=plan['expected_formal_counts'],sanity_counts=plan['sanity_expected_counts'],
        source_count=len(plan['source_sha256']),source_review=plan['source_review']))

kind_to_objective={'joint_preview_relative_quality':'joint_quality','signed_gain_and_same_state_difference_regression':'joint_gainpair'}
configs=[(kind_to_objective[p['repair']['kind']],p['max_candidates'],p['cost_weight']) for p in all_plans.values()]
expected=set(itertools.product(grid['objectives'],grid['candidate_counts'],grid['trajectory_costs']))
original_plan_checks={name:dict(expected=digest,actual=sha(name),retained=spec['training_plan_sha256'].get(name)==digest) for name,digest in original['training_plan_sha256'].items()}
inputs={}
input_conflicts=[]
for mapping in [spec['input_sha256'],original['input_sha256']] + [p['input_sha256'] for p in all_plans.values()]:
    for name,digest in mapping.items():
        if name in inputs and inputs[name] != digest:
            input_conflicts.append(name)
        inputs[name]=digest
input_checks={}
for name,digest in inputs.items():
    if name.endswith(('.pth','.pt','.pth.tar')):
        input_checks[name]=dict(expected=digest,status='NOT_READ_WEIGHTS_PROHIBITED')
    elif (ROOT/name).is_file():
        actual=sha(name)
        input_checks[name]=dict(expected=digest,actual=actual,status='PASS' if actual==digest else 'FAIL')
    else:
        input_checks[name]=dict(expected=digest,status='NOT_LOCAL_RUNTIME_GATED')

protocol_diffs=[k for k in sorted(set(spec)|set(original)) if spec.get(k)!=original.get(k)]
owners=index['original_owners']
owner_counts=Counter(int(v['gpu']) for v in owners)
wave_counts={}
for phase in [4,6,8]:
    adds=Counter(v['gpu'] for v in index['plans'] if v['phase']==phase)
    owner_counts.update(adds)
    wave_counts[str(phase)]=dict(adds=dict(adds),cumulative=dict(owner_counts))

profile_sources={name:dict(expected=digest,actual=sha(name)) for name,digest in profile['source_sha256'].items()}
profile_plans={row['name']:read(row['plan']) for row in profile['cases']}
profile_plan_diff=[k for k in sorted(set(profile_plans['default'])|set(profile_plans['compact'])) if profile_plans['default'].get(k)!=profile_plans['compact'].get(k)]
summary=dict(index_sha256=sha(index_path),launcher_sha256=sha('tools/launch_vram18_training_wave.py'),register_sha256=sha('tools/register_vram18_training_plans.py'),protocol_sha256=sha(protocol_path),original_protocol_sha256=sha(original_path),
    grid_sha256=sha(index['grid']),grid_registered_sha256=index['grid_sha256'],source_sha256=sources,source_actual_sha256=source_actual,source_count=len(sources),source_mismatches=source_mismatches,source_conflicts=source_conflicts,source_syntax=syntax,
    all_plans_count=len(all_plans),new_plans_count=len(index['plans']),plan_checks=plan_checks,original_plan_checks=original_plan_checks,
    unique_configuration_count=len(set(configs)),configuration_count=len(configs),grid_exact=set(configs)==expected,missing_configurations=sorted(expected-set(configs)),extra_configurations=sorted(set(configs)-expected),
    unique_run_names=len({v for p in all_plans.values() for v in p['runs'].values()}),unique_task_names=len({v for p in all_plans.values() for v in p['tasks'].values()}),
    protocol_changed_fields=protocol_diffs,original_protocol_fields_unchanged=all(spec.get(k)==v for k,v in original.items() if k not in ['scope','source_review','training_plan_sha256','source_sha256']),
    protocol_nonmap={k:v for k,v in spec.items() if k not in ['source_sha256','input_sha256','training_plan_sha256','toolkit_code_sha256']},
    input_bindings=inputs,input_checks=input_checks,input_conflicts=input_conflicts,wave_counts=wave_counts,
    original_owners_match_plan=owners==read('reports/FOUR_GPU_EIGHT_TRAIN_SIX_HOUR_NODE_PLAN_20261026.json')['tasks'],
    baseline_owners_match=all(any(v['task']==o['task'] and v['pid']==o['pid'] and str(v['gpu'])==str(o['gpu']) for v in read('reports/VRAM18_PRE_EXPANSION_BASELINE_20261027.json')['tasks']) for o in owners),
    profile_plan_sha256=sha(profile_path),profile_source_checks=profile_sources,profile_case_plan_checks={row['plan']:dict(expected=row['plan_sha256'],actual=sha(row['plan'])) for row in profile['cases']},
    profile_plan_differences=profile_plan_diff,profile_compact_environment=profile_plans['compact']['execution_env'],
    profile_environment_matches_new_plans=all(p['execution_env']==profile_plans['compact']['execution_env'] for name,p in all_plans.items() if name not in original['training_plan_sha256']))
(OUT/'deterministic_checks.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k not in ['source_sha256','source_actual_sha256','source_syntax','input_checks','input_bindings','plan_checks','profile_source_checks','protocol_nonmap']},indent=2))
print('NEW_PLAN_CHECKS',json.dumps(plan_checks,indent=2))
print('INPUT_CHECKS',json.dumps(input_checks,indent=2))
print('PROTOCOL_NONMAP',json.dumps(summary['protocol_nonmap'],indent=2))

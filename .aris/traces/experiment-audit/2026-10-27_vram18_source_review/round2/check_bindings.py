"""R2 local binding regression. Uses the existing source-only verifier."""
import ast
import contextlib
import hashlib
import json
from pathlib import Path

ROOT=Path('C:/Users/gb/projects/Dual-RGBT')
OUT=ROOT/'.aris/traces/experiment-audit/2026-10-27_vram18_source_review/round2'
tree=ast.parse((OUT.parent/'source_checks.py').read_text())
for node in tree.body:
    if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name):
        if node.targets[0].id=='OUT':node.value=ast.parse("ROOT / '.aris/traces/experiment-audit/2026-10-27_vram18_source_review/round2'",mode='eval').body
        if node.targets[0].id=='profile_path':node.value=ast.Constant('reports/HOST_ALLOCATOR_PROFILE_R4_PLAN_20261027.json')
ast.fix_missing_locations(tree)
with (OUT/'binding_verifier.stdout.txt').open('w') as log,contextlib.redirect_stdout(log):
    exec(compile(tree,'existing_source_only_verifier','exec'),{})
e=json.loads((OUT/'deterministic_checks.json').read_bytes())
read=lambda n:json.loads((ROOT/n).read_bytes())
sha=lambda n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest()
index=read('reports/VRAM18_TRAINING_PLAN_INDEX_20261027.json')
profile=read(index['host_profile_plan']);review=read(profile['source_review'])
assert review['status']=='SOURCE_PASS' and review['blocking_issues']==0
assert review['profile_plan_sha256']==sha(index['host_profile_plan'])
assert review['source_sha256']==profile['source_sha256']
assert all(sha(n)==h for n,h in profile['source_sha256'].items())
assert sha(index['observer_source'])==index['observer_sha256']
assert sha(index['baseline_previous'])==index['baseline_previous_sha256']
assert sha(index['baseline_reference'])==index['baseline_reference_sha256']
before=read(index['baseline_previous']);after=read(index['baseline_reference'])
dt=after['observed']-before['observed'];assert dt>0
old={x['native']['task']:x for x in before['tasks']}
rates={x['task']:(x['updates']-old[x['task']]['logged_updates'])/dt for x in after['tasks']}
assert len(rates)==8 and all(v>0 for v in rates.values())
for row in after['tasks']:
    a=old[row['task']]
    assert (a['native']['pid'],a['child_pid'],a['native']['physical_gpu'])==(row['pid'],row['child_pid'],row['gpu'])
assert e['grid_exact'] and e['unique_configuration_count']==32 and e['source_count']==57 and not e['source_mismatches']
assert all(v['sha256']==v['registered_sha256'] and v['metadata_matches'] and v['objective_matches'] and not v['unexpected_differences'] and v['original_inputs_exact'] and v['sequence_order_exact'] and not v['source_removed'] and not v['source_changed'] for v in e['plan_checks'])
assert all(v['expected']==v['actual'] and v['retained'] for v in e['original_plan_checks'].values())
assert all(v['status']!='FAIL' for v in e['input_checks'].values())
summary=dict(status='PASS',index_sha256=e['index_sha256'],launcher_sha256=e['launcher_sha256'],register_sha256=e['register_sha256'],
    protocol_sha256=e['protocol_sha256'],source_count=57,original_sources_unchanged=55,original_plans_unchanged=8,
    observer_sha256=index['observer_sha256'],profile_plan_sha256=sha(index['host_profile_plan']),profile_review_sha256=sha(profile['source_review']),
    profile_helper_sha256={n:h for n,h in profile['source_sha256'].items() if 'allocator' in n},
    baseline_interval=dt,baseline_rates=rates,baseline_aggregate_rate=sum(rates.values()),
    new_plan_count=24,unique_configuration_count=32,runtime_profile_inspected=False)
(OUT/'bindings_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))

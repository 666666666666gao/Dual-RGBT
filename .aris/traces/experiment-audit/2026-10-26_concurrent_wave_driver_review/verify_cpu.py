"""AST/stdlib-only driver review; no import or execution of the operational driver."""
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[4]
TRACE=Path(__file__).resolve().parent
SOURCE='tools/launch_concurrent_training_wave.py'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
EXPECTED='cbcd5a4465cabd55dfa95d8797b8ff9832b3e7d3766f37600ec58ccae4324f6b'
assert sha(ROOT/SOURCE)==EXPECTED
tree=ast.parse((ROOT/SOURCE).read_text())
index=read(ROOT/'reports/CONCURRENT_TRAIN_PLAN_INDEX_20261026.json')
assert sha(ROOT/'reports/CONCURRENT_TRAIN_PLAN_INDEX_20261026.json')=='de4fe9afa72168e27b28a5bdf24f67a1989beabb62e51b788f67415904dc9993'
functions=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
assert [n.name for n in functions]==['owned_formals','launch']
remote_root='/data2/gb/Dual-RGBT'
python=remote_root+'/conda-envs/dual_rgbt/bin/python'
captured=[]
def capture(code):
    ast.parse(code)
    captured.append(code)
    return dict(code=code)
ns=dict(ROOT=ROOT,Path=Path,json=json,REMOTE=remote_root,PYTHON=python,remote=capture,capacity={'max_formal_tasks_per_gpu01':3})
exec(compile(ast.fix_missing_locations(ast.Module(body=functions,type_ignores=[])),'<driver-definitions-only>','exec'),ns)
original_owned=ns['owned_formals']()
assert [(v['gpu'],v['pid']) for v in original_owned]==[(0,828655),(1,1054384)]
cases=[]
for phase,token,minimum,gpus in [('k8cost0','_k8_cost0_',2,[0,1]),('k4cost05','_k4_cost0p05_',3,[0,1]),('reserved23','_k4_cost0_',1,[2,3])]:
    items=[v for v in index['plans'] if token in v['path']]
    assert len(items)==2 and [v['gpu'] for v in items]==gpus
    for v in items:
        plan=read(ROOT/v['path']);review=read(ROOT/plan['source_review'])
        assert sha(ROOT/v['path'])==v['sha256']==review['plan_sha256']
        assert review['status']=='SOURCE_PASS' and review['blocking_issues']==0 and review['source_sha256']==plan['source_sha256']
        assert v['gpu']==plan['physical_gpu']
        case=dict(plan=v['path'],plan_sha256=v['sha256'],gpu=plan['physical_gpu'],runs=plan['runs'],tasks=plan['tasks'])
        for sanity in [True,False]:
            ns['launch'](case,sanity,None if sanity else 'mock-M0-complete-digest')
            generated=ast.parse(captured[-1])
            popen=next(n for n in ast.walk(generated) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='Popen')
            cmd=ast.literal_eval(popen.args[0])
            stage='sanity' if sanity else 'formal'
            assert cmd[:7]==[python,plan['launch_wrapper'],'--task',plan['tasks'][stage],'--gpu',str(case['gpu']),'--']
            assert cmd[7:]==[python,remote_root+'/scripts/train_candidate_'+('joint_gainpair' if 'joint_gainpair' in case['plan'] else 'joint_quality')+'_trajectory.py','--plan',case['plan'],'--run',plan['runs'][stage]]+(['--sanity'] if sanity else ['--sanity-complete-sha256','mock-M0-complete-digest'])
        cases.append(dict(phase=phase,plan=v['path'],gpu=case['gpu'],minimum_capacity=minimum,generated_sanity_and_formal_scripts='AST_PASS',wrapper_prefix_length=7))

capacity_assertions=[n for n in tree.body if isinstance(n,ast.Assert) and n.lineno in [7,8]]
assert len(capacity_assertions)==2
capacity_code=compile(ast.fix_missing_locations(ast.Module(body=capacity_assertions,type_ignores=[])),'<capacity-assertions>','exec')
capacity_cases=[]
for phase,minimum in [('k8cost0',2),('k4cost05',3),('reserved23',1)]:
    for value in [minimum-1,minimum]:
        env=dict(capacity=dict(status='ADMIT_BOUNDED_CONFIG_TRAINING',result_audit_blocking_issues=0,max_formal_tasks_per_gpu01=value),args=SimpleNamespace(phase=phase))
        accepted=True
        try:exec(capacity_code,env)
        except AssertionError:accepted=False
        assert accepted==(value>=minimum)
        capacity_cases.append(dict(phase=phase,capacity=value,accepted=accepted))

# Execute only the generated pure hardware assertions against fake paths/subprocess outputs.
sample=ast.parse(captured[-1])
exact_pid_assert=next(n for n in sample.body if isinstance(n,ast.Assert) and 'actual == expected' in ast.unparse(n))
hardware_if=next(n for n in sample.body if isinstance(n,ast.If) and ast.unparse(n.test)=='gpu >= 2')
hardware_code=compile(ast.fix_missing_locations(ast.Module(body=[exact_pid_assert,hardware_if],type_ignores=[])),'<mock-hardware-assertions>','exec')
class MockPath:
    def __init__(self,value,existing):self.value=value;self.existing=existing
    def __truediv__(self,value):return MockPath(self.value+'/'+str(value),self.existing)
    def exists(self):return self.value in self.existing
hardware_cases=[]
for gpu,old_pid in [(2,1028233),(3,1028232)]:
    for label,old_exists,pids,memory,expected_accept in [('free',False,set(),499,True),('old_ReID_pid',True,set(),499,False),('unexpected_NVML_pid',False,{998877},499,False),('memory_boundary',False,set(),500,False)]:
        existing={f'/proc/{old_pid}'} if old_exists else set()
        env=dict(gpu=gpu,expected=set(),actual=pids,Path=lambda p:MockPath(p,existing),subprocess=SimpleNamespace(check_output=lambda *a,**k:str(memory)))
        accepted=True
        try:exec(hardware_code,env)
        except AssertionError:accepted=False
        assert accepted==expected_accept
        hardware_cases.append(dict(gpu=gpu,case=label,accepted=accepted))
for count in [2,3]:
    env=dict(gpu=1,expected=set(range(count)),actual=set(range(count)))
    accepted=True
    try:exec(hardware_code,env)
    except AssertionError:accepted=False
    assert accepted==(count<3)
    hardware_cases.append(dict(gpu=1,case=f'capacity3_existing{count}',accepted=accepted))

# Embedded M0/startup templates compile with real case serialization; no remote execution.
templates=[n for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str) and n.value.startswith('import ') and '\n' in n.value]
assert len(templates)==3
for node in templates:
    if 'case=%r' in node.value and 'owned=%r' not in node.value:
        ast.parse(node.value%(remote_root,dict(plan=index['plans'][0]['path'],plan_sha256=index['plans'][0]['sha256'],gpu=0)))
assert not any(isinstance(n,(ast.Try,ast.While)) for n in ast.walk(tree))
summary=dict(status='CPU_DRIVER_CHECK_PASS',driver_sha256=EXPECTED,index_sha256=sha(ROOT/'reports/CONCURRENT_TRAIN_PLAN_INDEX_20261026.json'),cases=cases,
    generated_launch_scripts=12,capacity_boundary_cases=capacity_cases,mocked_hardware_gate_cases=hardware_cases,
    initial_owned_formal_records=original_owned,embedded_remote_templates_compiled=3,
    capacity_decision_exists_at_review=(ROOT/'reports/TRAIN_CONCURRENCY_CAPACITY_DECISION_20261026.json').exists(),
    driver_imported_or_run=False,remote_executed=False,subprocess_launched=False,NN_launched=False,source_edited=False)
(TRACE/'cpu_verification.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k not in ['cases','initial_owned_formal_records','capacity_boundary_cases','mocked_hardware_gate_cases']},indent=2))

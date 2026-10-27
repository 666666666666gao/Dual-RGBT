"""Exact source AST with inert I/O; all health objects are synthetic fixtures."""
import ast
import contextlib
import copy
import hashlib
import io
import json
import math
from pathlib import Path
from types import SimpleNamespace

ROOT=Path('C:/Users/gb/projects/Dual-RGBT')
OUT=ROOT/'.aris/traces/experiment-audit/2026-10-27_vram18_source_review/round2'
tree=ast.parse((ROOT/'tools/launch_vram18_training_wave.py').read_text())
index=json.loads((ROOT/'reports/VRAM18_TRAINING_PLAN_INDEX_20261027.json').read_bytes())
remote_root='/data2/gb/Dual-RGBT';python=remote_root+'/conda-envs/dual_rgbt/bin/python'
launch_node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='launch')
programs=[];captured=[]
def capture(code):
    ast.parse(code);captured.append(code);return {'source_only':True}
namespace=dict(ROOT=ROOT,Path=Path,json=json,REMOTE=remote_root,PYTHON=python,
    owned=[dict(e,cuda_ready=True) for e in index['original_owners']],temporary=[],compact_peak=1,remote=capture)
exec(compile(ast.Module(body=[launch_node],type_ignores=[]),'source_launch_builder','exec'),namespace)
assignments=[n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='code' for t in n.targets)]
def template(node):return node.value.left.value
witnesses=[n for n in assignments if template(n).startswith(('import hashlib,json\n','import csv,json,math,time\n'))]
assert len(witnesses)==2
for row in index['plans']:
    plan=json.loads((ROOT/row['path']).read_bytes())
    case=dict(plan=row['path'],plan_sha256=row['sha256'],gpu=row['gpu'],runs=plan['runs'],tasks=plan['tasks'])
    namespace['args']=SimpleNamespace(phase=row['phase'])
    for sanity in [True,False]:
        stage='sanity' if sanity else 'formal'
        namespace['launch'](case,sanity,None if sanity else '0'*64)
        code=captured[-1];parsed=ast.parse(code)
        call=next(n for n in ast.walk(parsed) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='Popen')
        command=ast.literal_eval(call.args[0])
        expected=[python,plan['launch_wrapper'],'--task',plan['tasks'][stage],'--gpu',str(row['gpu']),'--',python,remote_root+'/scripts/train_candidate_'+row['objective']+'_trajectory.py','--plan',row['path'],'--run',plan['runs'][stage]]
        expected+=['--sanity'] if sanity else ['--sanity-complete-sha256','0'*64]
        assert command==expected
        programs.append(dict(plan=row['path'],kind=stage+'_launch',syntax='PASS',command=command,source_sha256=hashlib.sha256(code.encode()).hexdigest(),source=code))
        if not sanity:case['formal']=dict(task=plan['tasks']['formal'],pid=1,gpu=row['gpu'],command=command,launched=0)
    for node in witnesses:
        code=eval(compile(ast.Expression(node.value),'source_witness_builder','eval'),dict(REMOTE=remote_root,c=case))
        ast.parse(code)
        programs.append(dict(plan=row['path'],kind='sanity_admission' if template(node).startswith('import hashlib') else 'formal_startup',syntax='PASS',source=code))
measurement=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='measurement')
exec(compile(ast.Module(body=[measurement],type_ignores=[]),'source_measurement_builder','exec'),namespace)
namespace['measurement'](namespace['owned'])
programs.append(dict(kind='wave_measurement',syntax='PASS',source=captured[-1]))
observer=ast.parse((ROOT/index['observer_source']).read_text())
observer_code=next(n.value for n in ast.walk(observer) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='code' for t in n.targets))
code=eval(compile(ast.Expression(observer_code),'source_observer_builder','eval'),dict(REMOTE=remote_root,node={'tasks':index['original_owners']}))
ast.parse(code);programs.append(dict(kind='scheduled_observer',syntax='PASS',source=code))

guard=next(n for n in ast.walk(ast.parse(programs[0]['source'])) if isinstance(n,ast.Assert) and isinstance(n.test,ast.BoolOp) and any(isinstance(v,ast.Compare) and isinstance(v.left,ast.Name) and v.left.id=='required' for v in n.test.values))
guard_code=compile(ast.Expression(guard.test),'exact_ownership_condition','eval')
ownership_fixtures=[
 ('cpu_initializing_registered_child',dict(required={1,2,3},actual={1,2,3},expected={1,2,3,4},limit=6),True),
 ('initializing_child_now_has_cuda',dict(required={1,2,3},actual={1,2,3,4},expected={1,2,3,4},limit=6),True),
 ('foreign_cuda_child',dict(required={1,2,3},actual={1,2,3,99},expected={1,2,3,4},limit=6),False),
 ('missing_warmed_child',dict(required={1,2,3},actual={1,2},expected={1,2,3,4},limit=6),False),
 ('registered_slot_cap_even_without_cuda',dict(required={1},actual={1},expected={1,2,3,4},limit=4),False)]
ownership=[]
for name,values,wanted in ownership_fixtures:
    actual=eval(guard_code,values);assert actual==wanted
    ownership.append(dict(fixture=name,expected= wanted,actual=actual,status='PASS'))

class MemoryPath:
    def __init__(self,key,store):self.key=str(key);self.store=store
    def __truediv__(self,key):return MemoryPath(self.key.rstrip('/')+'/'+str(key),self.store)
    def read_bytes(self):return self.store[self.key]
    def write_text(self,text):self.store[self.key]=text.encode();return len(text)
    def exists(self):return self.key in self.store

health_node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='score_health')
baseline_previous=json.loads((ROOT/index['baseline_previous']).read_bytes())
baseline_reference=json.loads((ROOT/index['baseline_reference']).read_bytes())
base_dt=baseline_reference['observed']-baseline_previous['observed']
old={r['native']['task']:r['logged_updates'] for r in baseline_previous['tasks']}
controls={r['task']:(r['updates']-old[r['task']])/base_dt for r in baseline_reference['tasks']}
def fixture(kind,phase=4):
    memory={};root=MemoryPath('root',memory)
    setjson=lambda name,value:memory.__setitem__('root/'+name,json.dumps(value).encode())
    index_data=copy.deepcopy(index)
    setjson('reports/VRAM18_TRAINING_PLAN_INDEX_20261027.json',index_data)
    # Hashes use the original exact input bytes, not JSON reserialization.
    memory['root/'+index['baseline_previous']]=(ROOT/index['baseline_previous']).read_bytes()
    memory['root/'+index['baseline_reference']]=(ROOT/index['baseline_reference']).read_bytes()
    names=list(controls)+['SYNTHETIC_NEW_CASE_%d'%i for i in range(phase*4-8)]
    before=[];after=[]
    for i,name in enumerate(names):
        native=dict(task=name,state='RUNNING',pid=1000+i,physical_gpu=str(i%4),command=['SYNTHETIC_NOT_EXECUTABLE',name])
        initial=dict(native=native,child_pid=2000+i,logged_updates=100,finite_logged_losses=True,host_memory={'Swap':'0 kB'})
        end=copy.deepcopy(initial)
        delta=math.ceil(controls[name]*.9*1200) if name in controls else 1200
        end['logged_updates']+=delta
        before.append(initial);after.append(end)
    observed=1200
    available=16*1024*1024
    if kind=='aggregate_not_improved':
        for row in after:row['logged_updates']=101
    if kind=='original_retention_below_floor':after[0]['logged_updates']=100+math.floor(controls[names[0]]*.39*1200)
    if kind=='swap_growth_over_limit':after[0]['host_memory']['Swap']=str(64*1024+1)+' kB'
    if kind=='insufficient_ram':available=8*1024*1024-1
    if kind=='short_interval':observed=599
    if kind=='changed_child_identity':after[0]['child_pid']+=10000
    state=dict(state='FORMAL_STARTUP_PASS',index_sha256=hashlib.sha256(memory['root/reports/VRAM18_TRAINING_PLAN_INDEX_20261027.json']).hexdigest(),measurement_start=dict(observed=0,tasks=before))
    state_name='reports/VRAM18_WAVE%d_EXECUTION_20261027.json'%phase
    setjson(state_name,state);setjson('synthetic_health_node.json',dict(observed=observed,tasks=after))
    if phase>4:setjson('reports/VRAM18_WAVE%d_HEALTH_20261027.json'%(phase-2),dict(status='ADMIT_NEXT_WAVE',aggregate_updates_per_second=100 if kind=='previous_wave_faster' else 15))
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    namespace=dict(json=json,ROOT=root,sha=sha,remote=lambda code:dict(MemAvailable=available))
    exec(compile(ast.Module(body=[health_node],type_ignores=[]),'exact_health_scorer','exec'),namespace)
    args=SimpleNamespace(phase=phase,health_node='synthetic_health_node.json')
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            namespace['score_health'](args,index_data,root/state_name)
        except AssertionError:
            return 'REJECTED_ASSERTION'
    return json.loads(memory['root/reports/VRAM18_WAVE%d_HEALTH_20261027.json'%phase])['status']

health=[]
for name,phase,expected in [
 ('healthy_wave',4,'ADMIT_NEXT_WAVE'),('aggregate_not_improved',4,'HOLD_CURRENT_WAVE'),
 ('original_retention_below_floor',4,'HOLD_CURRENT_WAVE'),('swap_growth_over_limit',4,'HOLD_CURRENT_WAVE'),
 ('insufficient_ram',4,'HOLD_CURRENT_WAVE'),('short_interval',4,'REJECTED_ASSERTION'),
 ('changed_child_identity',4,'REJECTED_ASSERTION'),('previous_wave_faster',6,'HOLD_CURRENT_WAVE'),
 ('healthy_wave',6,'ADMIT_NEXT_WAVE')]:
    result=fixture(name,phase);assert result==expected,(name,result,expected)
    health.append(dict(fixture=name,phase=phase,expected=expected,actual=result,status='PASS'))

result=dict(status='PASS',scope='SYNTHETIC_SOURCE_REGRESSION_ONLY_NOT_RUNTIME_EVIDENCE',remote_execution=False,nn_execution=False,
    generated_program_count=len(programs),training_argv_count=48,ownership_fixtures=ownership,health_fixtures=health)
(OUT/'generated_remote_programs.json').write_text(json.dumps(programs,indent=2)+'\n')
(OUT/'focused_regressions.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))

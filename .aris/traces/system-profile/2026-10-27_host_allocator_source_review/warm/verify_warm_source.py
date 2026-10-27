"""Focused warm callback source audit; no remote, torch, gc or libc execution."""
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[5]
TRACE = Path(__file__).resolve().parent
REMOTE = '/data2/gb/Dual-RGBT'
PYTHON = REMOTE + '/conda-envs/dual_rgbt/bin/python'
read = lambda name: json.loads((ROOT / name).read_bytes())
sha = lambda name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
PROFILE = 'reports/HOST_ALLOCATOR_WARM_PROFILE_PLAN_20261027.json'
CASE = 'reports/host_allocator_warm_case_plan_20261027.json'
expected = {
 'scripts/train_candidate_host_warm_compact.py':'d0ed5388da0ce02ff28c42d1acc9ebf6fd278b3825cf2c9c8e9eaf4f291d8472',
 'tools/profile_host_allocator_warm.py':'afb3f822c1e5763c3a029b3c206e34cb081c7ef9d35663566bcc8ada73d3ced1',
 PROFILE:'d1d0d33c037f51c5a49b7269c411dcd80fb9a998a91b1d387039e10c818d4bb3',
 CASE:'412849a40f095ee58f8fc12b03c28731c3f41d8584c09f28d58ed08b07ae2213'}
assert all(sha(n) == h for n,h in expected.items())
profile, case = read(PROFILE), read(CASE)
assert len(profile['source_sha256']) == 55 and len(case['source_sha256']) == 52
assert all(sha(n) == h for n,h in profile['source_sha256'].items())
assert all(profile['source_sha256'][n] == h for n,h in case['source_sha256'].items())
original = read('reports/concurrent_joint_quality_k4_cost0_plan_20261026.json')
assert all(case['source_sha256'][n] == h for n,h in original['source_sha256'].items())
assert case['input_sha256'] == original['input_sha256'] and case['sequence_names'] == original['sequence_names']
prior_case = read('reports/host_allocator_trim_case_plan_20261027.json')
changed = sorted(k for k in set(case)|set(prior_case) if case.get(k) != prior_case.get(k))
assert changed == ['launch_trainer','registered_grid','runs','sanity_native_status','scope','source_review','source_sha256','tasks']
old_trim = (ROOT/'tools/profile_host_allocator_trim.py').read_text()
source = (ROOT/'tools/profile_host_allocator_warm.py').read_text()
assert source == old_trim.replace('HOST_ALLOCATOR_TRIM_', 'HOST_ALLOCATOR_WARM_').replace('PAIRED_HOST_COMPACT_TRIM_M0_COMPLETE', 'PAIRED_HOST_COMPACT_WARM_M0_COMPLETE')

wave = read('reports/VRAM18_WAVE4_EXECUTION_20261027.json')
measurement = wave['measurement_start']
derived = [dict(task=x['native']['task'],pid=x['native']['pid'],gpu=x['native']['physical_gpu'],
                child_pid=x['child_pid'],command=x['native']['command']) for x in measurement['tasks']]
assert profile['owners'] == derived and len(derived) == 16
assert len({x['pid'] for x in derived}) == len({x['child_pid'] for x in derived}) == 16
assert {g:sum(x['gpu']==g for x in derived) for g in '0123'} == {g:4 for g in '0123'}
reference = profile['cases'][0]
assert reference == read('reports/HOST_ALLOCATOR_TRIM_PROFILE_PLAN_20261027.json')['cases'][0]
assert sha(reference['reference_sample']) == reference['reference_sample_sha256']
assert sha(reference['plan']) == reference['plan_sha256']
ref = read(reference['reference_sample'])
assert ref['native']['state'] == 'COMPLETE' and ref['native']['exit_code'] == 0
assert ref['case']['plan_sha256'] == reference['plan_sha256']

tree = ast.parse(source)
assignments = [n.value for n in ast.walk(tree) if isinstance(n,ast.Assign) and isinstance(n.value,ast.BinOp)
               and any(isinstance(t,ast.Name) and t.id == 'code' for t in n.targets)]
sampler = next(x for x in assignments if x.left.value.startswith('import csv'))
comparison = next(x for x in assignments if x.left.value.startswith('import json,torch'))
new_case = profile['cases'][1]
assert new_case['kind'] == 'new_admission' and new_case['plan_sha256'] == sha(CASE)
code = eval(compile(ast.Expression(sampler),'<template-only>','eval'),{'__builtins__':{}},dict(REMOTE=REMOTE,case=new_case,plan=profile,PYTHON=PYTHON))
sampler_tree = ast.parse(code);compile(sampler_tree,'generated_warm_sampler','exec')
(TRACE/'generated_sampler.py').write_text(code,encoding='utf-8')
command = next(n.value for n in ast.walk(sampler_tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='command' for t in n.targets))
argv = eval(compile(ast.Expression(command),'<argv-only>','eval'),{'__builtins__':{'str':str}},dict(p=case,gpu=2,r=PurePosixPath(REMOTE),trainer=case['launch_trainer'],case=new_case))
assert argv == [PYTHON,case['launch_wrapper'],'--task',case['tasks']['sanity'],'--gpu','2','--',PYTHON,
                REMOTE+'/'+case['launch_trainer'],'--plan',CASE,'--run',case['runs']['sanity'],'--sanity']
comparison_code = eval(compile(ast.Expression(comparison),'<template-only>','eval'),{'__builtins__':{}},dict(REMOTE=REMOTE,plan=profile))
compile(comparison_code,'generated_warm_comparison','exec');(TRACE/'generated_comparison.py').write_text(comparison_code,encoding='utf-8')
assert 'torch.allclose(a,b,atol=1e-8,rtol=1e-6)' in comparison_code

# Exact wrapper-main AST; original step, stage, memory, gc and libc are all mocks.
events, writes, step_args, returned = [], [], [], []
sentinels = [object() for _ in range(6)]
native_argv = argv[argv.index('--')+1:]
fake_sys = SimpleNamespace(argv=native_argv[1:].copy(),executable=PYTHON)
saved_argv = fake_sys.argv.copy()
model, optimizer, prefixes = object(), object(), object()
losses = [object() for _ in range(6)]

class Memory:
 def read_text(self):
  events.append('memory')
  return 'header\nRss: 100 kB\nPss: 90 kB\n'

class Run:
 def __truediv__(self,name):
  assert name == 'host_memory_warmup.json';return self
 def write_text(self,value):
  events.append('write_warm_memory');writes.append(json.loads(value))

stage_args = [model,optimizer,case,object(),Run(),object(),object(),True]
def original_step(loss,m,o,p,s):
 index=len(step_args)
 assert loss is losses[index] and m is model and o is optimizer and p is prefixes and s is True
 step_args.append((loss,m,o,p,s));events.append('original_step_'+str(index+1))
 return sentinels[index]

fake_module = SimpleNamespace(step_optimizer=original_step)
def original_stage(*received):
 assert all(a is b for a,b in zip(received,stage_args)) and len(received)==8
 events.append('original_stage')
 for i in range(6):
  result=fake_module.step_optimizer(losses[i],model,optimizer,prefixes,True)
  assert result is sentinels[i];returned.append(result)
  assert len(writes) == (0 if i<3 else 1)
 return 'unchanged_stage_return'
fake_module.trajectory_stage=original_stage
def original_main():
 events.append('original_main')
 assert native_argv == [fake_sys.executable]+fake_sys.argv
 assert fake_module.trajectory_stage(*stage_args)=='unchanged_stage_return'
fake_module.main=original_main
def import_module(name):
 assert name=='train_candidate_joint_quality_trajectory';return fake_module
class Trim:
 def __call__(self,pad):
  assert pad==0 and self.argtypes==['size_t'] and self.restype=='int'
  assert len(step_args)==4
  events.append('trim_after_step_4');return 1
trim=Trim()
def libc(name):
 assert name=='libc.so.6';events.append('libc');return SimpleNamespace(malloc_trim=trim)
def memory_path(name):
 assert name=='/proc/self/smaps_rollup';return Memory()
main_def=next(n for n in ast.parse((ROOT/'scripts/train_candidate_host_warm_compact.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='main')
state=dict(ROOT=ROOT,sys=fake_sys,json=json,importlib=SimpleNamespace(import_module=import_module),Path=memory_path,
           gc=SimpleNamespace(collect=lambda:events.append('gc')),ctypes=SimpleNamespace(CDLL=libc,c_size_t='size_t',c_int='int'))
exec(compile(ast.Module(body=[main_def],type_ignores=[]),'<exact-warm-main-mocked>','exec'),state);state['main']()
assert events==['original_main','original_stage','original_step_1','original_step_2','original_step_3','original_step_4',
                'memory','gc','libc','trim_after_step_4','memory','write_warm_memory','original_step_5','original_step_6']
assert len(writes)==1 and writes[0]['optimizer_updates']==4 and fake_sys.argv==saved_argv
assert len(returned)==6 and all(a is b for a,b in zip(returned,sentinels))
startup=read('reports/HOST_ALLOCATOR_TRIM_STARTUP_MEMORY_20261027.json')['record']
pss_reduction=int(startup['before']['Pss'].split()[0])-int(startup['after']['Pss'].split()[0]);assert pss_reduction==82
checks=dict(status='PASS',hashes=expected,aggregate_sources=55,case_sources=52,original50_sources_24_inputs_sequence_unchanged=True,
 owner_count=16,owners_exact_snapshot_match=True,owners_per_gpu={g:4 for g in '0123'},snapshot_observed=measurement['observed'],
 controller_semantics_unchanged_from_trim=True,reachable_generated_programs_compiled=2,generated_argv=argv,
 default_reference_hash_bound=True,completed_reference_not_relaunched=True,case_changed_fields=changed,
 callback_fixture=dict(status='PASS',five_argument_identities=True,six_return_identities=True,stage_eight_argument_identities=True,
   argv_native_identity=True,first_three_no_trim=True,fourth_only_once=True,fifth_sixth_no_repeat=True,record_updates=4,events=events),
 inherited_optimizer_tolerance=dict(atol=1e-8,rtol=1e-6,finite_required=True,actual_bit_flag_and_max_error=True),
 archived_startup_pss_reduction_kib=pss_reduction,warm_effectiveness='UNTESTED',remote_queries=False,real_gc=False,real_libc=False,torch_import=False,neural_execution=False)
for name in list(expected)+['reports/VRAM18_WAVE4_EXECUTION_20261027.json','reports/HOST_ALLOCATOR_TRIM_STARTUP_MEMORY_20261027.json','reports/HOST_ALLOCATOR_PAIRED_TRIM_PROFILE_20261027.json']:
 p=TRACE/'inputs'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/name).read_bytes())
(TRACE/'verification.json').write_text(json.dumps(checks,indent=2)+'\n',encoding='utf-8')
print(json.dumps(checks,indent=2))

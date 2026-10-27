"""Synthetic R3 source fixtures; no imports of project code, torch, libc or remote execution."""
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

ROOT=Path('C:/Users/gb/projects/Dual-RGBT')
OUT=ROOT/'.aris/traces/experiment-audit/2026-10-27_vram18_source_review/round3'
read=lambda n:json.loads((ROOT/n).read_bytes())
index=read('reports/VRAM18_TRAINING_PLAN_INDEX_20261027.json')
spec=read('reports/VRAM18_EVALUATION_PROTOCOL_20261027.json')
remote_root='/data2/gb/Dual-RGBT';python=remote_root+'/conda-envs/dual_rgbt/bin/python'
tree=ast.parse((ROOT/'tools/launch_vram18_training_wave.py').read_text())

class MemoryPath:
    def __init__(self,key,store):self.key=str(key);self.store=store
    def __truediv__(self,key):return MemoryPath(self.key.rstrip('/')+'/'+str(key),self.store)
    def read_bytes(self):return self.store[self.key]
    def read_text(self):return self.read_bytes().decode()
    def write_text(self,text,**kwargs):self.store[self.key]=text.encode();return len(text)

# Exact compact wrapper main with inert import, GC, libc and filesystem objects.
wrapper=ast.parse((ROOT/'scripts/train_candidate_host_compact.py').read_text())
wrapper_main=next(n for n in wrapper.body if isinstance(n,ast.FunctionDef) and n.name=='main')
wrapper_results=[]
for objective,expected_module in [('joint_preview_relative_quality','train_candidate_joint_quality_trajectory'),('signed_gain_and_same_state_difference_regression','train_candidate_joint_gainpair_trajectory')]:
    for sanity in [True,False]:
        events=[];memory={};root=MemoryPath('root',memory)
        plan={'repair':{'kind':objective}};memory['root/plan.json']=json.dumps(plan).encode()
        memory['/proc/self/smaps_rollup']=b'SYNTHETIC ONLY\nPss: 1000 kB\nSwap: 0 kB\n'
        argv=['/data2/gb/Dual-RGBT/scripts/train_candidate_host_compact.py','--plan','plan.json','--run','SYNTHETIC']+(['--sanity'] if sanity else ['--sanity-complete-sha256','0'*64])
        saved_argv=list(argv);result_object=object()
        values=(object(),object(),plan,object(),root/'run',object(),object(),sanity)
        def original(*args):
            assert all(a is b for a,b in zip(args,values)) and len(args)==len(values)
            events.append('original_trajectory');return result_object
        module=SimpleNamespace(trajectory_stage=original)
        def original_main():
            events.append('original_main')
            assert module.trajectory_stage(*values) is result_object
        module.main=original_main
        def import_module(name):assert name==expected_module;events.append('import_original');return module
        def collect():events.append('gc_collect');return 0
        class Trim:
            def __call__(self,value):assert value==0;events.append('malloc_trim_0');return 1
        trim=Trim()
        def cdll(name):assert name=='libc.so.6';events.append('load_libc_stub');return SimpleNamespace(malloc_trim=trim)
        ns=dict(json=json,ROOT=root,Path=lambda p:MemoryPath(p,memory),sys=SimpleNamespace(argv=argv),
                importlib=SimpleNamespace(import_module=import_module),gc=SimpleNamespace(collect=collect),ctypes=SimpleNamespace(CDLL=cdll,c_size_t=object(),c_int=object()))
        exec(compile(ast.Module(body=[wrapper_main],type_ignores=[]),'exact_wrapper_main','exec'),ns)
        ns['main']();assert argv==saved_argv
        assert events==['import_original','original_main','gc_collect','load_libc_stub','malloc_trim_0','original_trajectory']
        receipt=json.loads(memory['root/run/host_memory_startup.json']);assert receipt['malloc_trim_return']==1
        wrapper_results.append(dict(objective=objective,sanity=sanity,status='PASS',same_object_arguments=True,argv_unchanged=True,one_gc_one_trim=True))

# Exact native metadata path checks for all original8 and new24 plans.
contract=ast.parse((ROOT/'scripts/joint_evaluation_contract.py').read_text())
metadata=next(n for n in contract.body if isinstance(n,ast.FunctionDef) and n.name=='training_metadata')
trainers={'joint_preview_relative_quality':'scripts/train_candidate_joint_quality_trajectory.py','signed_gain_and_same_state_difference_regression':'scripts/train_candidate_joint_gainpair_trajectory.py'}
prefixes=('executor.preview.','executor.quality.','query_gain.')
native_results=[]
for plan_path in spec['training_plan_sha256']:
    p=read(plan_path);memory={};root=MemoryPath('root',memory)
    def store(name,value):memory['root/'+name]=json.dumps(value).encode()
    memory['root/'+plan_path]=(ROOT/plan_path).read_bytes()
    shaf=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    run=p['runs']['formal'];directory='runs/'+run
    # This exists only in an in-memory dict and is deliberately not a checkpoint.
    memory['root/'+directory+'/latest.pth']=b'SYNTHETIC_FIXTURE_NOT_A_CHECKPOINT'
    config=dict(sanity=False,run=run,stage='candidate_c',plan=plan_path,plan_sha256=spec['training_plan_sha256'][plan_path],source_sha256=p['source_sha256'],input_sha256=p['input_sha256'],factory_checkpoint=p['factory_checkpoint'],max_candidates=p['max_candidates'],quality_rank_supervision='gap_weighted_margin',adapted_modules=list(prefixes),sanity_complete_sha256='0'*64)
    complete=dict(status='COMPLETE',sanity=False,**spec['formal_counts'],seen_train_sequences=p['sequence_names'],selected_execution_bit_equal=True,teacher_preserves_tracker_state=True,frozen_state_tensors=251,changed_state_tensors=['SYNTHETIC_%d'%i for i in range(14)],checkpoint_sha256=shaf(root/(directory+'/latest.pth')))
    trainer=p['launch_trainer'] if 'launch_trainer' in p else trainers[p['repair']['kind']]
    command=[python,remote_root+'/'+trainer,'--plan',plan_path,'--run',run,'--sanity-complete-sha256','0'*64]
    native=dict(state='COMPLETE',exit_code=0,task=p['tasks']['formal'],physical_gpu=str(p['physical_gpu']),command=command)
    native_path='runs/pipeline_tasks/'+p['tasks']['formal']+'/status.json'
    store(directory+'/config.json',config);store(directory+'/COMPLETE.json',complete);store(native_path,native)
    ns=dict(json=json,sha=shaf,PREFIXES=prefixes,TRAINERS=trainers,REMOTE_ROOT=remote_root,REMOTE_PYTHON=python)
    exec(compile(ast.Module(body=[metadata],type_ignores=[]),'exact_native_metadata','exec'),ns)
    ns['training_metadata'](root,spec,plan_path)
    wrong=copy.deepcopy(native);wrong['command'][1]=remote_root+'/'+('scripts/train_candidate_joint_quality_trajectory.py' if 'launch_trainer' in p else 'scripts/train_candidate_host_compact.py')
    store(native_path,wrong)
    rejected=False
    try:ns['training_metadata'](root,spec,plan_path)
    except AssertionError:rejected=True
    assert rejected
    native_results.append(dict(plan=plan_path,trainer=trainer,status='PASS',correct_path_accepted=True,wrong_path_rejected=True))

# Expand actual current launcher strings for each sequential case/stage in each wave.
launch=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='launch')
programs=[];captured=[]
def capture(code):ast.parse(code);captured.append(code);return {'source_only':True}
namespace=dict(ROOT=ROOT,Path=Path,json=json,REMOTE=remote_root,PYTHON=python,compact_peak=1,
    owned=[dict(e,cuda_ready=True) for e in index['original_owners']],temporary=[],remote=capture)
exec(compile(ast.Module(body=[launch],type_ignores=[]),'launch_builder','exec'),namespace)
witnesses=[n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='code' for t in n.targets) and isinstance(n.value,ast.BinOp) and n.value.left.value.startswith(('import hashlib,json\n','import csv,json,math,time\n'))]
slot_sequences={}
for phase in [4,6,8]:
    rows=[r for r in index['plans'] if r['phase']==phase]
    cases=[dict(plan=r['path'],plan_sha256=r['sha256'],gpu=r['gpu']) for r in rows]
    namespace.update(args=SimpleNamespace(phase=phase),cases=cases)
    for sanity in [True,False]:
        stage='sanity' if sanity else 'formal';slots=[]
        for row,case in zip(rows,cases):
            p=read(row['path']);namespace['launch'](case,sanity,None if sanity else '0'*64)
            code=captured[-1];parsed=ast.parse(code)
            call=next(n for n in ast.walk(parsed) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='Popen')
            command=ast.literal_eval(call.args[0])
            expected=[python,p['launch_wrapper'],'--task',p['tasks'][stage],'--gpu',str(row['gpu']),'--',python,remote_root+'/'+p['launch_trainer'],'--plan',row['path'],'--run',p['runs'][stage]]+(['--sanity'] if sanity else ['--sanity-complete-sha256','0'*64])
            assert command==expected
            slot=next(ast.literal_eval(n.value) for n in ast.walk(parsed) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='new_slots' for t in n.targets));slots.append(slot)
            programs.append(dict(plan=row['path'],kind=stage+'_launch',syntax='PASS',new_slots=slot,command=command,source=code))
            case[stage]=dict(task=p['tasks'][stage],pid=1,gpu=row['gpu'],command=command,launched=0)
        assert slots==list(range(8,0,-1));slot_sequences[str(phase)+'_'+stage]=slots
    for case in cases:
        for n in witnesses:
            code=eval(compile(ast.Expression(n.value),'witness_builder','eval'),dict(REMOTE=remote_root,c=case));ast.parse(code)
            programs.append(dict(plan=case['plan'],kind='sanity_admission' if n.value.left.value.startswith('import hashlib') else 'formal_startup',syntax='PASS',source=code))

# The R2 ownership and health-policy bodies are unchanged by direct source inspection;
# reuse those accepted focused fixtures rather than run unrelated repetitions.
ram_assert=next(n for n in ast.walk(ast.parse(programs[0]['source'])) if isinstance(n,ast.Assert) and 'inflight_extra' in ast.unparse(n.test) and 'available' in ast.unparse(n.test))
ram_code=compile(ast.Expression(ram_assert.test),'exact_wave_ram_guard','eval')
G=1024*1024
ram_inputs=[
 ('initial_current_wave',80*G,8,4*G,0,True),
 ('one_child_halfway_to_peak',78*G,7,4*G,2*G,True),
 ('same_reserved_total_at_peak',76*G,7,4*G,0,True),
 ('initializing_headroom_prevents_overadmission',20*G,1,4*G,14*G,False),
 ('exact_eight_gib_margin_rejected',12*G,1,4*G,0,False)]
ram_results=[]
for name,available,new_slots,peak,inflight_extra,expected in ram_inputs:
    actual=eval(ram_code,dict(available=available,new_slots=new_slots,peak=peak,inflight_extra=inflight_extra))
    assert actual==expected
    ram_results.append(dict(fixture=name,status='PASS',actual=actual,remaining_kib=available-new_slots*peak-inflight_extra))

# Compile the new profile's generated sampler and tolerance comparison, never execute.
profile=read(index['host_profile_plan'])
profiletree=ast.parse((ROOT/'tools/profile_host_allocator_trim.py').read_text())
codes=[n for n in ast.walk(profiletree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='code' for t in n.targets)]
for n in codes:
    code=eval(compile(ast.Expression(n.value),'profile_string_builder','eval'),dict(REMOTE=remote_root,case=profile['cases'][1],plan=profile,PYTHON=python))
    ast.parse(code);programs.append(dict(kind='trim_profile_sampler' if 'subprocess,time' in code else 'trim_profile_comparison',syntax='PASS',source=code))

# Actual launcher tolerance-gate assertions, with deliberately synthetic receipts.
main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
gate_nodes=[n for n in main.body if isinstance(n,ast.Assert) and 84<=n.lineno<=89]
gate_code=compile(ast.Module(body=gate_nodes,type_ignores=[]),'exact_profile_numeric_gates','exec')
receipt=dict(status='PAIRED_HOST_COMPACT_TRIM_M0_COMPLETE',cases=[dict(case={'name':n},native={'exit_code':0}) for n in ['default','compact']],equality=dict(state_tensors_bit_equal=265,optimizer_bit_equal=False,optimizer_tolerance_pass=True,optimizer_atol=1e-8,optimizer_rtol=1e-6,trajectory_csv_bit_equal=True,query_labels_bit_equal=True))
proof_results=[]
for name,changed,accept in [('tolerant_non_bit_exact_optimizer',{},True),('wrong_tolerance',{'optimizer_atol':1e-7},False),('nonexact_model',{'state_tensors_bit_equal':264},False),('csv_difference',{'query_labels_bit_equal':False},False),('optimizer_tolerance_failed',{'optimizer_tolerance_pass':False},False)]:
    value=copy.deepcopy(receipt);value['equality'].update(changed)
    okay=True
    try:exec(gate_code,dict(profile=value))
    except AssertionError:okay=False
    assert okay==accept
    proof_results.append(dict(fixture=name,status='PASS',accepted=okay))

result=dict(status='PASS',scope='SYNTHETIC_SOURCE_FIXTURES_ONLY_NO_RUNTIME_EVIDENCE',wrapper_fixtures=wrapper_results,native_path_fixtures=native_results,
    generated_programs=len(programs),argv_checked=48,new_slot_sequences=slot_sequences,ram_fixtures=ram_results,host_proof_gate_fixtures=proof_results,
    reused_R2_ownership_health_fixtures=True,remote_nn_libc_or_runtime_io=False)
(OUT/'generated_remote_programs.json').write_text(json.dumps(programs,indent=2)+'\n')
(OUT/'focused_regressions.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='native_path_fixtures'},indent=2))

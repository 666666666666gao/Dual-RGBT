"""Targeted external-scheduler source comparison and inert lifecycle fixtures."""
import ast
import contextlib
import copy
import difflib
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

ROOT=Path('C:/Users/gb/projects/Dual-RGBT')
OUT=ROOT/'.aris/traces/experiment-audit/2026-10-27_vram18_source_review/serial_controller_round1'
read=lambda n:json.loads((ROOT/n).read_bytes())
sha=lambda n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest()
serial_path='tools/launch_vram18_serial_training_wave.py';v1_path='tools/launch_vram18_training_wave.py'
serial=(ROOT/serial_path).read_text();v1=(ROOT/v1_path).read_text();st=ast.parse(serial);vt=ast.parse(v1)
r3=read('.aris/traces/experiment-audit/2026-10-27_vram18_source_review/round3/review_receipt.json')
index=read('reports/VRAM18_TRAINING_PLAN_INDEX_20261027.json')
assert sha('reports/VRAM18_TRAINING_PLAN_INDEX_20261027.json')==r3['index_sha256']=='e4ff359c6abc177d66d53da3416194f471ce42c5471d7984d062b78f94b61f44'
assert sha(v1_path)==r3['launcher_sha256']=='1335fa7e5015b09a47a908afc3a3a9a1f641fdf5f709541aa7a0694b30839832'
assert sha(serial_path)=='c1ef685e700b57d3a7754c4730edb889702abc235ab8b1b20e563eab217fb4b1'
assert all(sha(n)==h for n,h in r3['source_sha256'].items())
assert all(sha(n)==h for n,h in r3['training_plan_sha256'].items())
assert all(sha(n)==h for n,h in r3['case_review_sha256'].items())
assert all(sha(n)==h for n,h in r3['report_sha256'].items())
for name in ['measurement','score_health']:
    a=next(n for n in vt.body if isinstance(n,ast.FunctionDef) and n.name==name)
    b=next(n for n in st.body if isinstance(n,ast.FunctionDef) and n.name==name)
    assert ast.dump(a)==ast.dump(b)
sm=next(n for n in st.body if isinstance(n,ast.FunctionDef) and n.name=='main')
vm=next(n for n in vt.body if isinstance(n,ast.FunctionDef) and n.name=='main')
sf={n.name:n for n in ast.walk(sm) if isinstance(n,ast.FunctionDef)}
vf={n.name:n for n in ast.walk(vm) if isinstance(n,ast.FunctionDef)}
vlaunch=copy.deepcopy(vf['launch'])
for n in ast.walk(vlaunch):
    if isinstance(n,ast.Name) and n.id=='cases':n.id='active_batch'
assert ast.dump(vlaunch)==ast.dump(sf['launch'])
phase_call=next(n for n in ast.walk(sm) if isinstance(n,ast.Call) and n.args and isinstance(n.args[0],ast.Constant) and n.args[0].value=='--phase')
assert next(ast.literal_eval(k.value) for k in phase_call.keywords if k.arg=='choices')==[6]

vstrings=[n.value.left.value for n in ast.walk(vm) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='code' for t in n.targets)]
for name,prefix in [('admit','import hashlib,json\n'),('startup','import csv,json,math,time\n')]:
    expr=sf[name].body[0].value.args[0]
    actual=expr.left.value
    previous=next(s for s in vstrings if s.startswith(prefix))
    assert ast.dump(ast.parse(actual.replace('%r','None')))==ast.dump(ast.parse(previous.replace('%r','None')))

# Admission/proof/previous-health blocks are unchanged before the state constructor.
# Only phase argument and independent review path are intentionally different.
prefix_v=[n for n in vm.body if n.lineno<121]
prefix_s=[n for n in sm.body if n.lineno<121]
assert len(prefix_v)==len(prefix_s)
changed=[(a.lineno,b.lineno) for a,b in zip(prefix_v,prefix_s) if ast.dump(a)!=ast.dump(b)]
assert changed==[(70,70),(73,73)]

phase_rows=[r for r in index['plans'] if r['phase']==6]
assert len(phase_rows)==8
base_cases=[]
for row in phase_rows:
    p=read(row['path']);v=read(p['source_review'])
    assert v['plan_sha256']==row['sha256'] and v['source_sha256']==p['source_sha256'] and v['status']=='SOURCE_PASS' and v['blocking_issues']==0
    base_cases.append(dict(plan=row['path'],plan_sha256=row['sha256'],gpu=row['gpu'],runs=p['runs'],tasks=p['tasks']))

schedule=[n for n in sm.body if n.lineno>=164]
program_records=[]
def run_fixture(failure=None):
    cases=copy.deepcopy(base_cases);state=dict(state='SOURCE_ADMITTED_HOST_PROFILE_PASSED',phase=6,cases=cases)
    owned=[dict(e,cuda_ready=True) for e in index['original_owners']]
    owned += [dict(task='SYNTHETIC_PRIOR_PHASE4_%d'%i,pid=10000+i,gpu=i%4,cuda_ready=True) for i in range(8)]
    events=[];snapshots=[];clock=[0];pid=[20000];sanity_pending=set();formal_pending=set();launch_times={};launch_slots=[];formal_count=[0]
    class Report:
        def write_text(self,text):
            value=json.loads(text);snapshots.append(value)
            events.append(dict(event='persist',state=value['state'],sanities=sum('sanity'in c for c in value['cases']),formals=sum('formal'in c for c in value['cases']),time=clock[0]))
            return len(text)
    report=Report()
    def sleep(value):events.append(dict(event='sleep',seconds=value,time=clock[0]));clock[0]+=value
    def value_of(parsed,key):return next(ast.literal_eval(n.value) for n in ast.walk(parsed) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==key for t in n.targets))
    def remote(code):
        parsed=ast.parse(code);case=value_of(parsed,'case')
        islaunch=any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='Popen' for n in ast.walk(parsed))
        if islaunch:
            call=next(n for n in ast.walk(parsed) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='Popen')
            command=ast.literal_eval(call.args[0]);sanity=command[-1]=='--sanity';kind='sanity' if sanity else 'formal'
            nslots=value_of(parsed,'new_slots');launch_slots.append((kind,nslots))
            plan=read(case['plan']);expected=[namespace['PYTHON'],plan['launch_wrapper'],'--task',plan['tasks'][kind],'--gpu',str(case['gpu']),'--',namespace['PYTHON'],namespace['REMOTE']+'/'+plan['launch_trainer'],'--plan',case['plan'],'--run',plan['runs'][kind]]+(['--sanity'] if sanity else ['--sanity-complete-sha256','0'*64])
            assert command==expected
            if sanity:
                assert not formal_pending and len(sanity_pending)<4
                assert case['gpu'] not in {g for _,g in sanity_pending}
                if failure=='sanity_launch_4' and len([e for e in events if e['event']=='sanity_launch'])==3:raise AssertionError('SYNTHETIC_PRELAUNCH_CAPACITY_FAILURE')
                sanity_pending.add((case['plan'],case['gpu']))
            else:
                assert not sanity_pending and not formal_pending
                formal_pending.add(case['plan']);formal_count[0]+=1
                assert all(v['cuda_ready'] for v in namespace['owned'])
            pid[0]+=1;receipt=dict(task=plan['tasks'][kind],pid=pid[0],gpu=case['gpu'],command=command,launched=clock[0])
            launch_times[(case['plan'],kind)]=clock[0]
            events.append(dict(event=kind+'_launch',plan=case['plan'],gpu=case['gpu'],new_slots=nslots,time=clock[0]))
            if failure is None:program_records.append(dict(kind=kind+'_launch',plan=case['plan'],source=code,command=command))
            return receipt
        if code.startswith('import hashlib,json\n'):
            assert clock[0]-launch_times[(case['plan'],'sanity')]>=180
            if failure=='sanity_admit_group2' and len([e for e in events if e['event']=='sanity_admit'])==4:raise AssertionError('SYNTHETIC_NATIVE_NONZERO')
            sanity_pending.remove((case['plan'],case['gpu']))
            events.append(dict(event='sanity_admit',plan=case['plan'],gpu=case['gpu'],time=clock[0]))
            if failure is None:program_records.append(dict(kind='sanity_admission',plan=case['plan'],source=code))
            return dict(complete_sha256='0'*64,native={'state':'COMPLETE','exit_code':0})
        assert code.startswith('import csv,json,math,time\n')
        assert clock[0]-launch_times[(case['plan'],'formal')]>=90
        if failure=='formal_startup_2' and formal_count[0]==2:raise AssertionError('SYNTHETIC_STARTUP_FAILED')
        formal_pending.remove(case['plan'])
        events.append(dict(event='formal_startup',plan=case['plan'],gpu=case['gpu'],time=clock[0]))
        if failure is None:program_records.append(dict(kind='formal_startup',plan=case['plan'],source=code))
        return dict(child_pid=30000+formal_count[0],logged_updates=1,finite_logged_losses=True)
    def measurement(owners):
        assert not sanity_pending and not formal_pending and len(owners)==24 and all(v['cuda_ready'] for v in owners)
        events.append(dict(event='measurement',time=clock[0]));return dict(observed=clock[0],tasks=[])
    namespace=dict(ROOT=ROOT,Path=Path,json=json,REMOTE='/data2/gb/Dual-RGBT',PYTHON='/data2/gb/Dual-RGBT/conda-envs/dual_rgbt/bin/python',compact_peak=1,
        cases=cases,state=state,owned=owned,temporary=[],args=SimpleNamespace(phase=6),report_path=report,time=SimpleNamespace(sleep=sleep),remote=remote,measurement=measurement)
    exec(compile(ast.Module(body=[sf[n] for n in ['launch','admit','startup']],type_ignores=[]),'exact_serial_helpers','exec'),namespace)
    failed=False
    with contextlib.redirect_stdout(io.StringIO()):
        try:exec(compile(ast.Module(body=schedule,type_ignores=[]),'exact_serial_schedule','exec'),namespace)
        except AssertionError:
            if failure is None:raise
            failed=True
    # Every successful launch must be durable before another launch, sleep or admission.
    for i,event in enumerate(events):
        if event['event'] in ('sanity_launch','formal_launch'):
            assert events[i+1]['event']=='persist'
    if failure is None:
        assert not failed and state['state']=='FORMAL_STARTUP_PASS'
        assert [v for kind,v in launch_slots if kind=='sanity']==[4,3,2,1,4,3,2,1]
        assert [v for kind,v in launch_slots if kind=='formal']==[1]*8
        assert [e['gpu'] for e in events if e['event']=='sanity_launch']==[0,1,2,3,0,1,2,3]
        assert [e['gpu'] for e in events if e['event']=='formal_launch']==[0,1,2,3,0,1,2,3]
        assert len([e for e in events if e['event']=='formal_startup'])==8
    else:
        assert failed and snapshots and state['state']!='FORMAL_STARTUP_PASS'
        last=snapshots[-1]
        if failure=='sanity_launch_4':assert sum('sanity'in c for c in last['cases'])==3 and not any('formal'in c for c in last['cases'])
        if failure=='sanity_admit_group2':assert sum('sanity'in c for c in last['cases'])==8 and not any('formal'in c for c in last['cases'])
        if failure=='formal_startup_2':assert sum('formal'in c for c in last['cases'])==2 and sum('startup'in c for c in last['cases'])==1
    return dict(fixture=failure or 'full_success',status='PASS',synthetic_failure=failed,final_state=state['state'],events=events)

lifecycle=[run_fixture(n) for n in [None,'sanity_launch_4','sanity_admit_group2','formal_startup_2']]
assert len(program_records)==32
for row in program_records:ast.parse(row['source'])
ram=next(n for n in ast.walk(ast.parse(program_records[0]['source'])) if isinstance(n,ast.Assert) and 'available' in ast.unparse(n.test) and 'inflight_extra' in ast.unparse(n.test))
ram_code=compile(ast.Expression(ram.test),'unchanged_exact_ram_guard','eval');G=1024*1024
ram_results=[]
for name,values,expected in [('sanity_group4',dict(available=48*G,new_slots=4,peak=6*G,inflight_extra=0),True),('inflight_reserved',dict(available=30*G,new_slots=3,peak=6*G,inflight_extra=5*G),False),('one_formal',dict(available=15*G,new_slots=1,peak=6*G,inflight_extra=0),True),('one_formal_at_margin',dict(available=14*G,new_slots=1,peak=6*G,inflight_extra=0),False)]:
    actual=eval(ram_code,values);assert actual==expected;ram_results.append(dict(fixture=name,status='PASS',accepted=actual))

result=dict(status='PASS',scope='SYNTHETIC_SOURCE_ONLY_NO_RUNTIME_OR_PROFILE_QUERIES',index_sha256=sha('reports/VRAM18_TRAINING_PLAN_INDEX_20261027.json'),launcher_sha256=sha(serial_path),v1_launcher_sha256=sha(v1_path),
    source_maps_58_unchanged=True,all32_training_plans_unchanged=True,all24_case_guards_unchanged=True,all_R3_reports_unchanged=True,
    measurement_and_health_ast_exact=True,launch_ast_only_cases_to_active_batch=True,admit_startup_templates_ast_exact=True,
    preflight_only_phase_and_review_path_differ=True,phase_choices=[6],generated_programs=32,argv_count=16,
    lifecycle_fixtures=lifecycle,ram_fixtures=ram_results,
    warm_startup_scope='90s delay plus actual native/wrapper/child/argv/env/config identity and at least one contiguous finite-loss update; no >=4-update, steady-state memory or throughput guarantee.',runtime_certified=False)
(OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
(OUT/'generated_programs.json').write_text(json.dumps(program_records,indent=2)+'\n')
(OUT/'serial_vs_v1.diff').write_text(''.join(difflib.unified_diff(v1.splitlines(True),serial.splitlines(True),fromfile=v1_path,tofile=serial_path)))
(OUT/'launch_vram18_serial_training_wave.py').write_text(serial)
print(json.dumps({k:v for k,v in result.items() if k!='lifecycle_fixtures'},indent=2))

"""Stagger phase6 admissions within measured hostRAM; preserve all model/trainer recipes."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from complete_pipeline import REMOTE,PYTHON,SCP,remote
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def measurement(owned):
    code="""import csv,json,math,subprocess,time
from pathlib import Path
r=Path(%r);owners=%r;tasks=[]
for e in owners:
 n=json.loads((r/'runs/pipeline_tasks'/e['task']/'status.json').read_bytes());assert n['state']=='RUNNING' and n['pid']==e['pid']
 pp=Path('/proc')/str(n['pid']);argv=(pp/'cmdline').read_bytes().decode().split(chr(0))[:-1]
 children=(pp/'task'/str(n['pid'])/'children').read_text().split();assert len(children)==1
 cp=Path('/proc')/children[0];assert (cp/'cmdline').read_bytes().decode().split(chr(0))[:-1]==n['command']
 run=n['command'][n['command'].index('--run')+1]
 with (r/'runs'/run/'trajectory_steps.csv').open(newline='') as f:rows=[v for v in csv.DictReader(f) if None not in v.values()]
 assert rows and len(rows)==int(rows[-1]['step']) and all(all(math.isfinite(float(v[k])) for k in v if k=='loss' or k.endswith('_loss')) for v in rows)
 memory={v.split(':',1)[0]:v.split(':',1)[1].strip() for v in (cp/'smaps_rollup').read_text().splitlines()[1:]}
 tasks.append(dict(native=n,child_pid=int(children[0]),logged_updates=len(rows),finite_logged_losses=True,host_memory=memory,wrapper=argv[1],run=run))
mem={v.split(':',1)[0]:int(v.split()[1]) for v in Path('/proc/meminfo').read_text().splitlines()}
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],universal_newlines=True).strip().splitlines()
print(json.dumps(dict(observed=time.time(),tasks=tasks,meminfo_kib=mem,gpu_csv=gpu,scope='Actual same-wave progress/host-memory node, instantGPUutil, no weights/temperature/power.')))
"""%(REMOTE,owned)
    return remote(code)

def score_health(args,index,state_path):
    state=json.loads(state_path.read_bytes());assert state['state']=='FORMAL_STARTUP_PASS'
    assert state['index_sha256']==sha(ROOT/'reports/VRAM18_TRAINING_PLAN_INDEX_20261027.json')
    start=state['measurement_start'];end=json.loads((ROOT/args.health_node).read_bytes())
    interval=end['observed']-start['observed'];assert interval>=600
    before={t['native']['task']:t for t in start['tasks']};after={t['native']['task']:t for t in end['tasks']}
    assert set(before)==set(after) and len(after)==4*args.phase
    rates={}
    for name,a in before.items():
        b=after[name];assert b['native']['state']=='RUNNING' and b['native']['pid']==a['native']['pid']
        assert b['native']['command']==a['native']['command'] and b['child_pid']==a['child_pid'] and b['finite_logged_losses']
        assert b['logged_updates']>a['logged_updates'];rates[name]=(b['logged_updates']-a['logged_updates'])/interval
    a=json.loads((ROOT/index['baseline_previous']).read_bytes());b=json.loads((ROOT/index['baseline_reference']).read_bytes())
    assert sha(ROOT/index['baseline_previous'])==index['baseline_previous_sha256'] and sha(ROOT/index['baseline_reference'])==index['baseline_reference_sha256']
    old={t['native']['task']:t['logged_updates'] for t in a['tasks']};dt=b['observed']-a['observed']
    controls={t['task']:(t['updates']-old[t['task']])/dt for t in b['tasks']}
    assert set(controls)=={e['task'] for e in index['original_owners']}
    retained={n:rates[n]/v for n,v in controls.items()};reference=sum(controls.values())
    if args.phase>4:
        prev=json.loads((ROOT/('reports/VRAM18_WAVE%d_HEALTH_20261027.json'%(args.phase-2))).read_bytes());assert prev['status']=='ADMIT_NEXT_WAVE'
        reference=prev['aggregate_updates_per_second']
    swap_start=sum(int(t['host_memory']['Swap'].split()[0]) for t in before.values())
    swap_end=sum(int(t['host_memory']['Swap'].split()[0]) for t in after.values())
    resources=remote("import json\nfrom pathlib import Path\nm={v.split(':',1)[0]:int(v.split()[1]) for v in Path('/proc/meminfo').read_text().splitlines()}\nprint(json.dumps(m))")
    ratio=sum(rates.values())/reference
    okay=ratio>1.0 and min(retained.values())>=0.4 and swap_end-swap_start<=64*1024 and resources['MemAvailable']>=8*1024*1024
    result=dict(status='ADMIT_NEXT_WAVE' if okay else 'HOLD_CURRENT_WAVE',phase=args.phase,index_sha256=state['index_sha256'],execution_sha256=sha(state_path),
        raw_node=args.health_node,raw_node_sha256=sha(ROOT/args.health_node),interval_seconds=interval,aggregate_updates_per_second=sum(rates.values()),
        previous_wave_reference_updates_per_second=reference,aggregate_ratio=ratio,original_eight_rate_retention=retained,min_original_rate_retention=min(retained.values()),
        swap_growth_kib=swap_end-swap_start,mem_available_kib=resources['MemAvailable'],rates=rates,
        scope='Operational staged capacity decision across actual evolution states; not paired full-training speedup, accuracy or onlineFPS. No job is stopped or restarted by this decision.')
    out=ROOT/('reports/VRAM18_WAVE%d_HEALTH_20261027.json'%args.phase);assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',type=int,choices=[6],required=True);ap.add_argument('--health-node');args=ap.parse_args()
    index_path='reports/VRAM18_TRAINING_PLAN_INDEX_20261027.json'
    index=json.loads((ROOT/index_path).read_bytes())
    review=json.loads((ROOT/'reports/VRAM18_SERIAL_CONTROLLER_SOURCE_REVIEW_20261027.json').read_bytes())
    assert review['status']=='SOURCE_PASS' and review['blocking_issues']==0
    assert review['index_sha256']==sha(ROOT/index_path)
    assert review['launcher_sha256']==sha(Path(__file__))
    profile_path=ROOT/index['host_profile']
    profile=json.loads(profile_path.read_bytes())
    hp=json.loads((ROOT/index['host_profile_plan']).read_bytes());hr=json.loads((ROOT/hp['source_review']).read_bytes())
    assert hr['status']=='SOURCE_PASS' and hr['blocking_issues']==0 and hr['profile_plan_sha256']==sha(ROOT/index['host_profile_plan'])
    assert hr['source_sha256']==hp['source_sha256']
    assert profile['profile_plan_sha256']==hr['profile_plan_sha256'] and profile['source_sha256']==hp['source_sha256']
    assert [v['case'] for v in profile['cases']]==hp['cases']
    assert profile['status']=='PAIRED_HOST_COMPACT_TRIM_M0_COMPLETE'
    assert profile['equality']['state_tensors_bit_equal']==265 and profile['equality']['optimizer_tolerance_pass']
    assert profile['equality']['optimizer_atol']==1e-8 and profile['equality']['optimizer_rtol']==1e-6
    assert profile['equality']['trajectory_csv_bit_equal'] and profile['equality']['query_labels_bit_equal']
    assert [v['case']['name'] for v in profile['cases']]==['default','compact']
    assert all(v['native']['exit_code']==0 for v in profile['cases'])
    compact_peak=profile['cases'][1]['peak_sampled_pss_kib']
    assert compact_peak<profile['cases'][0]['peak_sampled_pss_kib']
    previous=[]
    owned=[dict(e,cuda_ready=True) for e in index['original_owners']]
    for phase in [4,6,8]:
        path=ROOT/('reports/VRAM18_WAVE%d_EXECUTION_20261027.json'%phase)
        if phase<args.phase:
            j=json.loads(path.read_bytes());assert j['state']=='FORMAL_STARTUP_PASS'
            health_path=ROOT/('reports/VRAM18_WAVE%d_HEALTH_20261027.json'%phase)
            health=json.loads(health_path.read_bytes());assert health['status']=='ADMIT_NEXT_WAVE' and health['index_sha256']==sha(ROOT/index_path) and health['execution_sha256']==sha(path)
            assert health['aggregate_ratio']>1.0 and health['min_original_rate_retention']>=0.4 and health['swap_growth_kib']<=64*1024 and health['mem_available_kib']>=8*1024*1024
            previous.append(j)
            owned.extend(dict(task=c['formal']['task'],pid=c['formal']['pid'],gpu=c['gpu'],cuda_ready=True) for c in j['cases'])
    report_path=ROOT/('reports/VRAM18_WAVE%d_EXECUTION_20261027.json'%args.phase)
    if args.health_node:
        score_health(args,index,report_path);return
    assert not report_path.exists()
    cases=[]
    for c in index['plans']:
        if c['phase']!=args.phase:continue
        p=json.loads((ROOT/c['path']).read_bytes());v=json.loads((ROOT/p['source_review']).read_bytes())
        assert sha(ROOT/c['path'])==c['sha256'] and v['status']=='SOURCE_PASS' and v['blocking_issues']==0
        assert v['plan_sha256']==c['sha256'] and v['source_sha256']==p['source_sha256']
        for n,h in p['source_sha256'].items():assert sha(ROOT/n)==h
        cases.append(dict(plan=c['path'],plan_sha256=c['sha256'],gpu=p['physical_gpu'],runs=p['runs'],tasks=p['tasks']))
    assert len(cases)==8
    for name in ['tools/launch_vram18_training_wave.py','tools/register_vram18_training_plans.py',index_path,index['source_review']]:
        subprocess.run(SCP+[str(ROOT/name),'2026:'+REMOTE+'/'+name],check=True,timeout=300)
    for c in cases:
        p=json.loads((ROOT/c['plan']).read_bytes())
        for name in [c['plan'],p['source_review']]:subprocess.run(SCP+[str(ROOT/name),'2026:'+REMOTE+'/'+name],check=True,timeout=300)
    state=dict(controller_source='tools/launch_vram18_serial_training_wave.py',controller_sha256=sha(Path(__file__)),startup_seconds_per_new_formal=90,maximum_simultaneous_new_sanities=4,state='SOURCE_ADMITTED_HOST_PROFILE_PASSED',phase=args.phase,index_sha256=sha(ROOT/index_path),host_profile_sha256=sha(profile_path),cases=cases)
    report_path.write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(dict(state=state['state'],phase=args.phase,cases=len(cases))),flush=True)
    temporary=[]
    def launch(c,sanity,complete_sha=None):
        plan=json.loads((ROOT/c['plan']).read_bytes());stage='sanity' if sanity else 'formal'
        trainer=plan['launch_trainer']
        command=[PYTHON,plan['launch_wrapper'],'--task',plan['tasks'][stage],'--gpu',str(c['gpu']),'--',PYTHON,REMOTE+'/'+trainer,'--plan',c['plan'],'--run',plan['runs'][stage]]
        command+=['--sanity'] if sanity else ['--sanity-complete-sha256',complete_sha]
        code="""import hashlib,json,os,subprocess,time
from pathlib import Path
r=Path(%r);case=%r;owned=%r;plan=json.loads((r/case['plan']).read_bytes());gpu=case['gpu'];limit=%r;peak=%r;new_slots=%r
assert hashlib.sha256((r/case['plan']).read_bytes()).hexdigest()==case['plan_sha256']
expected=set();required=set();active_count=0;inflight_extra=0
for row in owned:
 n=json.loads((r/'runs/pipeline_tasks'/row['task']/'status.json').read_bytes());assert n['pid']==row['pid']
 assert n['state']=='RUNNING' or (n['state']=='COMPLETE' and n['exit_code']==0)
 if n['state']=='COMPLETE':continue
 active_count+=1
 if not row['cuda_ready']:
  pp=Path('/proc')/str(n['pid']);children=(pp/'task'/str(n['pid'])/'children').read_text().split();assert len(children)==1
  memory={v.split(':',1)[0]:v.split(':',1)[1].strip() for v in (Path('/proc')/children[0]/'smaps_rollup').read_text().splitlines()[1:]}
  inflight_extra+=max(0,peak-int(memory['Pss'].split()[0]))
 if int(row['gpu'])!=gpu:continue
 pp=Path('/proc')/str(n['pid']);children=(pp/'task'/str(n['pid'])/'children').read_text().split();assert len(children)==1
 cp=Path('/proc')/children[0];assert (cp/'cmdline').read_bytes().decode().split(chr(0))[:-1]==n['command'];expected.add(int(children[0]))
 if row['cuda_ready']:required.add(int(children[0]))
uuid=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid','--format=csv,noheader'],universal_newlines=True).strip()
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],universal_newlines=True)
actual={int(v.split(',')[1]) for v in apps.splitlines() if v.split(',')[0].strip()==uuid}
assert required<=actual<=expected and len(expected)<limit,(actual,expected,required,limit)
used=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],universal_newlines=True).strip());assert used+3500<24*1024
available=int(next(v.split()[1] for v in Path('/proc/meminfo').read_text().splitlines() if v.startswith('MemAvailable:')))
assert available-new_slots*peak-inflight_extra>8*1024*1024,(available,new_slots,peak,inflight_extra)
assert not (r/'runs/pipeline_tasks'/%r).exists() and not (r/'runs'/%r).exists()
env=dict(os.environ);env.update(plan['execution_env'])
with (r/'reports'/(%r+'.launcher.log')).open('wb') as log:p=subprocess.Popen(%r,cwd=str(r),env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(task=%r,pid=p.pid,gpu=gpu,command=%r,launched=time.time(),verified_coexisting_gpu_pids=sorted(actual),available_kib_before_launch=available)))
"""%(REMOTE,c,owned+temporary,args.phase,compact_peak,sum(stage not in z for z in active_batch),plan['tasks'][stage],plan['runs'][stage],plan['tasks'][stage],command,plan['tasks'][stage],command)
        return remote(code)
    def admit(c):
        return remote("import hashlib,json\nfrom pathlib import Path\nr=Path(%r);case=%r;p=json.loads((r/case['plan']).read_bytes());d=r/'runs'/p['runs']['sanity']\nn=json.loads((r/p['sanity_native_status']).read_bytes());a=json.loads((d/'COMPLETE.json').read_bytes());s=json.loads((d/'SANITY.json').read_bytes());cfg=json.loads((d/'config.json').read_bytes())\nassert n['state']=='COMPLETE' and n['exit_code']==0 and n['physical_gpu']==str(case['gpu'])\nassert a['sanity'] and {k:a[k] for k in p['sanity_expected_counts']}==p['sanity_expected_counts']\nassert a['selected_execution_bit_equal'] and a['teacher_preserves_tracker_state'] and s['status']=='PASS' and s['strict_reload_prediction_bit_equal']\nassert s['frozen_state_tensors']==251 and len(s['changed_state_tensors'])==14\nassert cfg['plan_sha256']==case['plan_sha256'] and cfg['source_sha256']==p['source_sha256'] and cfg['input_sha256']==p['input_sha256']\nprint(json.dumps(dict(native=n,complete=a,probe=s,complete_sha256=hashlib.sha256((d/'COMPLETE.json').read_bytes()).hexdigest())))\n"%(REMOTE,c))
    def startup(c):
        return remote("import csv,json,math,time\nfrom pathlib import Path\nr=Path(%r);case=%r;p=json.loads((r/case['plan']).read_bytes());n=json.loads((r/'runs/pipeline_tasks'/p['tasks']['formal']/'status.json').read_bytes())\nassert n['state']=='RUNNING' and n['pid']==case['formal']['pid'] and n['command']==case['formal']['command'][7:]\npp=Path('/proc')/str(n['pid']);assert (pp/'cmdline').read_bytes().decode().split(chr(0))[:-1]==case['formal']['command']\nchildren=(pp/'task'/str(n['pid'])/'children').read_text().split();assert len(children)==1\ncp=Path('/proc')/children[0];assert (cp/'cmdline').read_bytes().decode().split(chr(0))[:-1]==n['command']\nenv=dict(v.split('=',1) for v in (cp/'environ').read_bytes().decode().split(chr(0)) if v);assert all(env[k]==v for k,v in p['execution_env'].items())\nd=r/'runs'/p['runs']['formal'];cfg=json.loads((d/'config.json').read_bytes());assert not cfg['sanity'] and cfg['plan_sha256']==case['plan_sha256']\nwith (d/'trajectory_steps.csv').open(newline='') as f:rows=[v for v in csv.DictReader(f) if None not in v.values()]\nassert rows and len(rows)==int(rows[-1]['step']) and all(all(math.isfinite(float(v[k])) for k in v if k=='loss' or k.endswith('_loss')) for v in rows)\nmem={v.split(':',1)[0]:v.split(':',1)[1].strip() for v in (cp/'smaps_rollup').read_text().splitlines()[1:]}\nprint(json.dumps(dict(observed=time.time(),native=n,child_pid=int(children[0]),logged_updates=len(rows),last_update=rows[-1],finite_logged_losses=True,host_memory=mem,total_parameters=103218887)))\n"%(REMOTE,c))
    by_gpu={g:[c for c in cases if c['gpu']==g] for g in range(4)}
    assert all(len(v)==2 for v in by_gpu.values())
    for slot in range(2):
        active_batch=[by_gpu[g][slot] for g in range(4)]
        for c in active_batch:
            c['sanity']=launch(c,True);temporary.append(dict(task=c['sanity']['task'],pid=c['sanity']['pid'],gpu=c['gpu'],cuda_ready=False))
            report_path.write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(dict(state='SANITY_IDENTITY_PERSISTED',task=c['sanity']['task'],pid=c['sanity']['pid'])),flush=True);time.sleep(2)
        state['state']='SANITY_BATCH%d_LAUNCHED'%slot;report_path.write_text(json.dumps(state,indent=2)+'\n')
        time.sleep(180)
        for c in active_batch:c['sanity_admission']=admit(c)
        state['state']='SANITY_BATCH%d_COMPLETE'%slot;report_path.write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(dict(state=state['state'])),flush=True)
    formal_order=[by_gpu[g][slot] for slot in range(2) for g in range(4)]
    for c in formal_order:
        active_batch=[c]
        c['formal']=launch(c,False,c['sanity_admission']['complete_sha256']);owned.append(dict(task=c['formal']['task'],pid=c['formal']['pid'],gpu=c['gpu'],cuda_ready=False))
        state['state']='FORMAL_IDENTITY_PERSISTED';report_path.write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(dict(state=state['state'],task=c['formal']['task'],pid=c['formal']['pid'])),flush=True)
        time.sleep(90)
        c['startup']=startup(c);owned[-1]['cuda_ready']=True
        state['state']='FORMAL_WARM_STARTUP_PASSED';report_path.write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(dict(state=state['state'],task=c['formal']['task'],updates=c['startup']['logged_updates'])),flush=True)
    state['measurement_start']=measurement(owned)
    state['state']='FORMAL_STARTUP_PASS';state['next_observation_epoch']=max(c['formal']['launched'] for c in cases)+1200
    report_path.write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(dict(state=state['state'],phase=args.phase,cases=[dict(gpu=c['gpu'],pid=c['formal']['pid'],child_pid=c['startup']['child_pid'],updates=c['startup']['logged_updates']) for c in cases],next_observation_epoch=state['next_observation_epoch'])),flush=True)

if __name__=='__main__':main()

import argparse,hashlib,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from complete_pipeline import REMOTE,PYTHON,SCP,remote
parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['k8cost0','k4cost05','reserved23'],required=True);args=parser.parse_args()
capacity=json.loads((ROOT/'reports/TRAIN_CONCURRENCY_CAPACITY_DECISION_20261026.json').read_bytes())
assert capacity['status']=='ADMIT_BOUNDED_CONFIG_TRAINING' and capacity['result_audit_blocking_issues']==0
assert capacity['max_formal_tasks_per_gpu01'] >= {'k8cost0':2,'k4cost05':3,'reserved23':1}[args.phase]
assert hashlib.sha256((ROOT/capacity['result_audit']).read_bytes()).hexdigest()==capacity['result_audit_sha256']
index=json.loads((ROOT/'reports/CONCURRENT_TRAIN_PLAN_INDEX_20261026.json').read_bytes())
token={'k8cost0':'_k8_cost0_','k4cost05':'_k4_cost0p05_','reserved23':'_k4_cost0_'}[args.phase]
items=[v for v in index['plans'] if token in v['path']];assert len(items)==2
report=ROOT/('reports/CONCURRENT_TRAIN_WAVE_'+args.phase.upper()+'_EXECUTION_20261026.json');assert not report.exists()
state=dict(phase=args.phase,state='SOURCE_ADMITTED_HARDWARE_PENDING',capacity_decision_sha256=hashlib.sha256((ROOT/'reports/TRAIN_CONCURRENCY_CAPACITY_DECISION_20261026.json').read_bytes()).hexdigest(),cases=[])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for v in items:
 plan=json.loads((ROOT/v['path']).read_bytes());review=json.loads((ROOT/plan['source_review']).read_bytes())
 assert sha(ROOT/v['path'])==v['sha256'] and review['status']=='SOURCE_PASS' and review['blocking_issues']==0
 assert review['plan_sha256']==v['sha256'] and review['source_sha256']==plan['source_sha256']
 files=[v['path'],plan['source_review'],plan['registered_grid']]
 if plan['physical_gpu']>=2:files.append(plan['launch_wrapper'])
 for name in files:subprocess.run(SCP+[str(ROOT/name),'2026:'+REMOTE+'/'+name],check=True,timeout=300)
 state['cases'].append(dict(plan=v['path'],plan_sha256=v['sha256'],gpu=plan['physical_gpu'],runs=plan['runs'],tasks=plan['tasks']))
report.write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(state),flush=True)

def owned_formals():
 rows=[]
 for name in ['reports/JOINT_QUALITY_TRAIN_EXECUTION_20261026.json','reports/JOINT_GAINPAIR_TRAIN_EXECUTION_20261026.json']:
  j=json.loads((ROOT/name).read_bytes());v=j['formal'];rows.append(dict(task=v['task'],pid=v['pid'],gpu=v['gpu']))
 for name in ['k8cost0','k4cost05','reserved23']:
  p=ROOT/('reports/CONCURRENT_TRAIN_WAVE_'+name.upper()+'_EXECUTION_20261026.json')
  if p.exists():
   j=json.loads(p.read_bytes())
   rows.extend(dict(task=v['formal']['task'],pid=v['formal']['pid'],gpu=v['gpu']) for v in j['cases'] if 'formal' in v)
 return rows

def launch(case,sanity,sanity_sha=None):
 plan=json.loads((ROOT/case['plan']).read_bytes());stage='sanity' if sanity else 'formal';gpu=case['gpu'];task=plan['tasks'][stage]
 trainer='scripts/train_candidate_'+('joint_gainpair' if 'joint_gainpair' in case['plan'] else 'joint_quality')+'_trajectory.py'
 command=[PYTHON,plan['launch_wrapper'],'--task',task,'--gpu',str(gpu),'--',PYTHON,REMOTE+'/'+trainer,'--plan',case['plan'],'--run',plan['runs'][stage]]
 command+=['--sanity'] if sanity else ['--sanity-complete-sha256',sanity_sha]
 code="""import hashlib,json,os,subprocess,time
from pathlib import Path
r=Path(%r);case=%r;owned=%r;plan=json.loads((r/case['plan']).read_bytes());gpu=case['gpu']
assert hashlib.sha256((r/case['plan']).read_bytes()).hexdigest()==case['plan_sha256']
expected=set()
for row in owned:
 if row['gpu']!=gpu:continue
 n=json.loads((r/'runs/pipeline_tasks'/row['task']/'status.json').read_bytes());assert n['state']=='RUNNING' and n['pid']==row['pid']
 children=(Path('/proc')/str(n['pid'])/'task'/str(n['pid'])/'children').read_text().split();assert len(children)==1
 p=Path('/proc')/children[0];assert (p/'cmdline').read_bytes().decode().split(chr(0))[:-1]==n['command'];expected.add(int(children[0]))
uuid=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
actual={int(v.split(',')[1]) for v in apps.splitlines() if v.split(',')[0].strip()==uuid};assert actual==expected,(actual,expected)
if gpu>=2:
 assert not (Path('/proc')/str({2:1028233,3:1028232}[gpu])).exists()
 assert int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<500
else:assert len(expected)<%r
assert not (r/'runs/pipeline_tasks'/%r).exists() and not (r/'runs'/%r).exists()
with (r/'reports'/(%r+'.launcher.log')).open('wb') as log:
 p=subprocess.Popen(%r,cwd=str(r),env=dict(os.environ,OPENCV_FOR_THREADS_NUM='1'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(task=%r,pid=p.pid,gpu=gpu,command=%r,launched=time.time(),verified_coexisting_gpu_pids=sorted(actual))))
"""%(REMOTE,case,owned_formals(),capacity['max_formal_tasks_per_gpu01'],task,plan['runs'][stage],task,command,task,command)
 return remote(code)

for case in state['cases']:case['sanity']=launch(case,True)
state['state']='SANITIES_LAUNCHED';report.write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(state),flush=True)
time.sleep(180)
for case in state['cases']:
 code="""import hashlib,json
from pathlib import Path
r=Path(%r);case=%r;plan=json.loads((r/case['plan']).read_bytes());d=r/'runs'/plan['runs']['sanity']
n=json.loads((r/plan['sanity_native_status']).read_bytes());assert n['state']=='COMPLETE' and n['exit_code']==0 and n['physical_gpu']==str(case['gpu'])
c=json.loads((d/'COMPLETE.json').read_bytes());s=json.loads((d/'SANITY.json').read_bytes());cfg=json.loads((d/'config.json').read_bytes())
assert c['sanity'] and {k:c[k] for k in plan['sanity_expected_counts']}==plan['sanity_expected_counts']
assert c['selected_execution_bit_equal'] and c['teacher_preserves_tracker_state']
assert s['status']=='PASS' and s['strict_reload_prediction_bit_equal'] and s['four_request_legacy_bit_equal'] and s['frozen_state_tensors']==251 and len(s['changed_state_tensors'])==14
assert cfg['plan_sha256']==case['plan_sha256'] and cfg['source_sha256']==plan['source_sha256'] and cfg['input_sha256']==plan['input_sha256']
print(json.dumps(dict(native=n,complete=c,probe=s,complete_sha256=hashlib.sha256((d/'COMPLETE.json').read_bytes()).hexdigest())))
"""%(REMOTE,case)
 case['sanity_admission']=remote(code);case['formal']=launch(case,False,case['sanity_admission']['complete_sha256']);report.write_text(json.dumps(state,indent=2)+'\n')
state['state']='SANITY_COMPLETE_FORMALS_LAUNCHED';report.write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(state),flush=True)
time.sleep(180)
for case in state['cases']:
 code="""import csv,json,math,time
from pathlib import Path
r=Path(%r);case=%r;p=json.loads((r/case['plan']).read_bytes());n=json.loads((r/'runs/pipeline_tasks'/p['tasks']['formal']/'status.json').read_bytes())
assert n['state']=='RUNNING' and n['pid']==case['formal']['pid'] and n['command']==case['formal']['command'][7:]
parent=Path('/proc')/str(n['pid']);assert (parent/'cmdline').read_bytes().decode().split(chr(0))[:-1]==case['formal']['command']
children=(parent/'task'/str(n['pid'])/'children').read_text().split();assert len(children)==1
child=Path('/proc')/children[0];assert (child/'cmdline').read_bytes().decode().split(chr(0))[:-1]==n['command']
env=dict(v.split('=',1) for v in (child/'environ').read_bytes().decode().split(chr(0)) if v);assert env['CUDA_VISIBLE_DEVICES']==str(case['gpu']) and env['OPENCV_FOR_THREADS_NUM']=='1'
d=r/'runs'/p['runs']['formal'];cfg=json.loads((d/'config.json').read_bytes());assert not cfg['sanity'] and cfg['plan_sha256']==case['plan_sha256']
with (d/'trajectory_steps.csv').open(newline='') as f:rows=[v for v in csv.DictReader(f) if None not in v.values()]
assert rows and len(rows)==int(rows[-1]['step']) and all(all(math.isfinite(float(v[k])) for k in v if k=='loss' or k.endswith('_loss')) for v in rows)
print(json.dumps(dict(observed=time.time(),native=n,child_pid=int(children[0]),logged_updates=len(rows),last_update=rows[-1],finite_logged_losses=True,source_input_model_admission_passed=True,total_parameters=103218887)))
"""%(REMOTE,case)
 case['startup']=remote(code)
state['state']='FORMAL_STARTUP_PASS';state['next_observation_epoch']=max(v['formal']['launched'] for v in state['cases'])+1800
report.write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(dict(state=state['state'],cases=[dict(gpu=v['gpu'],formal_pid=v['formal']['pid'],child_pid=v['startup']['child_pid'],updates=v['startup']['logged_updates']) for v in state['cases']],next_observation_epoch=state['next_observation_epoch'])),flush=True)

"""Paired real four-update admissions with default and compact glibc allocation."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from complete_pipeline import REMOTE,PYTHON,SCP,SSH,remote
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);args=parser.parse_args()
    plan=json.loads((ROOT/args.plan).read_bytes());review=json.loads((ROOT/plan['source_review']).read_bytes())
    assert review['status']=='SOURCE_PASS' and review['blocking_issues']==0
    assert review['profile_plan_sha256']==sha(ROOT/args.plan) and review['source_sha256']==plan['source_sha256']
    assert all(sha(ROOT/n)==h for n,h in plan['source_sha256'].items())
    out=ROOT/plan['output'];assert not out.exists()
    for case in plan['cases']:
        p=json.loads((ROOT/case['plan']).read_bytes());v=json.loads((ROOT/p['source_review']).read_bytes())
        assert sha(ROOT/case['plan'])==case['plan_sha256'] and v['status']=='SOURCE_PASS' and v['blocking_issues']==0
        assert v['plan_sha256']==case['plan_sha256'] and v['source_sha256']==p['source_sha256']
        for name in [case['plan'],p['source_review']]:subprocess.run(SCP+[str(ROOT/name),'2026:'+REMOTE+'/'+name],check=True,timeout=300)
    results=[]
    for case in plan['cases']:
        code="""import csv,hashlib,json,os,signal,subprocess,time
from pathlib import Path
r=Path(%r);case=%r;owners=%r; p=json.loads((r/case['plan']).read_bytes());gpu=p['physical_gpu']
assert hashlib.sha256((r/case['plan']).read_bytes()).hexdigest()==case['plan_sha256']
expected=set()
for e in owners:
 if int(e['gpu'])!=gpu:continue
 n=json.loads((r/'runs/pipeline_tasks'/e['task']/'status.json').read_bytes());assert n['state']=='RUNNING' and n['pid']==e['pid']
 c=Path('/proc')/str(e['child_pid']);assert c.exists() and (c/'cmdline').read_bytes().decode().split(chr(0))[:-1]==e['command'];expected.add(e['child_pid'])
uuid=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid','--format=csv,noheader'],universal_newlines=True).strip()
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],universal_newlines=True)
actual={int(v.split(',')[1]) for v in apps.splitlines() if v.split(',')[0].strip()==uuid};assert actual==expected
assert int(next(v.split()[1] for v in Path('/proc/meminfo').read_text().splitlines() if v.startswith('MemAvailable:')))>12*1024*1024
assert not (r/'runs'/p['runs']['sanity']).exists() and not (r/'runs/pipeline_tasks'/p['tasks']['sanity']).exists()
alloc=['MALLOC_ARENA_MAX','MALLOC_MMAP_THRESHOLD_','MALLOC_TRIM_THRESHOLD_']
env={k:v for k,v in os.environ.items() if k not in alloc};env.update(p['execution_env'])
trainer='scripts/train_candidate_joint_quality_trajectory.py'
command=[%r,p['launch_wrapper'],'--task',p['tasks']['sanity'],'--gpu',str(gpu),'--',%r,str(r/trainer),'--plan',case['plan'],'--run',p['runs']['sanity'],'--sanity']
with (r/'reports'/(p['tasks']['sanity']+'.launcher.log')).open('wb') as log:child=subprocess.Popen(command,cwd=str(r),env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
success=False;deadline=time.monotonic()+540
try:
 samples=[];native_path=r/'runs/pipeline_tasks'/p['tasks']['sanity']/'status.json'
 time.sleep(2)
 while True:
  assert time.monotonic()<deadline,'PROFILE_REMOTE_DEADLINE_540_SECONDS'
  n=json.loads(native_path.read_bytes());assert n['pid']==child.pid
  if n['state']!='RUNNING':break
  assert child.poll() is None,'PROFILE_WRAPPER_EXITED_WITHOUT_TERMINAL_RECEIPT'
  children=(Path('/proc')/str(child.pid)/'task'/str(child.pid)/'children').read_text().split()
  if children:
   assert len(children)==1;c=Path('/proc')/children[0]
   if c.exists():
    fields={v.split(':',1)[0]:v.split(':',1)[1].strip() for v in (c/'smaps_rollup').read_text().splitlines()[1:]}
    environment=dict(v.split('=',1) for v in (c/'environ').read_bytes().decode().split(chr(0)) if v)
    assert all(environment[k]==v for k,v in p['execution_env'].items())
    samples.append(dict(observed=time.time(),pid=int(children[0]),rss_kib=int(fields['Rss'].split()[0]),pss_kib=int(fields['Pss'].split()[0]),private_dirty_kib=int(fields['Private_Dirty'].split()[0]),swap_kib=int(fields['Swap'].split()[0])))
  time.sleep(2)
 assert n['state']=='COMPLETE' and n['exit_code']==0 and child.wait()==0
 complete=json.loads((r/'runs'/p['runs']['sanity']/'COMPLETE.json').read_bytes());probe=json.loads((r/'runs'/p['runs']['sanity']/'SANITY.json').read_bytes())
 assert complete['sanity'] and complete['updates']==4 and complete['tracked_frames']==20 and probe['status']=='PASS' and probe['strict_reload_prediction_bit_equal']
 assert samples
 print(json.dumps(dict(case=case,native=n,complete=complete,probe=probe,samples=samples,peak_sampled_pss_kib=max(v['pss_kib'] for v in samples),last_sample_pss_kib=samples[-1]['pss_kib'],scope='2-second sampled whole realM0 process PSS including reload clone; not continuous peak or formal steady-state.')))
 success=True
finally:
 if not success:
  killed=False
  if child.poll() is None:
   os.killpg(child.pid,signal.SIGKILL);killed=True
  code=child.wait()
  failure=dict(status='PROFILE_FAILED_OWNED_GROUP_REAPED',wrapper_pid=child.pid,wrapper_exit_code=code,killed_owned_profile_group=killed,observed=time.time(),scope='Disposable profile group only; no original training touched and no retry.')
  (r/'reports'/(p['tasks']['sanity']+'.profile_failure.json')).write_text(json.dumps(failure,indent=2))
"""%(REMOTE,case,plan['owners'],PYTHON,PYTHON)
        raw=subprocess.run(SSH,input=code,text=True,encoding='utf-8',capture_output=True,check=False,timeout=600)
        transcript=ROOT/('reports/HOST_ALLOCATOR_'+case['name'].upper()+'_SSH_20261027.json');assert not transcript.exists()
        transcript.write_text(json.dumps(dict(stdout=raw.stdout,stderr=raw.stderr,ssh_exit_code=raw.returncode),indent=2)+'\n')
        assert raw.returncode==0,raw.stderr
        result=json.loads(raw.stdout);results.append(result)
        (ROOT/case['sample_output']).write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(dict(case=case['name'],state=result['native']['state'],peak_sampled_pss_kib=result['peak_sampled_pss_kib'],last_sample_pss_kib=result['last_sample_pss_kib'])),flush=True)
    code="""import json,torch
from pathlib import Path
r=Path(%r);cases=%r;torch.set_num_threads(1)
paths=[r/'runs'/json.loads((r/c['plan']).read_bytes())['runs']['sanity'] for c in cases]
a,b=[torch.load(p/'latest.pth',map_location='cpu') for p in paths]
assert tuple(a['net'])==tuple(b['net']) and len(a['net'])==265
assert all(torch.equal(a['net'][n],b['net'][n]) for n in a['net'])
def equal(a,b):
 if isinstance(a,torch.Tensor):return torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return a==b
assert equal(a['optimizer'],b['optimizer'])
for name in ['trajectory_steps.csv','query_labels.csv']:assert (paths[0]/name).read_bytes()==(paths[1]/name).read_bytes()
print(json.dumps(dict(state_tensors_bit_equal=265,optimizer_bit_equal=True,trajectory_csv_bit_equal=True,query_labels_bit_equal=True,scope='Four actual GPU updates/20trackingframes only; not full training equivalence or accuracy.')))
"""%(REMOTE,plan['cases'])
    equality=remote(code)
    report=dict(status='PAIRED_HOST_ALLOCATOR_M0_COMPLETE',profile_plan_sha256=sha(ROOT/args.plan),cases=results,equality=equality,source_sha256=plan['source_sha256'],scope=plan['scope'])
    out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(status=report['status'],equality=equality)),flush=True)

if __name__=='__main__':main()

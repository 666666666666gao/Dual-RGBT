import csv,hashlib,json,os,signal,subprocess,time
from pathlib import Path
r=Path('/data2/gb/Dual-RGBT');case={'name': 'compact', 'plan': 'reports/host_allocator_compact_plan_20261027.json', 'plan_sha256': '08c5a5ec9d32c812fb966674d0ba60f11ac14f5bf99cb5555c9a95e37449c2cb', 'sample_output': 'reports/HOST_ALLOCATOR_COMPACT_SAMPLES_20261027.json'};owners=[{'task': 'train_global_candidate_joint_quality_trajectory_formal_20261026', 'pid': 828655, 'child_pid': 828658, 'gpu': '0', 'command': ['/data2/gb/Dual-RGBT/conda-envs/dual_rgbt/bin/python', '/data2/gb/Dual-RGBT/scripts/train_candidate_joint_quality_trajectory.py', '--plan', 'reports/global_candidate_joint_quality_trajectory_plan_20261026.json', '--run', 'global_candidate_joint_quality_trajectory_formal_20261026', '--sanity-complete-sha256', 'f4d568cf2d48268afc5be16efc4435459dff07f8fdefc031a2d74f98592387e8'], 'kind': 'train', 'run': 'global_candidate_joint_quality_trajectory_formal_20261026', 'wrapper': 'scripts/run_task.py'}, {'task': 'train_global_candidate_joint_gainpair_trajectory_formal_20261026', 'pid': 1054384, 'child_pid': 1054385, 'gpu': '1', 'command': ['/data2/gb/Dual-RGBT/conda-envs/dual_rgbt/bin/python', '/data2/gb/Dual-RGBT/scripts/train_candidate_joint_gainpair_trajectory.py', '--plan', 'reports/global_candidate_joint_gainpair_trajectory_plan_20261026.json', '--run', 'global_candidate_joint_gainpair_trajectory_formal_20261026', '--sanity-complete-sha256', '048317e6f75d27deaf93c60d77df3fe097e9614efb50b7204f69eed9aa00f2f7'], 'kind': 'train', 'run': 'global_candidate_joint_gainpair_trajectory_formal_20261026', 'wrapper': 'scripts/run_task.py'}, {'task': 'train_joint_quality_k8_cost0_formal_20261026', 'pid': 1170719, 'child_pid': 1170720, 'gpu': '0', 'command': ['/data2/gb/Dual-RGBT/conda-envs/dual_rgbt/bin/python', '/data2/gb/Dual-RGBT/scripts/train_candidate_joint_quality_trajectory.py', '--plan', 'reports/concurrent_joint_quality_k8_cost0_plan_20261026.json', '--run', 'joint_quality_k8_cost0_formal_20261026', '--sanity-complete-sha256', '9f634035bc29cd2e79306029e28efc7a81c066f5bf4bec223da4a54d250df845'], 'kind': 'train', 'run': 'joint_quality_k8_cost0_formal_20261026', 'wrapper': 'scripts/run_task.py'}, {'task': 'train_joint_gainpair_k8_cost0_formal_20261026', 'pid': 1170862, 'child_pid': 1170878, 'gpu': '1', 'command': ['/data2/gb/Dual-RGBT/conda-envs/dual_rgbt/bin/python', '/data2/gb/Dual-RGBT/scripts/train_candidate_joint_gainpair_trajectory.py', '--plan', 'reports/concurrent_joint_gainpair_k8_cost0_plan_20261026.json', '--run', 'joint_gainpair_k8_cost0_formal_20261026', '--sanity-complete-sha256', '9664a5e76e1c7bc7c3b62ecb8b63524706f09af879a39ba0fd3e6c369b03c175'], 'kind': 'train', 'run': 'joint_gainpair_k8_cost0_formal_20261026', 'wrapper': 'scripts/run_task.py'}, {'task': 'train_joint_quality_k4_cost0p05_formal_20261026', 'pid': 1185099, 'child_pid': 1185117, 'gpu': '0', 'command': ['/data2/gb/Dual-RGBT/conda-envs/dual_rgbt/bin/python', '/data2/gb/Dual-RGBT/scripts/train_candidate_joint_quality_trajectory.py', '--plan', 'reports/concurrent_joint_quality_k4_cost0p05_plan_20261026.json', '--run', 'joint_quality_k4_cost0p05_formal_20261026', '--sanity-complete-sha256', 'e037a755376bd8685c65bd812a325b67287da1ec6413ee194e291024855bc4b3'], 'kind': 'train', 'run': 'joint_quality_k4_cost0p05_formal_20261026', 'wrapper': 'scripts/run_task.py'}, {'task': 'train_joint_gainpair_k4_cost0p05_formal_20261026', 'pid': 1185311, 'child_pid': 1185312, 'gpu': '1', 'command': ['/data2/gb/Dual-RGBT/conda-envs/dual_rgbt/bin/python', '/data2/gb/Dual-RGBT/scripts/train_candidate_joint_gainpair_trajectory.py', '--plan', 'reports/concurrent_joint_gainpair_k4_cost0p05_plan_20261026.json', '--run', 'joint_gainpair_k4_cost0p05_formal_20261026', '--sanity-complete-sha256', '71fbe6f05e235aff49aff938e58944404c2fa72d30cc43512e29bdfea111e229'], 'kind': 'train', 'run': 'joint_gainpair_k4_cost0p05_formal_20261026', 'wrapper': 'scripts/run_task.py'}, {'task': 'train_joint_quality_k4_cost0_formal_20261026', 'pid': 1633707, 'child_pid': 1633708, 'gpu': '2', 'command': ['/data2/gb/Dual-RGBT/conda-envs/dual_rgbt/bin/python', '/data2/gb/Dual-RGBT/scripts/train_candidate_joint_quality_trajectory.py', '--plan', 'reports/concurrent_joint_quality_k4_cost0_plan_20261026.json', '--run', 'joint_quality_k4_cost0_formal_20261026', '--sanity-complete-sha256', '26a85194666438ac5e22d5871c9d61ac3c4a0d32f660f2c20331d5e8cbf9e1db'], 'run': 'joint_quality_k4_cost0_formal_20261026', 'kind': 'train', 'wrapper': 'scripts/run_task_four_gpu.py'}, {'task': 'train_joint_gainpair_k4_cost0_formal_20261026', 'pid': 1633907, 'child_pid': 1633908, 'gpu': '3', 'command': ['/data2/gb/Dual-RGBT/conda-envs/dual_rgbt/bin/python', '/data2/gb/Dual-RGBT/scripts/train_candidate_joint_gainpair_trajectory.py', '--plan', 'reports/concurrent_joint_gainpair_k4_cost0_plan_20261026.json', '--run', 'joint_gainpair_k4_cost0_formal_20261026', '--sanity-complete-sha256', '78bd577871e372fc969fdffc07dab8eca92e44c1e8ee0f9bc5634425b7000f17'], 'run': 'joint_gainpair_k4_cost0_formal_20261026', 'kind': 'train', 'wrapper': 'scripts/run_task_four_gpu.py'}]; p=json.loads((r/case['plan']).read_bytes());gpu=p['physical_gpu']
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
command=['/data2/gb/Dual-RGBT/conda-envs/dual_rgbt/bin/python',p['launch_wrapper'],'--task',p['tasks']['sanity'],'--gpu',str(gpu),'--','/data2/gb/Dual-RGBT/conda-envs/dual_rgbt/bin/python',str(r/trainer),'--plan',case['plan'],'--run',p['runs']['sanity'],'--sanity']
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

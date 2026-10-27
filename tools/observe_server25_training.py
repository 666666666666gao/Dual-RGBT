"""One scheduled observation of the additional25 controller and its eight jobs."""
import argparse
import base64
import hashlib
import json
import sys
import time

from server25_runtime import ROOT, REMOTE, remote


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--node', required=True)
    args = parser.parse_args()
    node = json.loads((ROOT / args.node).read_bytes())
    output = ROOT / node['output']
    assert not output.exists()
    assert hashlib.sha256((ROOT / 'tools/observe_server25_training.py').read_bytes()).hexdigest() == node['observer_sha256']
    print(json.dumps(dict(state='WAITING_FOR_SERVER25_NODE', deadline=node['observe_at_epoch'])), flush=True)
    while time.time() < node['observe_at_epoch']:
        time.sleep(min(60, node['observe_at_epoch'] - time.time()))
    code = """import base64,hashlib,json,subprocess,sys,time
from pathlib import Path
r=Path(%r);sys.path.insert(0,str(r/'tools'))
from launch_server25_training import startup
n=json.loads((r/'runs/pipeline_tasks/admit_server25_remaining_grid_20261027/status.json').read_bytes())
assert n['pid']==%r
report=r/'reports/SERVER25_REMAINING_GRID_EXECUTION_20261027.json'
raw=report.read_bytes();execution=json.loads(raw);tasks=[]
before=dict(observed=time.time(),ticks=[int(v) for v in Path('/proc/stat').read_text().splitlines()[0].split()[1:9]])
if execution['state']=='EIGHT_FORMALS_STARTUP_PASS':
 assert n['state']=='COMPLETE' and n['exit_code']==0
 tasks=[startup(c) for c in execution['cases']]
time.sleep(10)
after=dict(observed=time.time(),ticks=[int(v) for v in Path('/proc/stat').read_text().splitlines()[0].split()[1:9]])
delta=[b-a for a,b in zip(before['ticks'],after['ticks'])];assert sum(delta)>0
mem={v.split(':')[0]:int(v.split()[1]) for v in Path('/proc/meminfo').read_text().splitlines()}
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],universal_newlines=True).splitlines()
out=dict(observed=time.time(),controller=n,execution_state=execution['state'],execution_sha256=hashlib.sha256(raw).hexdigest(),execution_base64=base64.b64encode(raw).decode(),tasks=tasks,meminfo_kib=mem,gpu_csv=gpu,cpu_sample=dict(before=before,after=after,busy_percent=100*(sum(delta)-delta[3]-delta[4])/sum(delta)))
if n['state']!='COMPLETE':out['controller_log_tail']=(r/'runs/pipeline_tasks/admit_server25_remaining_grid_20261027/run.log').read_text()[-7000:]
print(json.dumps(out))
""" % (REMOTE, node['controller_pid'])
    result = remote(code)
    raw = base64.b64decode(result.pop('execution_base64'))
    assert hashlib.sha256(raw).hexdigest() == result['execution_sha256']
    canonical = ROOT / 'reports/SERVER25_REMAINING_GRID_EXECUTION_20261027.json'
    if result['execution_state'] == 'EIGHT_FORMALS_STARTUP_PASS':
        if canonical.exists():
            assert canonical.read_bytes() == raw
        else:
            canonical.write_bytes(raw)
    result.update(node_plan=args.node, node_plan_sha256=hashlib.sha256((ROOT / args.node).read_bytes()).hexdigest())
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(output=node['output'], controller=result['controller']['state'],
                          execution_state=result['execution_state'], tasks=len(result['tasks']),
                          gpu=result['gpu_csv'], mem_available_kib=result['meminfo_kib']['MemAvailable'])), flush=True)


if __name__ == '__main__':
    main()

"""One scheduled read-only observation of the formal training and final dev."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from complete_pipeline import REMOTE, remote, wait_until


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--node', required=True)
    args = parser.parse_args()
    node = json.loads((ROOT / args.node).read_bytes())
    output = ROOT / node['output']
    assert not output.exists()
    print(json.dumps(dict(state='WAITING_FOR_REGISTERED_NODE', pid=os.getpid(),
                          deadline=node['observe_at_epoch'])), flush=True)
    wait_until(node['observe_at_epoch'])
    code = """import csv,json,math,time
from pathlib import Path
r=Path(%r); entries=%r; tasks=[]
for e in entries:
 n=json.loads((r/'runs/pipeline_tasks'/e['task']/'status.json').read_bytes())
 assert n['pid']==e['pid'] and n['physical_gpu']==e['gpu'] and n['command']==e['command']
 item=dict(native=n)
 if n['state']=='RUNNING':
  p=Path('/proc')/str(n['pid'])
  assert (p/'cmdline').read_bytes().decode().split(chr(0))[:-1]==[n['command'][0],'scripts/run_task.py','--task',e['task'],'--gpu',e['gpu'],'--']+n['command']
  assert [int(v) for v in (p/'task'/str(n['pid'])/'children').read_text().split()]==[e['child_pid']]
  child=Path('/proc')/str(e['child_pid'])
  assert (child/'cmdline').read_bytes().decode().split(chr(0))[:-1]==n['command']
  assert int(next(v.split()[1] for v in (child/'status').read_text().splitlines() if v.startswith('PPid:')))==n['pid']
  item['child_pid']=e['child_pid']
 directory=r/'runs'/e['run']
 if e['kind']=='train':
  with (directory/'trajectory_steps.csv').open(newline='') as handle:
   rows=[v for v in csv.DictReader(handle) if None not in v.values()]
  assert rows and int(rows[-1]['step'])==len(rows)
  assert all(all(math.isfinite(float(v[k])) for k in ('loss','quality_loss','gain_loss','quality_regression_loss','quality_rank_loss')) for v in rows)
  item.update(logged_updates=len(rows),last_update=rows[-1],finite_logged_losses=True)
 elif n['state']=='RUNNING':
  progress=json.loads((directory/'progress.json').read_bytes())
  item.update(completed_sequences=len(progress),completed_frames=sum(v['frames'] for v in progress))
 if n['state']=='COMPLETE':
  assert n['exit_code']==0
  item['complete']=json.loads((directory/'COMPLETE.json').read_bytes())
 elif n['state']!='RUNNING':
  item['failure_log_tail']=(r/'runs/pipeline_tasks'/e['task']/'run.log').read_text()[-6000:]
 tasks.append(item)
print(json.dumps(dict(observed=time.time(),tasks=tasks)))
""" % (REMOTE, node['tasks'])
    observed = remote(code)
    output.write_text(json.dumps(observed, indent=2) + '\n')
    print(json.dumps(observed), flush=True)


if __name__ == '__main__':
    main()

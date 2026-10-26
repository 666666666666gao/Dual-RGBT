"""One read-only observation at the registered ETA; never launch or stop a job."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from complete_pipeline import REMOTE, remote


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--node', required=True)
    args = parser.parse_args()
    node = json.loads((ROOT / args.node).read_bytes())
    output = ROOT / node['output']
    assert not output.exists()
    print(json.dumps(dict(pid=os.getpid(), state='WAITING_FOR_REGISTERED_NODE',
                          deadline=node['observe_at_epoch'], node=args.node)), flush=True)
    while time.time() < node['observe_at_epoch']:
        time.sleep(min(60, node['observe_at_epoch'] - time.time()))
    code = """import csv,json,time
from pathlib import Path
r=Path(%r); tasks=%r
records=[]
for expected in tasks:
 n=json.loads((r/'runs/pipeline_tasks'/expected['task']/'status.json').read_bytes())
 assert n['task']==expected['task'] and n['pid']==expected['pid']
 assert n['physical_gpu']==str(expected['gpu']) and n['command']==expected['command']
 wrapper=Path('/proc')/str(n['pid'])
 record=dict(native=n,wrapper_exists=wrapper.exists())
 if n['state']=='RUNNING':
  argv=(wrapper/'cmdline').read_bytes().decode().split(chr(0))[:-1]
  assert argv==[n['command'][0],'scripts/run_task.py','--task',n['task'],'--gpu',n['physical_gpu'],'--']+n['command']
  children=[int(v) for v in (wrapper/'task'/str(n['pid'])/'children').read_text().split()]
  assert children==[expected['child_pid']]
  child=Path('/proc')/str(children[0])
  assert (child/'cmdline').read_bytes().decode().split(chr(0))[:-1]==n['command']
  assert int(next(x.split()[1] for x in (child/'status').read_text().splitlines() if x.startswith('PPid:')))==n['pid']
  record.update(wrapper_argv=argv,child_pid=children[0])
 else:
  assert n['state']=='COMPLETE' and n['exit_code']==0,n
 directory=r/'runs'/n['task']
 progress=json.loads((directory/'progress.json').read_bytes())
 frames=0; seconds=[]
 for path in sorted((directory/'traces').glob('*.csv'),key=lambda q:q.stat().st_mtime):
  if path.stat().st_size:
   with path.open(newline='') as f: header,*rows=csv.reader(f)
   rows=[row for row in rows if len(row)==len(header)]
   frames+=max([int(row[0]) for row in rows]+[0])+1
   seconds.extend(float(row[header.index('seconds_including_decode')]) for row in rows)
 tail=seconds[-256:]; assert tail and min(tail)>0
 record.update(completed_sequences=len(progress),completed_frames=sum(row['frames'] for row in progress),
               trace_frames_including_init=frames,recent_service_fps=len(tail)/sum(tail))
 if n['state']=='COMPLETE': record['complete']=json.loads((directory/'COMPLETE.json').read_bytes())
 records.append(record)
print(json.dumps(dict(observed=time.time(),tasks=records)))
""" % (REMOTE, node['tasks'])
    observed = remote(code)
    previous = json.loads((ROOT / node['previous_observation']).read_bytes())
    elapsed = observed['observed'] - previous['observed']
    for record, before in zip(observed['tasks'], previous['tasks']):
        assert record['native']['task'] == before['native']['task']
        rate = (record['trace_frames_including_init'] - before['trace_frames_including_init']) / elapsed
        record['interval_wall_fps'] = rate
        if record['native']['state'] == 'RUNNING':
            record['estimated_remaining_seconds'] = (54646 - record['trace_frames_including_init']) / rate
    observed.update(node=args.node, window_seconds=elapsed,
                    scope='Buffered trace counters; varying frames and load, not a paired speed benchmark.')
    output.write_text(json.dumps(observed, indent=2) + '\n')
    print(json.dumps(observed), flush=True)


if __name__ == '__main__':
    main()

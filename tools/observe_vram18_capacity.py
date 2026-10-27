"""One scheduled full-wave capacity node; preserve raw endpoint resources."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from launch_vram18_serial_training_wave import measurement
from complete_pipeline import remote,wait_until

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--node',required=True);args=ap.parse_args()
    node=json.loads((ROOT/args.node).read_bytes());output=ROOT/node['output'];assert not output.exists()
    assert hashlib.sha256((ROOT/'tools/launch_vram18_serial_training_wave.py').read_bytes()).hexdigest()=='c1ef685e700b57d3a7754c4730edb889702abc235ab8b1b20e563eab217fb4b1'
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==node['observer_sha256']
    print(json.dumps(dict(state='WAITING_FOR_REGISTERED_CAPACITY_NODE',observe_at_epoch=node['observe_at_epoch'])),flush=True)
    wait_until(node['observe_at_epoch'])
    cpu_code="import json,time\nfrom pathlib import Path\nprint(json.dumps(dict(observed=time.time(),cpu_ticks=[int(v) for v in Path('/proc/stat').read_text().splitlines()[0].split()[1:]])))"
    before=remote(cpu_code);time.sleep(10)
    snapshot=measurement(node['tasks']);after=remote(cpu_code)
    assert len(snapshot['tasks'])==len(node['tasks'])
    delta=[b-a for a,b in zip(before['cpu_ticks'][:8],after['cpu_ticks'][:8])]
    assert sum(delta)>0 and all(v>=0 for v in delta)
    snapshot['cpu_sample']=dict(before=before,after=after,interval_seconds=after['observed']-before['observed'],busy_percent=100*(sum(delta)-delta[3]-delta[4])/sum(delta),scope='Aggregate logical-CPU ticks, idle and iowait excluded; includes endpoint snapshot overhead.')
    snapshot['node_plan']=args.node;snapshot['node_plan_sha256']=hashlib.sha256((ROOT/args.node).read_bytes()).hexdigest()
    output.write_text(json.dumps(snapshot,indent=2)+'\n')
    print(json.dumps(dict(state='CAPACITY_NODE_COMPLETE',output=node['output'],observed=snapshot['observed'],tasks=len(snapshot['tasks']),gpu_csv=snapshot['gpu_csv'],mem_available_kib=snapshot['meminfo_kib']['MemAvailable'],cpu_busy_percent=snapshot['cpu_sample']['busy_percent'])),flush=True)

if __name__=='__main__':main()

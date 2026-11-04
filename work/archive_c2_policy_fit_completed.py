"""Read-only terminal C2 FIT audit and one-time raw metadata archive."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REMOTE = r'''
import base64
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import sys

request = json.load(sys.stdin)
root, task = Path(request['root']), request['task']
sha = lambda raw: hashlib.sha256(raw).hexdigest()
plan_path = root / 'reports' / task / 'plan.json'
plan = json.loads(plan_path.read_bytes())
run = Path(plan['run'])
paths = dict(plan=plan_path, config=run / 'config.json',
    native=root / 'runs/pipeline_tasks' / task / 'status.json',
    complete=run / 'COMPLETE.json', precision=run / 'precision.json',
    review=Path(plan['review']['path']),
    acceptance=Path(plan['collection_acceptance']['path']),
    replay_native=Path(plan['collection_acceptance_native']['path']))
raw = {name: path.read_bytes() for name, path in paths.items()}
docs = {name: json.loads(value) for name, value in raw.items()}
native, complete = docs['native'], docs['complete']
assert plan == docs['config']
assert plan['stage'] == complete['stage'] == 'P2_C2_policy_refresh_fit'
assert plan['launch_allowed'] is True and plan['smoke'] is False
assert plan['host'] == request['host']
assert plan['task'] == native['task'] == task and plan['root'] == str(root)
assert native['state'] == complete['status'] == 'COMPLETE'
assert native['exit_code'] == 0 and native['command'] == plan['command']
assert native['physical_gpu'] == plan['gpu']
assert not Path('/proc', str(native['pid'])).exists()
assert (complete['epochs'], complete['steps'], complete['records']) == (5,890,91091)
assert complete['retained_epochs'] == [1,2,3,4,5]
assert complete['plan_sha256'] == sha(raw['plan'])
assert complete['executor_preserved'] is complete['all_policy_modules_changed'] is True
assert complete['strict_reload'] is True and complete['benchmark_evidence'] is False
assert complete['test_enabled'] is complete['dev_enabled'] is False
assert docs['precision'] == dict(matmul_allow_tf32=False,
    cudnn_allow_tf32=False,cudnn_benchmark=False)
for name,key in [('review','review'),('acceptance','collection_acceptance'),
                 ('replay_native','collection_acceptance_native')]:
    assert plan[key] == dict(path=str(paths[name]),sha256=sha(raw[name]))
assert docs['replay_native']['state'] == 'COMPLETE'
assert docs['replay_native']['exit_code'] == 0
assert docs['acceptance']['status'] == 'PASS'
assert docs['review']['status'] == 'SOURCE_PASS'
assert docs['review']['blocking_issues'] == 0
assert docs['review']['sources'] == plan['sources']
binding = sha(json.dumps({key:value for key,value in plan.items()
    if key not in ('launch_allowed','review')},sort_keys=True,
    separators=(',',':')).encode())
assert binding in docs['review']['admitted_plan_bindings']
for name,digest in plan['sources'].items():
    assert sha((root / name).read_bytes()) == digest,name
assert set(complete['artifacts']) == {'epoch_%03d.pth'%n for n in range(1,6)}
weights = {}
for name,digest in complete['artifacts'].items():
    path = run / name
    assert sha(path.read_bytes()) == digest,name
    weights[name] = dict(sha256=digest,bytes=path.stat().st_size)
raw['steps'] = (run / 'steps.csv').read_bytes()
rows = list(csv.DictReader(io.StringIO(raw['steps'].decode())))
assert len(rows) == 890
assert [int(row['step']) for row in rows] == list(range(1,891))
assert [int(row['epoch']) for row in rows] == [ep for ep in range(1,6) for _ in range(178)]
assert all(math.isfinite(float(value)) for row in rows for value in row.values())
raw['log'] = (root / 'runs/pipeline_tasks' / task / 'run.log').read_bytes()
print(json.dumps(dict(status='C2_POLICY_FIT_TERMINAL_METADATA_PASS',
    host=plan['host'],variant=plan['variant'],task=task,completed=native['completed'],
    fit_plan_binding=binding,weights=weights,steps=890,epochs=5,
    source_count=len(plan['sources']),benchmark_evidence=False,
    cpu_reconstruction_performed=False,
    evidence={name:dict(path=str(path),sha256=sha(raw[name])) for name,path in paths.items()},
    files={name:dict(raw=base64.b64encode(value).decode(),sha256=sha(value))
           for name,value in raw.items()})))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('host', 'python', 'root', 'task', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    output = Path(args.output)
    assert not output.exists()
    result = subprocess.run([
        'ssh', '-o', 'ClearAllForwardings=yes', '-o', 'BatchMode=yes',
        args.host, ' '.join(shlex.quote(word) for word in (args.python, '-B', '-c', REMOTE))],
        input=json.dumps(vars(args)).encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    assert result.returncode == 0, result.stderr.decode(errors='replace')
    response = json.loads(result.stdout)
    files = {}
    for name,item in response.pop('files').items():
        raw = base64.b64decode(item['raw'],validate=True)
        assert hashlib.sha256(raw).hexdigest() == item['sha256'],name
        files[name + ('.csv' if name == 'steps' else '.log' if name == 'log' else '.json')] = raw
    output.mkdir(parents=True,exist_ok=False)
    for name,raw in files.items():
        with (output / name).open('xb') as handle:
            handle.write(raw)
        assert (output / name).read_bytes() == raw
    hashes = {name:hashlib.sha256(raw).hexdigest() for name,raw in files.items()}
    (output / 'SHA256SUMS').write_text(''.join(
        digest+'  '+name+'\n' for name,digest in sorted(hashes.items())))
    (output / 'audit.json').write_text(json.dumps(response,indent=2)+'\n')
    print(json.dumps(response,indent=2))


if __name__ == '__main__':
    main()

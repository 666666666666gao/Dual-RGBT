"""Deploy reviewed C2 sources, reconstruct on CPU, then smoke before LasHeR."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from work.prepare_c2_policy_evaluation import NEW_SOURCES, binding

ENTRY = 'work/operate_c2_policy_evaluation.py'
REMOTE = r'''
import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

request = json.load(sys.stdin)
sha = lambda raw: hashlib.sha256(raw).hexdigest()
read = lambda path: json.loads(Path(path).read_bytes())
plans = request['plans']
plan = plans['lasher']
root, original = Path(plan['root']), Path(request['source_root'])
review_raw = base64.b64decode(request['review_raw'], validate=True)
assert sha(review_raw) == request['review_sha256']
review = json.loads(review_raw)
assert review['stage'] == 'P2_C2_policy_evaluation'
assert review['status'] == 'SOURCE_PASS' and review['blocking_issues'] == 0
assert review['sources'] == plan['sources']
assert review['operational_sources'] == request['operational_sources']
assert socket.gethostname() == {'2025':'ubun',
    '2026':'ubuntu-WS-C621E-SAGE-Series'}[plan['host']]
assert set(plans) == {'smoke','lasher'}
for name,p in plans.items():
    canonical = sha(json.dumps({k:v for k,v in p.items()
        if k not in ('launch_allowed','review')},sort_keys=True,
        separators=(',',':')).encode())
    assert canonical in review['admitted_plan_bindings']
    assert p['sources'] == plan['sources'] and p['root'] == str(root)
    assert p['host'] == plan['host'] and p['gpu'] == plan['gpu']
    assert p['checkpoint'] == plan['checkpoint']
    assert p['fit_evidence'] == plan['fit_evidence']
    assert p['launch_allowed'] is True
    assert p['review']['sha256'] == request['review_sha256']
    assert p['command'] == [request['python'],'-B',
        str(root / 'scripts/eval_c2_policy_completed.py'),'--plan',p['plan_path']]
    assert Path(p['plan_path']) == root / 'reports' / p['task'] / 'plan.json'
    assert p['smoke'] is (name == 'smoke')
    for item in p['fit_evidence'].values():
        assert sha(Path(item['path']).read_bytes()) == item['sha256']
    for item in (p['manifest'],p['protocol'],p['checkpoint']):
        assert sha(Path(item['path']).read_bytes()) == item['sha256']
    for name,digest in read(p['protocol']['path'])['toolkit_code_sha256'].items():
        assert sha((Path(p['toolkit']) / name.replace('\\','/')).read_bytes()) == digest
fit = read(plan['fit_evidence']['plan']['path'])
assert str(original) == fit['root'] and root != original
assert root.parent == original.parent and root.is_absolute()
new = {name:base64.b64decode(raw,validate=True)
       for name,raw in request['new_sources'].items()}
assert set(plan['sources']) == set(fit['sources']) | set(new)
assert not set(new).intersection(fit['sources'])
for name,digest in fit['sources'].items():
    assert plan['sources'][name] == digest
    assert sha((original / name).read_bytes()) == digest,name
for name,raw in new.items():
    assert sha(raw) == plan['sources'][name],name
action = request['action']
if action == 'deploy':
    assert not root.exists()
    root.mkdir(exist_ok=False)
    for name,digest in plan['sources'].items():
        raw = new[name] if name in new else (original / name).read_bytes()
        target = root / name
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(raw)
        assert sha(target.read_bytes()) == digest,name
    review_path = Path(plan['review']['path'])
    review_path.relative_to(root)
    review_path.parent.mkdir(parents=True,exist_ok=True)
    review_path.write_bytes(review_raw)
    for p in plans.values():
        path = Path(p['plan_path'])
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(p,indent=2)+'\n')
else:
    assert sha(Path(plan['review']['path']).read_bytes()) == request['review_sha256']
    for p in plans.values():
        assert read(p['plan_path']) == p
for name,digest in plan['sources'].items():
    assert sha((root / name).read_bytes()) == digest,name
os.chdir(root)
sys.path.insert(0,str(root))
sys.path.insert(0,str(root / 'scripts'))
from eval_c2_policy_completed import validate_plan
for p in plans.values():
    validate_plan(p,Path(p['plan_path']),root)
cpu_path = root / 'reports/cpu_reconstruction.json'
result = dict(action=action,host=plan['host'],root=str(root))
if action == 'deploy':
    import torch
    torch.set_num_threads(1)
    from dual_rgbt.c2_policy_evaluation import load_completed_policy
    model = load_completed_policy(plan['fit_evidence'],plan['checkpoint']['path'],
        plan['checkpoint']['sha256'],device='cpu')
    assert not torch.cuda.is_initialized()
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    witness = dict(status='C2_POLICY_CPU_RECONSTRUCTION_PASS',
        checkpoint=plan['checkpoint'],fit_evidence=plan['fit_evidence'],
        sources=plan['sources'],source_review_sha256=request['review_sha256'],
        cuda_initialized=False,parameters=sum(p.numel() for p in model.parameters()),
        benchmark_evidence=False)
    with cpu_path.open('x') as handle:
        json.dump(witness,handle,indent=2)
    result.update(status=witness['status'],witness=witness)
elif action in ('smoke','full'):
    cpu = read(cpu_path)
    assert cpu['status'] == 'C2_POLICY_CPU_RECONSTRUCTION_PASS'
    for key in ('checkpoint','fit_evidence','sources'):
        assert cpu[key] == plan[key],key
    assert cpu['source_review_sha256'] == request['review_sha256']
    assert cpu['cuda_initialized'] is False
    if action == 'full':
        smoke = plans['smoke']
        run = Path(smoke['run'])
        native = read(root / 'runs/pipeline_tasks' / smoke['task'] / 'status.json')
        complete = read(run / 'SMOKE.json')
        config = read(run / 'config.json')
        assert native['state'] == 'COMPLETE' and native['exit_code'] == 0
        assert not Path('/proc',str(native['pid'])).exists()
        assert native['task'] == complete['task'] == smoke['task']
        assert native['physical_gpu'] == smoke['gpu']
        assert native['command'] == config['native_command'] == smoke['command']
        assert config['plan'] == smoke and config['source_sha256'] == smoke['sources']
        assert complete['plan_sha256'] == sha(Path(smoke['plan_path']).read_bytes())
        assert complete['config_sha256'] == sha((run / 'config.json').read_bytes())
        assert complete['checkpoint_sha256'] == smoke['checkpoint']['sha256']
        assert complete['sequences'] == 1 and complete['frames'] == 11
        assert complete['smoke_only'] is complete['state_unchanged'] is True
        assert complete['optimizer_updates'] == complete['backward_calls'] == 0
        assert read(run / 'precision.json') == dict(matmul_allow_tf32=False,
            cudnn_allow_tf32=False,cudnn_benchmark=False)
    job = plans['smoke' if action == 'smoke' else 'lasher']
    native_path = root / 'runs/pipeline_tasks' / job['task'] / 'status.json'
    assert not native_path.parent.exists() and not Path(job['run']).exists()
    rows = subprocess.check_output(['nvidia-smi',
        '--query-gpu=index,memory.free,memory.used,temperature.gpu,power.limit,uuid',
        '--format=csv,noheader,nounits'],text=True).splitlines()
    row = next([word.strip() for word in line.split(',')] for line in rows
               if line.split(',')[0].strip() == job['gpu'])
    assert int(row[1]) >= 6000 and int(row[2]) < 500 and int(row[3]) < 85
    assert float(row[4]) == (250 if plan['host'] == '2025' else 350)
    processes = subprocess.check_output(['nvidia-smi',
        '--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],
        text=True).splitlines()
    assert not any(line.split(',')[0].strip() == row[5] for line in processes)
    assert shutil.disk_usage(root).free >= 5 * 2**30
    admission = dict(status='RUNTIME_ADMISSION_PASS',time=time.time(),
        task=job['task'],gpu_row=row,plan_sha256=sha(Path(job['plan_path']).read_bytes()),
        source_review_sha256=request['review_sha256'])
    with (Path(job['plan_path']).parent / 'runtime_admission.json').open('x') as handle:
        json.dump(admission,handle,indent=2)
    argv = [request['python'],'-B',str(root / 'scripts/run_task_four_gpu.py'),
            '--task',job['task'],'--gpu',job['gpu'],'--'] + job['command']
    child = subprocess.Popen(argv,cwd=str(root),stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True,
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
    result.update(status='LAUNCHED',task=job['task'],wrapper_pid=child.pid,
                  admission=admission)
elif action == 'inspect':
    result.update(status='OBSERVED',tasks=[])
    for p in plans.values():
        path = root / 'runs/pipeline_tasks' / p['task'] / 'status.json'
        item = dict(task=p['task'],native_exists=path.exists())
        if path.exists():
            item['native'] = read(path)
            item['wrapper_live'] = Path('/proc',str(item['native']['pid'])).exists()
            for name in ('progress.json','SMOKE.json','COMPLETE.json'):
                if (Path(p['run']) / name).exists():
                    item[name] = read(Path(p['run']) / name)
        result['tasks'].append(item)
print(json.dumps(result))
'''


def prepare_request(folder, review_path, action):
    plans = {name:json.loads((folder / (name+'.json')).read_bytes())
             for name in ('smoke','lasher')}
    raw = review_path.read_bytes()
    review = json.loads(raw)
    digest = hashlib.sha256(raw).hexdigest()
    operations = {ENTRY:hashlib.sha256((ROOT / ENTRY).read_bytes()).hexdigest()}
    assert review['status'] == 'SOURCE_PASS' and review['blocking_issues'] == 0
    assert review['stage'] == 'P2_C2_policy_evaluation'
    assert review['operational_sources'] == operations
    for plan in plans.values():
        assert binding(plan) in review['admitted_plan_bindings']
        assert review['sources'] == plan['sources']
        assert plan['launch_allowed'] is False and plan['review']['sha256'] == 'PENDING'
        plan['launch_allowed'] = True
        plan['review']['sha256'] = digest
    fit = json.loads((ROOT / 'reports/c2_policy_fit_r1_20261104' /
        ('dropmoe' if plans['lasher']['host']=='2025' else 'sot') /
        plans['lasher']['variant'] / 'completion_witness/plan.json').read_bytes())
    return dict(action=action,plans=plans,source_root=fit['root'],
        python=plans['lasher']['command'][0],review_sha256=digest,
        review_raw=base64.b64encode(raw).decode(),operational_sources=operations,
        new_sources={name:base64.b64encode((ROOT / name).read_bytes()).decode()
                     for name in NEW_SOURCES})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('deploy','smoke','full','inspect'))
    parser.add_argument('--plans',type=Path,required=True)
    parser.add_argument('--review',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    if args.output:
        assert not args.output.exists()
    request = prepare_request(args.plans,args.review,args.action)
    result = subprocess.run(['ssh','-o','ClearAllForwardings=yes','-o','BatchMode=yes',
        request['plans']['lasher']['host'],
        ' '.join(shlex.quote(word) for word in (request['python'],'-B','-c',REMOTE))],
        input=json.dumps(request).encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('xb') as handle:
            handle.write(result.stdout if result.returncode == 0 else result.stderr)
    assert result.returncode == 0,result.stderr.decode(errors='replace')[-6000:]
    print(result.stdout.decode())


if __name__ == '__main__':
    main()

"""Read-only CPU audit of archived results, TRAIN labels, and dataset annotations."""
import csv
import hashlib
import itertools
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

root=Path('/data2/gb/Dual-RGBT')
hashes={}
def sha(p):
    p=Path(p);h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    hashes[str(p.relative_to(root))]=h.hexdigest()
    return h.hexdigest()
def read(p):
    p=root/p;sha(p);return json.loads(p.read_bytes())

plan=read('reports/rankstop_dev_plan_server26_20261026.json')
for p,h in plan['source_sha256'].items():assert sha(root/p)==h,(p,h)
for p,h in plan['input_sha256'].items():assert sha(root/p)==h,(p,h)
assert sha(root/'checkpoints/CADTrack_ep0014.pth.tar')==plan['protected_checkpoint_sha256']
manifest=read(plan['manifests']['dev'])
full=read('data/manifest.json'); original=read('inputs/server25/manifest.json')
assert len(full['train'])==len(original['train'])==882 and len(full['dev'])==len(original['dev'])==97
assert not ({s['name'] for s in full['train']}&{s['name'] for s in full['dev']})
assert [s['name'] for s in full['dev']]==[s['name'] for s in manifest['sequences']]
gt_frames=0
for s,a,b in zip(manifest['sequences'],full['dev'],original['dev']):
    assert s['name']==a['name']==b['name'] and s['boxes']==a['boxes']==b['boxes']
    p=root/'data/LasHeR/train'/s['name']/'init.txt'
    gt=np.loadtxt(str(p),delimiter=',',dtype=np.float32,ndmin=2)
    assert np.array_equal(gt,np.asarray(s['boxes'],dtype=np.float32))
    assert len(s['rgb'])==len(s['tir'])==len(gt)
    assert np.array_equal(gt[0],np.asarray(s['initial_box'],dtype=np.float32))
    sha(p);gt_frames+=len(gt)
assert gt_frames==54646
for cost in [0.,.05,.15]:
    entry=plan['runs'][str(cost)]
    run=root/'runs'/entry['run']
    for p in run.rglob('*'):
        if p.is_file():sha(p)
    native=read('runs/pipeline_tasks/'+entry['task']+'/status.json')
    assert native['state']=='COMPLETE' and native['exit_code']==0
train=root/'runs/global_candidate_rankstop_trajectory_formal_20261018'
config=read(train/'config.json');complete=read(train/'COMPLETE.json')
assert complete['checkpoint_sha256']==plan['checkpoint_sha256']
assert complete['private_labels_sha256']==sha(train/'query_labels.csv')=='6ddfdb92feae3294ea85211a672fe6bf1b07cdcdd5a8abc1dfdf38b908bbfa47'
assert sha(train/'trajectory_steps.csv')=='2f22103097fdcaeb4483841cf5456160938a2ba0baadf393a7d70317f0a14e50'
training_plan=read('reports/global_candidate_rankstop_trajectory_plan_20261018.json')
assert config['plan_sha256']==plan['training_plan_sha256']==sha(root/'reports/global_candidate_rankstop_trajectory_plan_20261018.json')
assert config['source_sha256']==training_plan['source_sha256']
assert config['input_sha256']==training_plan['input_sha256']

# Recompute diagnostics without importing the author's analysis modules.
groups=defaultdict(lambda:dict(labels=0,exact_gain_violates_fixed_margin=0,idealized_wrong_call_cost_0_0=0,idealized_wrong_call_cost_0_05=0,idealized_wrong_call_cost_0_15=0))
pairs=defaultdict(lambda:defaultdict(int)); count=labels=0
expected_steps=((rnd,s['name'],frame) for rnd in range(1,3) for s in [next(x for x in full['train'] if x['name']==name) for name in training_plan['sequence_names']] for frame in s['valid_frames'] if frame>0 and frame%training_plan['label_stride']==0)
with (train/'query_labels.csv').open(newline='') as f,(train/'trajectory_steps.csv').open(newline='') as g:
    bystep=itertools.groupby(csv.DictReader(f),lambda r:int(r['step']))
    for decision in csv.DictReader(g):
        idx,raw=next(bystep); rows=list(raw);count+=1
        assert idx==int(decision['step'])==count
        assert next(expected_steps)==(int(decision['round']),decision['sequence'],int(decision['frame']))
        assert all(np.isfinite(float(decision[k])) for k in ['loss','quality_loss','gain_loss'])
        sample={}
        extra=[]
        for r in rows:
            assert all(r[k]==decision[k] for k in ['round','sequence','frame'])
            kind=int(r['kind']);purpose=r['purpose'];gain=np.float32(r['delta_from_stop'])
            assert np.isfinite(gain) and kind in range(4)
            assert gain==np.float32(np.float32(r['query_iou'])-np.float32(r['main_iou']))
            assert purpose in ['sampled','selected_extra']
            if purpose=='sampled':
                role=int(r['modality']);assert role not in sample;sample[role]=r
                assert int(r['kind'])==int(decision['sampled_kind'])==(count-1)%4
            else:extra.append(r)
            item=groups[(purpose,kind)];item['labels']+=1;labels+=1
            active=abs(gain)>np.float32(.01)
            item['exact_gain_violates_fixed_margin']+=int(active and abs(gain)<np.float32(.05))
            ideal=np.float32(np.sign(gain)*max(abs(gain),np.float32(.05))) if active else gain
            token=np.float32(1 if kind==3 else 193/385)
            for w,key in [(0.,'0_0'),(.05,'0_05'),(.15,'0_15')]:
                truth=np.float32(gain-np.float32(w)*token);pred=np.float32(ideal-np.float32(w)*token)
                item['idealized_wrong_call_cost_'+key]+=int(truth<=0 and pred>0)
        assert set(sample)=={0,1}
        assert bool(extra)==(decision['extra_selected_query']=='True')
        if extra:
            assert len(extra)==1
            selected=extra[0];other=sample[int(decision['chosen_modality'])]
            assert selected['modality']==other['modality']==decision['chosen_modality']
            assert int(selected['kind'])+1==int(decision['chosen_action'])
            assert selected['candidate_rank']==decision['chosen_rank']
            for k in ['main_geometry','main_box','main_raw_box','main_iou']:assert selected[k]==other[k]
            sk=int(selected['kind']);ak=int(other['kind']);same=sk==ak
            assert not same or selected['candidate_rank']!=other['candidate_rank']
            sg=np.float32(selected['delta_from_stop']);ag=np.float32(other['delta_from_stop'])
            su=np.float32(sg-np.float32(.05)*np.float32(1 if sk==3 else 193/385))
            au=np.float32(ag-np.float32(.05)*np.float32(1 if ak==3 else 193/385))
            for k in ['all','same_kind_other_rank' if same else 'different_kind']:
                x=pairs[k];x['pairs']+=1;x['sampled_raw_gain_better_over_0p01']+=int(np.float32(ag-sg)>np.float32(.01))
                x['sampled_net_utility_better_over_0p01']+=int(np.float32(au-su)>np.float32(.01))
                x['selected_negative_utility_sampled_positive']+=int(su<0<au)
                x['both_utilities_negative']+=int(su<0 and au<0)
                x['selected_kind_'+str(sk)]+=1
    assert next(bystep,None) is None and next(expected_steps,None) is None
assert count==182182 and labels==399401 and pairs['all']['pairs']==35037
rows=[]
for purpose in ['sampled','selected_extra']:
    for kind in range(4):
        item=groups[(purpose,kind)]
        row=dict(purpose=purpose,kind=kind,labels=item['labels'],exact_gain_violates_fixed_margin=item['exact_gain_violates_fixed_margin'])
        for w,key in [('0.0','0_0'),('0.05','0_05'),('0.15','0_15')]:row['idealized_wrong_call_cost_'+w]=item['idealized_wrong_call_cost_'+key]
        rows.append(row)

# Tensor inspection only: no model construction, forward, backward, or CUDA API.
import torch
torch.set_num_threads(1)
checkpoint=torch.load(str(train/'latest.pth'),map_location='cpu')
basepath=root/'runs/global_candidate_policy_formal_20261014/latest.pth'
base=torch.load(str(basepath),map_location='cpu')
assert checkpoint['config']==config and checkpoint['step']==182182 and checkpoint['stage']=='candidate_c'
assert not config['sanity']
net=checkpoint['net'];bn=base['net']
assert set(net)==set(bn) and len(net)==265
assert all(v.device.type=='cpu' and torch.isfinite(v).all().item() for v in net.values())
changed=[k for k in net if not torch.equal(net[k],bn[k])]
assert len(changed)==8 and all(k.startswith(('executor.quality.','query_gain.')) for k in changed)
assert len(net)-len(changed)==257
steps=[int(v['step'].item()) for v in checkpoint['optimizer']['state'].values()]
assert len(steps)==8 and set(steps)=={182182}
buffers={k:int(v.numel()) for k,v in net.items() if k.endswith(('running_mean','running_var','num_batches_tracked','token_cost'))}
params=sum(int(v.numel()) for k,v in net.items() if k not in buffers)
assert params==103218887,(params,buffers)
result=dict(status='PASS',source_count=len(plan['source_sha256']),input_count=len(plan['input_sha256']),gt_sequences=97,gt_frames=gt_frames,train_sequences=882,train_dev_disjoint=True,images_fully_rehashed=False,labels=labels,updates=count,pairs=dict(pairs),label_groups=rows,checkpoint_sha256=plan['checkpoint_sha256'],checkpoint_config=config,complete=complete,training_source_current={k:sha(root/k) for k in config['source_sha256']},checkpoint_tensors=dict(count=len(net),frozen_equal=257,changed=changed,optimizer_steps=steps,total_parameters=params,buffers=buffers),neural_execution=False,cuda_calls=False,hashes=hashes)
print(json.dumps(result,indent=2))

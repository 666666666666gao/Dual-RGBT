import csv
import hashlib
import json
from pathlib import Path
import numpy as np
root=Path('/data2/gb/Dual-RGBT')
manifest=json.loads((root/'data/manifest.json').read_bytes())
targets={};hashes={}
for s in manifest['train']:
    p=root/'data/LasHeR/train'/s['name']/'init.txt'
    a=np.loadtxt(str(p),delimiter=',',dtype=np.float32,ndmin=2)
    assert np.array_equal(a,np.asarray(s['boxes'],dtype=np.float32))
    targets[s['name']]=a
    hashes[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
p=root/'runs/global_candidate_rankstop_trajectory_formal_20261018/query_labels.csv'
main=[];query=[];gt=[];recorded_main=[];recorded_query=[];delta=[]
with p.open(newline='') as f:
    for r in csv.DictReader(f):
        main.append(json.loads(r['main_box']));query.append(json.loads(r['query_box']))
        gt.append(targets[r['sequence']][int(r['frame'])])
        recorded_main.append(float(r['main_iou']));recorded_query.append(float(r['query_iou']));delta.append(float(r['delta_from_stop']))
gt=np.asarray(gt,dtype=np.float32);main=np.asarray(main,dtype=np.float32);query=np.asarray(query,dtype=np.float32)
assert len(gt)==399401 and np.isfinite(gt).all() and (gt[:,2:]>0).all()
def iou(p,t):
    inter=np.maximum(np.minimum(p[:,:2]+p[:,2:],t[:,:2]+t[:,2:])-np.maximum(p[:,:2],t[:,:2]),0).prod(1)
    return inter/(p[:,2:].prod(1)+t[:,2:].prod(1)-inter)
a=iou(main,gt);b=iou(query,gt)
checks={}
for key,x,y in [('main_iou',a,recorded_main),('query_iou',b,recorded_query),('delta_from_stop',b-a,delta)]:
    y=np.asarray(y,dtype=np.float32)
    diff=float(np.max(np.abs(x-y)))
    assert diff<=1e-7,(key,diff)
    checks[key]=dict(max_abs_diff=diff,exact_rows=int((x==y).sum()),total_rows=len(x))
counts=dict(initializes=2*len(targets),tracked_frames=2*sum(len(x)-1 for x in targets.values()),updates=2*sum(sum(f>0 and f%5==0 for f in s['valid_frames']) for s in manifest['train']))
assert counts==dict(initializes=1764,tracked_frames=917106,updates=182182)
print(json.dumps(dict(status='PASS',train_gt_sequences=len(targets),counts=counts,geometry_checks=checks,annotation_sha256=hashes,scope='Stored executed boxes versus original TRAIN GT only; no neural replay.'),indent=2))

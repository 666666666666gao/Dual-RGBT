"""Independent CPU-only artifact verifier; never imports project neural modules."""
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path('C:/Users/gb/projects/Dual-RGBT')
OUT = Path(__file__).parent
TOOLKIT = Path('C:/Users/gb/Desktop/document/cadtrack/toolkit')
sys.path.insert(0, str(TOOLKIT))
from rgbt.dataset.basedataset import BaseRGBTDataet, TrackerResult
from rgbt.metrics.metrics import PR_LasHeR, NPR, SR_LasHeR

hashes = {}
def sha(p):
    p=Path(p); result=hashlib.sha256(p.read_bytes()).hexdigest()
    hashes[str(p)]=result
    return result
def read(rel):
    p=ROOT/rel; sha(p); return json.loads(p.read_bytes())
def equal(a,b,path=''):
    if isinstance(a,dict):
        assert set(a)==set(b),(path,set(a)^set(b))
        for k in a: equal(a[k],b[k],path+'/'+k)
    elif isinstance(a,list):
        assert len(a)==len(b),(path,len(a),len(b))
        for i,(x,y) in enumerate(zip(a,b)): equal(x,y,path+'/'+str(i))
    elif isinstance(a,float):
        assert np.isclose(a,b,rtol=0,atol=2e-12),(path,a,b)
    else: assert a==b,(path,a,b)

plan=read('reports/rankstop_dev_plan_server26_20261026.json')
plan_sha=sha(ROOT/'reports/rankstop_dev_plan_server26_20261026.json')
source={p:sha(ROOT/p) for p in plan['source_sha256']}
assert source==plan['source_sha256']
toolkit={p:sha(TOOLKIT/p) for p in plan['toolkit_code_sha256']}
assert toolkit==plan['toolkit_code_sha256']
manifest=read(plan['manifests']['dev'])
names=[x['name'] for x in manifest['sequences']]
assert len(names)==len(set(names))==97
assert sum(len(s['boxes']) for s in manifest['sequences'])==54646
summary=read('reports/RANKSTOP_COMPLETE_DEV_RESULT_20261026.json')
assert summary['checkpoint_sha256']==plan['checkpoint_sha256']
assert summary['plan_sha256']==plan_sha
reference=read('reports/candidate_complete_adopted_20261016/e1_dev_cost0p15.json')
assert plan['selection']['reference_dev']=={k:reference[k] for k in ['PR','NPR','SR']}
assert summary['reference_dev']==plan['selection']['reference_dev']
all_diag=[]; metrics=[]; diagnostics=[]; files_checked=0
for cost in [0.,.05,.15]:
    entry=plan['runs'][str(cost)]; run=ROOT/'runs'/entry['run']
    config=read(run/'config.json'); complete=read(run/'COMPLETE.json')
    nativepath=ROOT/'runs/pipeline_tasks'/entry['task']/'status.json'
    native=read(nativepath); progress=read(run/'progress.json'); report=read(entry['metrics'])
    arrays=np.load(ROOT/Path(entry['metrics']).with_suffix('.npz'),allow_pickle=False)
    sha(ROOT/Path(entry['metrics']).with_suffix('.npz'))
    assert native['state']=='COMPLETE' and native['exit_code']==0
    assert native['task']==complete['task']==config['task']==entry['task']
    assert native['command']==config['native_command']
    assert native['physical_gpu']==config['physical_gpu'] and native['physical_gpu'] in ['0','1']
    assert config['plan_sha256']==complete['plan_sha256']==report['plan_sha256']==plan_sha
    assert config['source_sha256']==source
    assert config['checkpoint_sha256']==complete['checkpoint_sha256']==report['checkpoint_sha256']==plan['checkpoint_sha256']
    assert config['checkpoint']==plan['checkpoint'] and config['cost_weight']==report['cost_weight']==cost
    assert config['run']==entry['run'] and config['split']=='dev' and not config['smoke_frames']
    assert not complete['smoke_only'] and complete['backward_calls']==complete['optimizer_updates']==0
    assert complete['frames']==config['split_frames']==report['frames']==54646
    assert complete['sequences']==config['split_sequences']==report['sequences']==97
    assert config['sequences']==names
    assert config['manifest_sha256']==sha(ROOT/plan['manifests']['dev'])
    assert report['native_task_sha256']==sha(nativepath)
    for f in ['predictions','groundtruth','traces']:
        assert {p.stem for p in (run/f).iterdir()}==set(names)
    for flag in ['initialization_included','image_decoding_included','model_loading_excluded']:
        assert report[flag] and complete[flag]
    assert config['total_parameters']==complete['total_parameters']==report['total_parameters']==103218887
    assert config['gpu_name']==complete['gpu_name']==report['gpu_name']
    assert config['input_sizes']==complete['input_sizes']==report['input_sizes']
    timer=np.load(run/'frame_seconds.npy');sha(run/'frame_seconds.npy')
    assert timer.shape==(54646,) and np.isfinite(timer).all() and (timer>0).all()
    timing=dict(fps=len(timer)/float(timer.sum()),mean_ms=1000*float(timer.mean()),p95_ms=1000*float(np.quantile(timer,.95)))
    for k,v in timing.items():
        assert np.isclose(v,report[k],atol=1e-11,rtol=0),(cost,k,v,report[k])
        assert np.isclose(v,complete[k],atol=1e-11,rtol=0)
    assert complete['wall_seconds']>=timer.sum()
    assert native['completed']-native['started']>=complete['wall_seconds']
    gt=BaseRGBTDataet(str(run/'groundtruth'),names,'ltwh')
    predictions=TrackerResult('audit',str(run/'predictions'),names,'','ltwh')
    official={}
    for key,metric in [('PR',PR_LasHeR()),('NPR',NPR()),('SR',SR_LasHeR())]:
        _,curve=metric(gt,predictions,names)
        assert np.array_equal(curve,arrays[key]),(cost,key,float(np.max(np.abs(curve-arrays[key]))))
        official[key]=100*float(curve[:,20].mean() if key!='SR' else curve.mean())
        assert official[key]==report[key]
    groups=defaultdict(lambda:defaultdict(float)); perseq=[]
    trace_count=triggers=badtriggers=bad20=offset=official_invalid=raw_invalid=0
    for index,(name,seq,p) in enumerate(zip(names,manifest['sequences'],progress)):
        assert name==seq['name']==p['sequence']==report['per_sequence'][index]['sequence']
        n=len(seq['boxes']); assert n==len(seq['rgb'])==len(seq['tir'])==p['frames']
        paths=[run/'predictions'/(name+'.txt'),run/'groundtruth'/(name+'.txt'),run/'traces'/(name+'.csv')]
        for path in paths:sha(path)
        pred=np.loadtxt(paths[0],delimiter=',',ndmin=2)
        target=np.loadtxt(paths[1],delimiter=',',dtype=np.float32,ndmin=2)
        assert pred.shape==target.shape==(n,4) and np.isfinite(pred).all() and (pred[:,2:]>0).all()
        assert np.array_equal(target,np.asarray(seq['boxes'],dtype=np.float32))
        assert np.allclose(pred[0],np.asarray(seq['initial_box'],dtype=np.float32),rtol=0,atol=5.01e-7)
        t=target.astype(np.float64)
        valid=np.isfinite(t).all(1)&(t[:,2:]>0).all(1)
        inter=np.maximum(np.minimum(pred[:,:2]+pred[:,2:],t[:,:2]+t[:,2:])-np.maximum(pred[:,:2],t[:,:2]),0).prod(1)
        iou=np.full(n,np.nan);iou[valid]=inter[valid]/(pred[valid,2:].prod(1)+t[valid,2:].prod(1)-inter[valid])
        raw_invalid+=int((~valid).sum());official_invalid+=int((target.round(0)<=0).any(1).sum())
        rows=list(csv.DictReader(paths[2].open(newline='')))
        assert [int(x['frame']) for x in rows]==list(range(1,n))
        assert np.array_equal(np.array([float(x['seconds_including_decode']) for x in rows]),timer[offset+1:offset+n])
        actions=np.zeros(5,dtype=int);modalities=np.zeros(2,dtype=int)
        st=int(valid[0] and iou[0]<.1);maxst=st;tr=bt=b20=0
        for row in rows:
            f=int(row['frame']);a=int(row['action']);m=int(row['modality']);c=float(row['confidence'])
            assert a in range(5) and m in [0,1] and np.isfinite(c)
            assert 0<float(row['seconds'])<=float(row['seconds_including_decode'])
            assert int(row['primary_tokens'])==385
            assert int(row['auxiliary_tokens'])==(0 if a==0 else 385 if a==4 else 193)
            cnt=int(row['candidate_count']);req=int(row['request_index']);rank=int(row['candidate_rank'])
            assert 1<=cnt<=8
            if a==0: assert req==-1 and rank==-1
            elif a in [1,2]:assert req==a-1 and rank==-1
            else:assert 0<=rank<cnt and req==2+2*rank+(a==4)
            actions[a]+=1;modalities[m]+=1
            g=groups[str(a)];g['frames']+=1;g['tir_primary']+=m==1;g['sum_confidence']+=c
            if valid[f]:g['valid_gt_frames']+=1;g['sum_raw_iou']+=iou[f];g['raw_iou_below_0p1']+=iou[f]<.1
            trigger=(f%5==0 and c>=.65);low=valid[f] and iou[f]<.1
            if trigger:
                tr+=1;g['rule_triggers']+=1
                if low:bt+=1;g['low_iou_rule_triggers']+=1;b20+=st>=20
            st=st+1 if low else 0;maxst=max(maxst,st)
        assert actions.tolist()==p['action_counts'] and modalities.tolist()==p['modality_counts']
        off={k:100*float(arrays[k][index,20] if k!='SR' else arrays[k][index].mean()) for k in ['PR','NPR','SR']}
        assert off=={k:report['per_sequence'][index][k] for k in off}
        perseq.append(dict(sequence=name,**off,frames=n,valid_gt_frames=int(valid.sum()),raw_iou_mean=float(iou[valid].mean()),action_counts=actions.tolist(),rule_triggers=tr,low_iou_rule_triggers=bt,low_iou_rule_triggers_after20=int(b20),max_low_iou_streak=int(maxst)))
        trace_count+=len(rows);triggers+=tr;badtriggers+=bt;bad20+=b20;offset+=n
    assert offset==54646 and trace_count==54549 and len(progress)==97
    for g in groups.values():g['mean_confidence']=g.pop('sum_confidence')/g['frames'];g['raw_iou_mean']=g.pop('sum_raw_iou')/g['valid_gt_frames']
    arm=dict(cost_weight=cost,official={k:report[k] for k in ['PR','NPR','SR','total_parameters','fps']},frames=offset,tracked_frames=trace_count,action_groups=dict(groups),rule_triggers=triggers,low_iou_rule_triggers=badtriggers,low_iou_rule_triggers_after20=int(bad20),per_sequence=perseq)
    all_diag.append(arm)
    sr=next(r for r in summary['rows'] if r['cost_weight']==cost)
    assert sr['metrics_sha256']==sha(ROOT/entry['metrics'])
    for k in ['PR','NPR','SR','fps','mean_ms','p95_ms','total_parameters']:assert sr[k]==report[k]
    metrics.append(dict(cost_weight=cost,**official,**timing,total_parameters=report['total_parameters'],gpu=report['gpu_name'],input_sizes=report['input_sizes'],summed_service_seconds=float(timer.sum()),wall_seconds=complete['wall_seconds'],native_exit_code=native['exit_code'],official_invalid_gt=official_invalid,raw_invalid_gt=raw_invalid,curves_exact=True,per_sequence_exact=True,progress_trace_times_exact=True,rule_triggers=triggers,low_iou_rule_triggers=badtriggers,low_iou_rule_triggers_after20=int(bad20)))
    print(json.dumps(metrics[-1]),flush=True)

for path in ['reports/RANKSTOP_COMPLETED_DEV_DIAGNOSTIC_20261026.json','reports/RANKSTOP_THREE_DEV_DIAGNOSTIC_20261026.json']:
    d=read(path);assert d['plan_sha256']==plan_sha
    assert d['producer_sha256']==sha(ROOT/'tools/analyze_rankstop_completed_dev.py')
    for p,h in d['input_sha256'].items():assert sha(ROOT/p)==h
    equal(d['arms'],all_diag[:len(d['arms'])],path)
    diagnostics.append(dict(path=path,arms=len(d['arms']),all_fields_match=True,input_hashes=len(d['input_sha256'])))
best=max(metrics,key=lambda x:(x['SR'],x['PR'],x['NPR'],-x['mean_ms']))
assert best['cost_weight']==summary['best_cost_weight']==0
eligible=best['SR']>reference['SR'];assert eligible==summary['exceeds_e1_dev_sr']==False
assert not summary['test_launched'] and summary['proposed_decision']=='REJECT_REPAIR_NO_TEST'
result=dict(status='PASS',source_hashes_checked=len(source),toolkit_hashes_checked=len(toolkit),plan_sha256=plan_sha,metrics=metrics,diagnostics=diagnostics,best_cost=best['cost_weight'],reference={k:reference[k] for k in ['PR','NPR','SR']},best_minus_reference={k:best[k]-reference[k] for k in ['PR','NPR','SR']},test_eligible=eligible,audited_input_hashes=hashes)
(OUT/'local_verification.json').write_text(json.dumps(result,indent=2)+'\n')
print('LOCAL_VERIFICATION_PASS',len(hashes),flush=True)

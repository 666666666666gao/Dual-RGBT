"""Describe completed dev trajectories; never infer unavailable query gains."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def raw_iou(prediction, target):
    valid = np.isfinite(target).all(1) & (target[:, 2:] > 0).all(1)
    left = np.maximum(prediction[:, :2], target[:, :2])
    right = np.minimum(prediction[:, :2]+prediction[:, 2:], target[:, :2]+target[:, 2:])
    overlap = np.maximum(right-left, 0).prod(1)
    result = np.full(len(target), np.nan)
    result[valid] = overlap[valid] / (prediction[valid, 2:].prod(1)+target[valid, 2:].prod(1)-overlap[valid])
    return result, valid


def summarize(plan, cost):
    entry = plan['runs'][str(cost)]
    run = ROOT/'runs'/entry['run']
    metric_path = ROOT/entry['metrics']
    metric = json.loads(metric_path.read_bytes())
    config = json.loads((run/'config.json').read_bytes())
    native_path = ROOT/'runs/pipeline_tasks'/entry['task']/'status.json'
    native = json.loads(native_path.read_bytes())
    assert native['state']=='COMPLETE' and native['exit_code']==0
    assert metric['checkpoint_sha256']==config['checkpoint_sha256']==plan['checkpoint_sha256']
    assert metric['cost_weight']==config['cost_weight']==cost
    assert metric['split']=='dev' and metric['sequences']==97 and metric['frames']==54646
    saved = {row['sequence']:row for row in metric['per_sequence']}
    assert set(saved)==set(config['sequences']) and len(saved)==97
    curves = np.load(metric_path.with_suffix('.npz'),allow_pickle=False)
    hashes = {str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in
              (metric_path,metric_path.with_suffix('.npz'),run/'config.json',run/'COMPLETE.json',native_path)}
    actions = defaultdict(lambda: defaultdict(float))
    sequences, supplied_gt = [], {}
    frames = tracked = triggers = bad_triggers = bad_after20 = 0
    for index,name in enumerate(config['sequences']):
        official = saved[name]
        assert official['PR']==100*curves['PR'][index,20]
        assert official['NPR']==100*curves['NPR'][index,20]
        assert official['SR']==100*curves['SR'][index].mean()
        paths=[run/'predictions'/(name+'.txt'),run/'groundtruth'/(name+'.txt'),run/'traces'/(name+'.csv')]
        for path in paths: hashes[str(path.relative_to(ROOT)).replace('\\','/')]=sha(path)
        prediction=np.loadtxt(paths[0],delimiter=',',ndmin=2)
        target=np.loadtxt(paths[1],delimiter=',',dtype=np.float32,ndmin=2).astype(np.float64)
        assert prediction.shape==target.shape and prediction.shape[1]==4
        assert np.isfinite(prediction).all() and (prediction[:,2:]>0).all()
        supplied_gt[name]=target
        iou,valid=raw_iou(prediction,target)
        with paths[2].open(newline='') as handle: trace=list(csv.DictReader(handle))
        assert [int(row['frame']) for row in trace]==list(range(1,len(target)))
        local_actions=np.zeros(5,dtype=int)
        local_triggers=local_bad=local_bad20=0
        prior_bad_streak=int(valid[0] and iou[0]<.1)
        max_bad_streak=prior_bad_streak
        for row in trace:
            frame,action,modality=int(row['frame']),int(row['action']),int(row['modality'])
            confidence=float(row['confidence'])
            assert action in range(5) and modality in (0,1) and np.isfinite(confidence)
            assert int(row['primary_tokens'])==385
            assert int(row['auxiliary_tokens'])==(0 if action==0 else 385 if action==4 else 193)
            item=actions[str(action)]
            item['frames']+=1
            item['tir_primary']+=modality==1
            item['sum_confidence']+=confidence
            local_actions[action]+=1
            if valid[frame]:
                item['valid_gt_frames']+=1
                item['sum_raw_iou']+=iou[frame]
                item['raw_iou_below_0p1']+=iou[frame]<.1
            trigger=frame%5==0 and confidence>=.65
            low=valid[frame] and iou[frame]<.1
            if trigger:
                local_triggers+=1
                item['rule_triggers']+=1
                if low:
                    local_bad+=1
                    item['low_iou_rule_triggers']+=1
                    local_bad20+=prior_bad_streak>=20
            prior_bad_streak=prior_bad_streak+1 if low else 0
            max_bad_streak=max(max_bad_streak,prior_bad_streak)
        frames+=len(target); tracked+=len(trace)
        triggers+=local_triggers; bad_triggers+=local_bad; bad_after20+=local_bad20
        sequences.append(dict(**official,frames=len(target),valid_gt_frames=int(valid.sum()),
            raw_iou_mean=float(np.mean(iou[valid])),action_counts=local_actions.tolist(),
            rule_triggers=local_triggers,low_iou_rule_triggers=local_bad,
            low_iou_rule_triggers_after20=local_bad20,max_low_iou_streak=max_bad_streak))
    assert frames==54646 and tracked==54549
    for item in actions.values():
        item['mean_confidence']=item.pop('sum_confidence')/item['frames']
        item['raw_iou_mean']=item.pop('sum_raw_iou')/item['valid_gt_frames']
    return dict(cost_weight=cost,official={k:metric[k] for k in ('PR','NPR','SR','total_parameters','fps')},
                frames=frames,tracked_frames=tracked,action_groups=dict(actions),
                rule_triggers=triggers,low_iou_rule_triggers=bad_triggers,
                low_iou_rule_triggers_after20=bad_after20,per_sequence=sequences),hashes,supplied_gt


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',required=True)
    parser.add_argument('--cost-weights',type=float,nargs='+',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    plan_path=ROOT/args.plan
    plan=json.loads(plan_path.read_bytes())
    arms,inputs,reference=[],{},None
    for cost in args.cost_weights:
        arm,hashes,gt=summarize(plan,cost)
        if reference is not None:
            assert set(gt)==set(reference) and all(np.array_equal(gt[name],reference[name]) for name in gt)
        else:
            reference=gt
        arms.append(arm); inputs.update(hashes)
    report=dict(status='COMPLETED_DEV_DESCRIPTIVE_CHECK_PASS',arms=arms,input_sha256=inputs,
        plan_sha256=sha(plan_path),producer_sha256=sha(Path(__file__)),
        scope='Completed full97 DEV only. Official per-sequence metrics are checked against unchanged scorer NPZ. '
              'Raw per-frame IoU uses saved predictions and unrounded float32 GT without GT clipping; '
              'it is a diagnostic, not official SR. Invalid GT excluded from raw diagnostics. '
              'Rule triggers are inferred from frame/confidence/source, not observed tensor-write events. '
              'Low-IoU queries do not establish harmful gains because primary pre-query boxes were not saved. '
              'Separate cost trajectories have different histories: differences are not same-state causal effects. '
              'Mixed device load precludes paired speedup inference. No active run/NN/weight reads.')
    output=ROOT/args.output
    assert not output.exists()
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in arm.items() if k!='per_sequence'} for arm in arms],indent=2))


if __name__=='__main__':
    main()

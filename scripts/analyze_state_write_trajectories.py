"""Temporal associations in the archived C20 dev traces; CPU only, no replay."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from analyze_full_dev_coverage import analyse_sequence


def future_summary(indices, values, horizon):
    iou, valid = values['raw_iou'], values['valid']
    means, recovered_at, persist = [], [], 0
    short, invalid = 0, 0
    for index in indices:
        if index + horizon >= len(iou):
            short += 1
            continue
        selected = slice(index + 1, index + horizon + 1)
        if not valid[selected].all():
            invalid += 1
            continue
        future = iou[selected]
        means.append(float(future.mean()))
        persist += int((future < .1).all())
        recovery = np.flatnonzero(future >= .5)
        if len(recovery):
            recovered_at.append(int(recovery[0]) + 1)
    assert len(means) + short + invalid == len(indices)
    return dict(anchors=len(indices), complete_valid_windows=len(means),
        excluded_sequence_end=short, excluded_invalid_gt=invalid,
        mean_future_iou=float(np.mean(means)) if means else None,
        low_iou_entire_window=persist, reaches_iou0p5=len(recovered_at),
        recovery_frame_sum=sum(recovered_at))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--events', required=True)
    args = parser.parse_args()
    run, output, events_path = Path(args.run), Path(args.output), Path(args.events)
    assert not output.exists() and not events_path.exists()
    reference_path = Path(args.reference)
    reference = json.loads(reference_path.read_bytes())
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    config = json.loads((run/'config.json').read_bytes())
    complete = json.loads((run/'COMPLETE.json').read_bytes())
    assert config['split']=='dev' and config['smoke_frames']==0 and config['shards']==1 and config['shard']==0
    assert config['checkpoint_sha256']==reference['checkpoint_sha256']
    assert sha(run/'config.json')==reference['input_config_sha256']
    assert sha(run/'COMPLETE.json')==reference['input_complete_sha256']
    assert config['cost_weight']==reference['cost_weight']==.05
    names=config['sequences']
    assert len(names)==len(set(names))==complete['sequences']==config['split_sequences']==97
    assert complete['frames']==config['split_frames']==54646 and not complete['smoke_only']
    for directory,suffix in (('predictions','*.txt'),('groundtruth','*.txt'),('traces','*.csv')):
        assert {p.stem for p in (run/directory).glob(suffix)}==set(names)
    hashes={row['sequence']:row['input_sha256'] for row in reference['per_sequence']}
    assert set(hashes)==set(names)
    horizons=(5,20,50)
    labels=('low_iou_update','low_iou_no_update_eligible','good_iou_update','intermediate_iou_update')
    records, per_sequence, episodes, calibration = [], [], [], []
    all_windows={label:{str(h):[] for h in horizons} for label in labels}
    all_predecessor={key:0 for key in ('low_iou','good_iou','intermediate_iou','invalid_or_unobserved')}
    prior_lengths=[]
    valid_total=0
    for name in names:
        base, values=analyse_sequence(run,name)
        assert base['input_sha256']==hashes[name]
        iou,valid,confidence=values['raw_iou'],values['valid'],values['confidence']
        assert ((confidence>=0)&(confidence<=1)).all()
        frames=np.arange(1,len(valid)+1)
        eligible=frames%5==0
        update=eligible&(confidence>=.65)
        low=valid&(iou<.1)
        good=valid&(iou>=.5)
        middle=valid&~low&~good
        masks=dict(low_iou_update=low&update,low_iou_no_update_eligible=low&eligible&~update,
            good_iou_update=good&update,intermediate_iou_update=middle&update)
        previous_bad=np.zeros(len(valid),dtype=np.int64)
        streak=0
        for index in range(len(valid)):
            previous_bad[index]=streak
            streak=streak+1 if low[index] else 0
        indices=np.flatnonzero(masks['low_iou_update'])
        previous={key:0 for key in all_predecessor}
        for index in indices:
            if index==0 or not valid[index-1]:
                previous['invalid_or_unobserved']+=1
            elif low[index-1]:
                previous['low_iou']+=1
            elif good[index-1]:
                previous['good_iou']+=1
            else:
                previous['intermediate_iou']+=1
            records.append(dict(sequence=name,frame=int(frames[index]),action=int(values['action'][index]),
                modality=int(values['modality'][index]),confidence=float(confidence[index]),
                raw_iou=float(iou[index]),preceding_consecutive_low_frames=int(previous_bad[index]),
                target_centre_inside=bool(values['inside'][index])))
        assert sum(previous.values())==len(indices)
        for key,value in previous.items():
            all_predecessor[key]+=value
        prior_lengths.extend(previous_bad[indices].tolist())
        starts=np.flatnonzero(low&~np.concatenate(([False],low[:-1])))
        ends=np.flatnonzero(low&~np.concatenate((low[1:],[False])))
        assert len(starts)==len(ends)
        for start,end in zip(starts,ends):
            episodes.append(dict(sequence=name,start_frame=int(start)+1,end_frame=int(end)+1,
                frames=int(end-start+1),update_events=int(update[start:end+1].sum()),
                ends_at_sequence_end=bool(end==len(valid)-1),
                ends_at_invalid_gt=bool(end<len(valid)-1 and not valid[end+1])))
        for label,mask in masks.items():
            for horizon in horizons:
                all_windows[label][str(horizon)].append(future_summary(np.flatnonzero(mask),values,horizon))
        bins=[]
        for lower,upper in ((.65,.75),(.75,.85),(.85,.95),(.95,1.)):
            mask=valid&eligible&(confidence>=lower)&(confidence<upper if upper<1 else confidence<=upper)
            bins.append(dict(lower=lower,upper=upper,count=int(mask.sum()),low_iou=int((mask&low).sum()),
                iou_sum=float(iou[mask].sum())))
        calibration.append(bins)
        per_sequence.append(dict(sequence=name,frames=base['frames'],valid_frames=int(valid.sum()),
            inferred_update_events=int((valid&update).sum()),low_iou_update_events=len(indices),
            low_iou_no_update_eligible=int(masks['low_iou_no_update_eligible'].sum()),
            low_iou_updates_after20_bad_frames=int((previous_bad[indices]>=20).sum()),input_sha256=base['input_sha256']))
        valid_total+=int(valid.sum())
    assert sum(row['frames'] for row in per_sequence)==54646 and valid_total==reference['valid_frames']
    assert sum(row['inferred_update_events'] for row in per_sequence)==reference['groups']['template_update']['frames']
    assert len(records)==reference['groups']['update_low_iou']['frames']==1053
    assert sum(row['inferred_update_events'] for row in per_sequence)==6430
    windows={}
    for label in labels:
        windows[label]={}
        for horizon in horizons:
            rows=all_windows[label][str(horizon)]
            total={key:sum(row[key] for row in rows) for key in rows[0] if key!='mean_future_iou'}
            count=total['complete_valid_windows']
            total['mean_future_iou']=sum(row['mean_future_iou']*row['complete_valid_windows'] for row in rows if row['complete_valid_windows'])/count if count else None
            windows[label][str(horizon)]=total
    pooled_bins=[]
    for index in range(4):
        row={key:sum(sequence[index][key] for sequence in calibration) for key in ('count','low_iou','iou_sum')}
        row.update(lower=calibration[0][index]['lower'],upper=calibration[0][index]['upper'])
        pooled_bins.append(row)
    assert sum(row['count'] for row in pooled_bins)==6430
    assert sum(row['low_iou'] for row in pooled_bins)==1053
    with events_path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    root=Path(__file__).resolve().parents[1]
    report=dict(scope='Archived C20 full-dev temporal associations, not intervention, official metrics or query gains.',
        sequences=97,frames=54646,valid_tracked_frames=valid_total,checkpoint_sha256=config['checkpoint_sha256'],
        input_run=str(run.resolve()),reference_sha256=sha(reference_path),input_config_sha256=sha(run/'config.json'),
        input_complete_sha256=sha(run/'COMPLETE.json'),source_sha256={name:sha(root/name) for name in
            ('scripts/analyze_state_write_trajectories.py','scripts/analyze_full_dev_coverage.py','dual_rgbt/tracker.py')},
        inferred_update_events=6430,low_iou_update_events=len(records),low_iou_update_predecessor=all_predecessor,
        low_iou_update_prior_streak_bins={label:sum(lower<=value<upper for value in prior_lengths) for label,lower,upper in
            (('zero',0,1),('one_to_four',1,5),('five_to_nineteen',5,20),('twenty_or_more',20,float('inf')))},
        low_iou_episodes=dict(count=len(episodes),low_frames=sum(row['frames'] for row in episodes),
            with_update=sum(row['update_events']>0 for row in episodes),long_at_least20=sum(row['frames']>=20 for row in episodes),
            end_at_sequence_end=sum(row['ends_at_sequence_end'] for row in episodes),end_at_invalid_gt=sum(row['ends_at_invalid_gt'] for row in episodes)),
        update_confidence_bins=pooled_bins,future_windows=windows,private_events_sha256=sha(events_path),per_sequence=per_sequence,
        neural_execution=False,limitations=[
            'One C20 checkpoint on archived97 dev; not full882 C, final test, new E1 or a multi-seed estimate.',
            'Raw continuous float32 geometry and positive-width/height GT validity, not toolkit PR/NPR/SR.',
            'Inferred write event uses unchanged frame%5 and confidence>=0.65; events are not modality write counts.',
            'Preceding low-IoU streak excludes frame zero and resets at invalid GT; invalid previous frames are separate.',
            'Future windows exclude sequence-short or any invalid future GT; horizons never cross sequence boundaries.',
            'Windows overlap and events within a sequence are dependent; counts are not independent error probabilities.',
            'Low-confidence nonwrites and high-confidence writes are not randomized or causally matched controls.',
            'No main-only counterfactual or template rollback; selected bad-query output does not establish harmful auxiliary gain.',
            'Associations cannot establish that template writes cause, prevent or worsen future drift.'])
    output.write_bytes((json.dumps(report,indent=2,allow_nan=False)+'\n').encode())
    print(json.dumps({key:value for key,value in report.items() if key not in ('per_sequence','source_sha256')},indent=2,allow_nan=False))


if __name__=='__main__':
    main()

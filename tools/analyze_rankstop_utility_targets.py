"""Audit old saved TRAIN utility targets and known same-state request pairs."""
import csv
import hashlib
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):
            value.update(block)
    return value.hexdigest()


def main():
    run=ROOT/'runs/global_candidate_rankstop_trajectory_formal_20261018'
    labels,steps=run/'query_labels.csv',run/'trajectory_steps.csv'
    assert digest(labels)=='6ddfdb92feae3294ea85211a672fe6bf1b07cdcdd5a8abc1dfdf38b908bbfa47'
    assert digest(steps)=='2f22103097fdcaeb4483841cf5456160938a2ba0baadf393a7d70317f0a14e50'
    gains,kinds,purposes=[],[],[]
    pairs=defaultdict(lambda:defaultdict(int))
    with labels.open(newline='') as label_file,steps.open(newline='') as step_file:
        grouped=itertools.groupby(csv.DictReader(label_file),key=lambda row:int(row['step']))
        count=0
        for decision,(step,values) in zip(csv.DictReader(step_file),grouped):
            values=list(values)
            assert step==int(decision['step'])==count+1
            assert all(all(v[k]==decision[k] for k in ('round','sequence','frame')) for v in values)
            sampled={int(v['modality']):v for v in values if v['purpose']=='sampled'}
            assert set(sampled)=={0,1}
            for row in values:
                gains.append(float(row['delta_from_stop'])); kinds.append(int(row['kind'])); purposes.append(row['purpose'])
            extra=[v for v in values if v['purpose']=='selected_extra']
            if extra:
                assert len(extra)==1 and decision['extra_selected_query']=='True'
                role=int(decision['chosen_modality']); selected=extra[0]; alternative=sampled[role]
                assert int(selected['modality'])==role and int(selected['kind'])+1==int(decision['chosen_action'])
                assert selected['candidate_rank']==decision['chosen_rank']
                assert selected['main_box']==alternative['main_box'] and selected['main_geometry']==alternative['main_geometry']
                assert selected['main_iou']==alternative['main_iou']
                same_kind=selected['kind']==alternative['kind']
                assert not same_kind or selected['candidate_rank']!=alternative['candidate_rank']
                selected_gain=np.float32(selected['delta_from_stop']); alternative_gain=np.float32(alternative['delta_from_stop'])
                cs=np.float32(1 if int(selected['kind'])==3 else 193/385)
                ca=np.float32(1 if int(alternative['kind'])==3 else 193/385)
                selected_utility=np.float32(selected_gain-np.float32(.05)*cs)
                alternative_utility=np.float32(alternative_gain-np.float32(.05)*ca)
                for key in ('all','same_kind_other_rank' if same_kind else 'different_kind'):
                    item=pairs[key]; item['pairs']+=1
                    item['sampled_raw_gain_better_over_0p01']+=int(alternative_gain-selected_gain>np.float32(.01))
                    item['sampled_net_utility_better_over_0p01']+=int(alternative_utility-selected_utility>np.float32(.01))
                    item['selected_negative_utility_sampled_positive']+=int(selected_utility<0<alternative_utility)
                    item['both_utilities_negative']+=int(selected_utility<0 and alternative_utility<0)
                    item['selected_kind_'+selected['kind']]+=1
            else:
                assert decision['extra_selected_query']=='False'
            count+=1
        assert next(grouped,None) is None
    assert count==182182 and len(gains)==399401 and pairs['all']['pairs']==35037
    gain=np.asarray(gains,dtype=np.float32); kind=np.asarray(kinds,dtype=np.int64); purpose=np.asarray(purposes)
    cost=np.where(kind==3,np.float32(1),np.float32(193/385)).astype(np.float32)
    active=np.abs(gain)>np.float32(.01)
    # Isolated-label optimum: the old .25 hinge outweighs its <=.04 regression error.
    isolated=np.where(active,np.sign(gain)*np.maximum(np.abs(gain),np.float32(.05)),gain)
    rows=[]
    for name in ('sampled','selected_extra'):
        for family in range(4):
            mask=(purpose==name)&(kind==family)
            row=dict(purpose=name,kind=family,labels=int(mask.sum()),
                exact_gain_violates_fixed_margin=int((mask&active&(np.abs(gain)<np.float32(.05))).sum()))
            for weight in (0.,.05,.15):
                true_utility=gain-np.float32(weight)*cost
                ideal_prediction_utility=isolated-np.float32(weight)*cost
                row['idealized_wrong_call_cost_'+str(weight)]=int((mask&(true_utility<=0)&(ideal_prediction_utility>0)).sum())
            rows.append(row)
    target=np.float32(.02); threshold=np.float32(.05)*np.float32(193/385); optimum=np.float32(.05)
    report=dict(status='SAVED_TRAIN_UTILITY_TARGET_ARITHMETIC_PASS',labels=len(gains),updates=count,
        pairs=dict(pairs),label_groups=rows,
        toy=dict(raw_gain=float(target),query_cost=float(threshold),calibrated_utility=float(target-threshold),
                 isolated_old_optimum_prediction=float(optimum),isolated_predicted_utility=float(optimum-threshold),
                 old_loss_at_calibrated=float(.25*(.05-float(target))),
                 old_loss_at_isolated_optimum=float(.5*(float(optimum)-float(target))**2)),
        input_sha256=dict(labels=digest(labels),steps=digest(steps)),
        scope='Archived evolving TRAIN labels only. Fixed-margin conflict and isolated-label optimum are '
              'objective diagnostics, not measured final-model calibration or deployed wrong-call rates. '
              'The real shared network couples predictions. Candidate pairs are only selected-extra versus '
              'one sampled request at the same role/state; no full-pool oracle or trajectory upper bound. '
              'Token costs use FP32 and traininglambda.05. No NN/active-training reads or changes.')
    output=ROOT/'reports/RANKSTOP_UTILITY_TARGET_DIAGNOSTIC_20261026.json'
    assert not output.exists(); output.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report))


if __name__=='__main__':
    main()

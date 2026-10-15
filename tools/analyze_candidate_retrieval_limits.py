"""CPU diagnosis of archived train-state proposals and executed query outputs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def separated(scores, coordinates, radius):
    remaining = scores.copy()
    chosen = []
    threshold = np.float32(radius ** 2)
    for _ in range(8):
        index = int(remaining.argmax())
        if not np.isfinite(remaining[index]):
            break
        chosen.append(index)
        distance = np.sum((coordinates - coordinates[index]) ** 2, axis=1, dtype=np.float32)
        remaining[distance < threshold] = -np.inf
    return chosen


def geometry(target, points, side):
    left = points - np.float32(side / 2)
    centre = target[:2] + target[2:] / np.float32(2)
    inside = ((centre >= left) & (centre < left + side)).all(1)
    extent = np.maximum(np.float32(0), np.minimum(target[:2] + target[2:], left + side)
                        - np.maximum(target[:2], left))
    area = np.prod(extent, axis=1, dtype=np.float32) / np.prod(target[2:], dtype=np.float32)
    return inside, area


def summarize(states):
    assert states
    result = dict(states=len(states), families={}, proposal_geometry={})
    for family in ('GLOBAL_LOCAL', 'GLOBAL_WIDE'):
        groups = [state['families'][family] for state in states]
        strategies = {}
        for strategy in ('top1', 'nearest_main_crop_centre', 'gt_best8'):
            values = np.asarray([group[strategy]['iou'] for group in groups])
            gains = values - np.asarray([state['main_iou'] for state in states])
            strategies[strategy] = dict(mean_query_iou=float(values.mean()),
                mean_signed_gain=float(gains.mean()), positive_gain=int((gains > 0).sum()),
                substantial_harm=int((gains <= -.1).sum()), query_iou_at_least_half=int((values >= .5).sum()))
        result['families'][family] = dict(strategies=strategies,
            gt_stop_inclusive_best8_gain=float(np.mean([max(group['gt_best8']['iou'] - state['main_iou'], 0)
                                                       for group,state in zip(groups,states)])),
            top1_centre_covered=sum(group['top1']['centre_inside'] for group in groups),
            top1_covered_but_iou_below_tenth=sum(group['top1']['centre_inside'] and group['top1']['iou'] < .1 for group in groups),
            pool_centre_covered=sum(group['pool_centre_covered'] for group in groups),
            pool_area_at_least_half=sum(group['pool_area'] >= .5 for group in groups),
            pool_area_at_least_half_but_best_iou_below_half=sum(group['pool_area'] >= .5 and group['gt_best8']['iou'] < .5 for group in groups),
            top1_below_half_but_best8_at_least_half=sum(group['top1']['iou'] < .5 <= group['gt_best8']['iou'] for group in groups))
        geometry_results = {}
        for variant in ('RAW_TOP8', 'HALF_RADIUS', 'DEFAULT', 'DOUBLE_RADIUS'):
            values = [group['geometry'][variant] for group in groups]
            geometry_results[variant] = dict(centre_covered=sum(value['centre_covered'] for value in values),
                mean_best_gt_area=float(np.mean([value['best_gt_area'] for value in values])),
                default_gained=sum(value['centre_covered'] and not group['geometry']['DEFAULT']['centre_covered'] for value,group in zip(values,groups)),
                default_lost=sum(not value['centre_covered'] and group['geometry']['DEFAULT']['centre_covered'] for value,group in zip(values,groups)),
                same_realized_count_as_default=sum(value['count'] == group['geometry']['DEFAULT']['count'] for value,group in zip(values,groups)))
            for label,eight_only in (('matched_realized_count',False),('both_have_eight',True)):
                paired = [(value,group['geometry']['DEFAULT']) for value,group in zip(values,groups)
                          if value['count']==group['geometry']['DEFAULT']['count']
                          and (not eight_only or value['count']==8)]
                geometry_results[variant][label] = dict(states=len(paired),
                    variant_centre_covered=sum(value['centre_covered'] for value,base in paired),
                    default_centre_covered=sum(base['centre_covered'] for value,base in paired),
                    gained=sum(value['centre_covered'] and not base['centre_covered'] for value,base in paired),
                    lost=sum(not value['centre_covered'] and base['centre_covered'] for value,base in paired))
        result['proposal_geometry'][family] = dict(variants=geometry_results,
            any_valid_grid_centre_covered=sum(group['grid_centre_covered'] for group in groups),
            grid_reachable_but_pool_misses=sum(group['grid_centre_covered'] and not group['pool_centre_covered'] for group in groups),
            raw_top8_covers_but_default_misses=sum(group['geometry']['RAW_TOP8']['centre_covered'] and not group['pool_centre_covered'] for group in groups))
    result['candidate_count_histogram'] = {
        variant: {str(count): sum(state['candidate_counts'][variant] == count for state in states) for count in range(1,9)}
        for variant in ('RAW_TOP8', 'HALF_RADIUS', 'DEFAULT', 'DOUBLE_RADIUS')}
    return result


def main():
    parser = argparse.ArgumentParser()
    for name in ('manifest', 'headers', 'candidates', 'scores', 'utilities', 'verified-analysis',
                 'prior-audit', 'output', 'state-table'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    paths = {name: Path(getattr(args,name)) for name in ('manifest','headers','candidates','scores','utilities')}
    previous = json.loads(Path(args.verified_analysis).read_bytes())
    audit = json.loads(Path(args.prior_audit).read_bytes())
    assert previous['status'] == 'CPU_ANALYSIS_PASS' and previous['states'] == 1752
    assert audit['blocking_issues'] == 0 and audit['deterministic_verdict'] == 'PASS'
    expected = dict(manifest=previous['input_sha256']['manifest'],headers=previous['input_sha256']['headers'],
                    candidates=previous['input_sha256']['candidates'],utilities=previous['input_sha256']['csv'],
                    scores=audit['private_artifact_sha256']['candidate_scores'])
    assert {name:sha(path) for name,path in paths.items()} == expected
    output, table = Path(args.output), Path(args.state_table)
    assert not output.exists() and not table.exists()
    root = Path(__file__).resolve().parents[1]
    formal_plan = json.loads((root/'reports/e1_paired_latency_plan.json').read_bytes())
    assert all(sha(root/name) == value for name,value in formal_plan['source_sha256'].items())
    manifest = json.loads(paths['manifest'].read_bytes())
    train = {row['name']:row for row in manifest['train']}
    headers = json.loads(paths['headers'].read_bytes())['items']
    candidates = json.loads(paths['candidates'].read_bytes())
    scores = np.load(paths['scores'],allow_pickle=False)
    assert len(headers) == len(candidates) == 1752 and scores.shape == (1752,1024)
    with paths['utilities'].open(newline='',encoding='utf-8') as handle:
        raw = list(csv.DictReader(handle))
    assert len(raw) == previous['auxiliary_queries'] == 31384
    by_state = {}
    for row in raw:
        row['candidate_rank'],row['state'],row['frame'] = (int(row[key]) for key in ('candidate_rank','state','frame'))
        row['iou'],row['main_iou'] = float(row['iou']),float(row['main_iou'])
        by_state.setdefault(row['state'],[]).append(row)
    assert set(by_state) == set(range(1752))
    states, individual = [], []
    for state,(name,frame,width,height) in enumerate(headers):
        group,proposal = by_state[state],candidates[state]
        assert all((row['sequence'],row['frame']) == (name,frame) for row in group)
        target = np.asarray(train[name]['boxes'][frame],dtype=np.float32)
        centre = target[:2]+target[2:]/np.float32(2)
        assert np.isfinite(target).all() and (target[2:]>0).all()
        assert 0<=centre[0]<width and 0<=centre[1]<height
        main = np.asarray(json.loads(group[0]['main_geometry']),dtype=np.float32)
        cells = np.arange(1024)
        coordinates = (np.stack([cells%32,cells//32],1).astype(np.float32)+np.float32(.5))/np.float32(32)*np.float32(max(width,height))
        valid = (coordinates[:,0]<width)&(coordinates[:,1]<height)
        assert np.isfinite(scores[state,valid]).all() and np.isneginf(scores[state,~valid]).all()
        # Every proposal variant sees only already-acquired scores and geometry.
        order = np.argsort(-scores[state],kind='stable')
        selections = dict(RAW_TOP8=[int(index) for index in order if valid[index]][:8])
        for variant,multiplier in (('HALF_RADIUS',.5),('DEFAULT',1.),('DOUBLE_RADIUS',2.)):
            selections[variant] = separated(scores[state],coordinates,float(main[2])/4*multiplier)
        assert selections['DEFAULT'] == proposal['indices']
        assert np.array_equal(coordinates[selections['DEFAULT']],np.asarray(proposal['points'],dtype=np.float32))
        assert len(group) == 2+2*len(selections['DEFAULT'])
        nominal_shortside = float(min(target[2:]))*32/max(width,height)
        outside = group[0]['main_centre_inside'] == 'False'
        assert proposal['native_main_outside'] == outside
        record = dict(state=state,sequence=name,frame=frame,main_iou=group[0]['main_iou'],
                      outside=outside,nominal_shortside_cells=nominal_shortside,
                      candidate_counts={key:len(value) for key,value in selections.items()},families={})
        for family,scale in (('GLOBAL_LOCAL',.5),('GLOBAL_WIDE',1.)):
            executed = sorted([row for row in group if row['region']==family],key=lambda row:row['candidate_rank'])
            assert [row['candidate_rank'] for row in executed] == list(range(len(selections['DEFAULT'])))
            side = np.float32(main[2]*scale)
            geometry_results = {}
            for variant,indices in selections.items():
                included,area = geometry(target,coordinates[indices],side)
                geometry_results[variant] = dict(centre_covered=bool(included.any()),best_gt_area=float(area.max()),count=len(indices))
            included,area = geometry(target,coordinates[selections['DEFAULT']],side)
            assert np.isclose(float(area.max()),proposal['by_k']['8']['local' if scale==.5 else 'wide']['best_gt_coverage'],rtol=0,atol=0)
            assert bool(included.any()) == proposal['by_k']['8']['local' if scale==.5 else 'wide']['centre_inside']
            grid_included,_ = geometry(target,coordinates[valid],side)
            nearest = int(np.sum((coordinates[selections['DEFAULT']]-(main[:2]+main[2]/np.float32(2)))**2,axis=1,dtype=np.float32).argmin())
            best = int(np.asarray([row['iou'] for row in executed]).argmax())
            choices = {key:dict(iou=executed[index]['iou'],centre_inside=bool(included[index]),rank=index)
                       for key,index in (('top1',0),('nearest_main_crop_centre',nearest),('gt_best8',best))}
            choices.update(pool_centre_covered=bool(included.any()),pool_area=float(area.max()),
                           grid_centre_covered=bool(grid_included.any()),geometry=geometry_results)
            record['families'][family] = choices
            individual.append(dict(state=state,sequence=name,frame=frame,family=family,outside=outside,
                nominal_shortside_cells=nominal_shortside,main_iou=record['main_iou'],
                top1_iou=choices['top1']['iou'],nearest_main_crop_centre_iou=choices['nearest_main_crop_centre']['iou'],
                best8_iou=choices['gt_best8']['iou'],best8_rank=best,
                top1_centre_inside=bool(included[0]),pool_centre_inside=bool(included.any()),
                all_grid_centre_inside=bool(grid_included.any()),pool_gt_area=float(area.max())))
        states.append(record)
    assert len({state['sequence'] for state in states}) == 20 and sum(state['outside'] for state in states) == 239
    summary = {}
    for scope in ('all','main_centre_outside'):
        subset = [state for state in states if scope=='all' or state['outside']]
        groups = dict(all=subset,shortside_below1=[state for state in subset if state['nominal_shortside_cells']<1],
                      shortside_1_to2=[state for state in subset if 1<=state['nominal_shortside_cells']<2],
                      shortside_at_least2=[state for state in subset if state['nominal_shortside_cells']>=2])
        summary[scope] = {key:summarize(value) for key,value in groups.items()}
    per_sequence = {name:summarize([state for state in states if state['sequence']==name])
                    for name in sorted({state['sequence'] for state in states})}
    with table.open('w',newline='',encoding='utf-8') as handle:
        writer = csv.DictWriter(handle,fieldnames=list(individual[0]))
        writer.writeheader()
        writer.writerows(individual)
    assert {name:sha(path) for name,path in paths.items()} == expected
    assert all(sha(root/name) == value for name,value in formal_plan['source_sha256'].items())
    report = dict(status='CPU_RETRIEVAL_LIMITS_COMPLETE',states=1752,train_sequences=20,executed_queries=31384,
        nn_calls=0,optimizer_updates=0,neural_weights_loaded=False,sealed_formal_sources_unchanged=len(formal_plan['source_sha256']),
        global_input_size=256,global_feature_grid=32,candidate_cap=8,default_nms_radius='main_search_side/4',
        fixed_state_scope='Old full-B primary-only train20 states with fitted preview ad059943. Not C20 dev or current E1 trajectories.',
        comparisons_scope='Changed NMS geometry uses saved logits and no new query encodings. Only default-pool query IoUs were actually executed. GT-best/STOP/grid reachability are diagnostic oracles, not deployment or trajectory bounds.',
        pixel_scope='Nominal short-side grid projection is not visible pixels, a CNN information bound or evidence that a higher preview resolution works.',
        cost_scope='Within a query family, executed choices use the same one-query encoder tokens; no new full-system latency or equal realized proposal counts are claimed.',
        per_scope=summary,per_sequence=per_sequence,input_sha256=expected,
        producer_sha256=sha(__file__),verified_analysis_sha256=sha(args.verified_analysis),prior_audit_sha256=sha(args.prior_audit),
        private_state_table_sha256=sha(table),independent_result_audit='PENDING')
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],states=1752,outside_states=239,
                         small_nominal_shortside_outside=summary['main_centre_outside']['shortside_below1']['states'],
                         default_candidate_reconstruction_exact=True,nn_calls=0)))


if __name__ == '__main__':
    main()

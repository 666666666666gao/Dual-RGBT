"""CPU-only verification of archived training profile records; stdlib only."""
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import struct

TRACE=Path(__file__).resolve().parent
ROOT=TRACE.parents[3]
hashes={}
checks=[]

def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

def load(name):
    path=ROOT/name
    hashes[str(path.relative_to(ROOT)).replace('\\','/')]=sha(path)
    return json.loads(path.read_bytes())

def check(name,value):
    checks.append(dict(check=name,passed=bool(value)))

def f32(value):
    return struct.unpack('f',struct.pack('f',value))[0]

def route_payload(rows):
    return json.dumps([dict(frame=r['frame'],box=r['box'],route={k:v for k,v in r['route'].items() if k!='seconds'}) for r in rows],sort_keys=True).encode()

plan=load('reports/TRAIN_CONCURRENCY_PLAN_20261026.json')
base=load(plan['training_plan'])
review=load(plan['source_review'])
launch=load('reports/TRAIN_CONCURRENCY_EXECUTION_20261026.json')
node=load('reports/TRAIN_CONCURRENCY_NODE_20261026.json')
grid=load('reports/TRAIN_CONCURRENCY_CANDIDATE_GRID_20261026.json')
formal_launch=load('reports/JOINT_GAINPAIR_TRAIN_EXECUTION_20261026.json')
formal_startup=load('reports/JOINT_GAINPAIR_TRAIN_STARTUP_20261026.json')
out=plan['output']
cfg=load(out+'/config.json')
done=load(out+'/COMPLETE.json')
native=load('runs/pipeline_tasks/'+plan['task']+'/status.json')
remote=load(str(TRACE.relative_to(ROOT)).replace('\\','/')+'/REMOTE_STATIC_BINDINGS.json')
check('plan_config_review_launch_hash',sha(ROOT/'reports/TRAIN_CONCURRENCY_PLAN_20261026.json')==cfg['plan_sha256']==review['plan_sha256']==launch['plan_sha256'])
check('source_review_launch_hash',sha(ROOT/plan['source_review'])==launch['source_review_sha256'])
check('config_plan_exact',cfg['plan']==plan)
check('review_source_binding',review['status']=='SOURCE_PASS' and review['blocking_issues']==0 and review['source_sha256']==plan['source_sha256'])
check('native_complete_exit0',native['state']=='COMPLETE' and native['exit_code']==0 and native['physical_gpu']=='1' and native['pid']==launch['pid'])
check('native_exact_command',native['command']==launch['command'][launch['command'].index('--')+1:])
check('native_node_binding',node['native']==native and {a['arm']:a for a in node['completed_arms']}=={a['arm']:a for a in done['arms']} and len(node['completed_arms'])==len(done['arms'])==5)
check('complete_binding',done['status']=='COMPLETE' and done['source_sha256']==plan['source_sha256'] and done['input_sha256']==plan['input_sha256'] and done['saved_checkpoints']==0)
check('inherited50_sources',len(base['source_sha256'])==50 and all(plan['source_sha256'].get(k)==v for k,v in base['source_sha256'].items()))
check('selected8_training_sequences',len(plan['sequence_names'])==len(set(plan['sequence_names']))==8 and set(plan['sequence_names']).issubset(base['sequence_names']))
check('existing_GPU1_formal_binding',formal_startup['native']['pid']==plan['coexisting_wrapper_pid']==formal_launch['formal']['pid'] and formal_startup['child_pid']==plan['coexisting_child_pid'])
check('existing_GPU1_training_recipe',formal_startup['config']['plan_sha256']==sha(ROOT/plan['training_plan']))
for group in ('source_sha256','input_sha256'):
    check('remote_map:'+group,set(remote[group])==set(plan[group]))
    for name,expected in plan[group].items():
        check('remote_binding:'+name,remote[group][name]==expected)
        if (ROOT/name).is_file():
            actual=sha(ROOT/name)
            hashes[name]=actual
            check('local_binding:'+name,actual==expected)
            if group=='source_sha256':ast.parse((ROOT/name).read_text(encoding='utf-8-sig'))
for name,expected in remote['artifacts_sha256'].items():
    check('finished_artifact_copy:'+name,sha(ROOT/name)==expected)
    hashes[name]=expected
hashes['tools/summarize_training_scaling.py']=sha(ROOT/'tools/summarize_training_scaling.py')
hashes['reports/TRAIN_CONCURRENCY_SOURCE_REVIEW_20261026.md']=sha(ROOT/'reports/TRAIN_CONCURRENCY_SOURCE_REVIEW_20261026.md')
log_path=ROOT/'runs/pipeline_tasks'/plan['task']/'run.log'
lines=log_path.read_text(encoding='utf-8').splitlines()
logs=[(i+1,json.loads(row)) for i,row in enumerate(lines) if row.startswith('{')]
arm_logs=[(i,row) for i,row in logs if 'arm' in row and 'workers' in row]
check('native_log5arms',len(arm_logs)==5 and [row for _,row in arm_logs]==done['arms'])
check('no_traceback_in_log',not any('Traceback (most recent call last)' in row for row in lines))
expected_changed=[]
for stem,indices in [('executor.preview.net',(0,2,4)),('executor.quality',(0,2)),('query_gain',(0,2))]:
    for i in indices:
        expected_changed.extend([stem+'.'+str(i)+'.weight',stem+'.'+str(i)+'.bias'])
train_meta={r['name']:r for r in remote['selected_train']}
references={}
summaries=[]
stream_audit=[]
all_pids=[]
loss_identity_max=0.
for ai,registered in enumerate(plan['arms']):
    name=registered['name']; directory=out+'/'+name
    arm=load(directory+'/summary.json')
    check(name+':all_summary_copies',arm==done['arms'][ai])
    n=registered['processes']
    check(name+':process_count_and_native_exits',arm['processes']==n and arm['child_exit_codes']==[0]*n)
    check(name+':worker_count',len(arm['workers'])==n)
    count=tracked=inits=0
    all_indices=[]
    stream_results=[]
    for wi,w in enumerate(arm['workers']):
        check(name+':worker_copy'+str(wi),load(directory+'/worker%d.json'%wi)==w)
        check(name+':worker_assignment'+str(wi),w['index']==wi and [s['index'] for s in w['streams']]==list(range(wi,8,n)))
        check(name+':freeze_receipt'+str(wi),w['state']=='RESULT' and w['frozen_state_tensors']==251 and w['changed_state_tensors']==expected_changed and w['total_parameters']==103218887)
        check(name+':worker_counts'+str(wi),w['updates']==32*len(w['streams']) and w['tracked_frames']==sum(s['tracked_frames'] for s in w['streams']) and w['initializes']==len(w['streams']))
        check(name+':positive_timing_memory'+str(wi),w['training_seconds']>0 and 0<w['peak_allocated_bytes']<=w['peak_reserved_bytes'])
        all_pids.append(w['pid'])
        for s in w['streams']:
            index=s['index'];all_indices.append(index)
            path=directory+'/stream%d_steps.json'%index
            rows=load(path)
            check(name+':stream%d_sha'%index,sha(ROOT/path)==s['steps_sha256'] and hashlib.sha256(route_payload(rows)).hexdigest()==s['prediction_route_sha256'])
            check(name+':stream%d_identity_work'%index,s['sequence']==plan['sequence_names'][index] and len(rows)==s['updates']==32 and [r['frame'] for r in rows]==train_meta[s['sequence']]['eligible_update_frames'] and rows[-1]['frame']==s['tracked_frames']==160)
            check(name+':stream%d_finite'%index,all(len(r['box'])==4 and min(r['box'][2:])>0 and all(math.isfinite(x) for x in r['box']) and len(r['losses'])==7 and all(math.isfinite(x) and x>=0 for x in r['losses']) and math.isfinite(r['route']['confidence']) and r['route']['seconds']>0 for r in rows))
            identities=[(r['losses'][0],f32(r['losses'][1]+r['losses'][2]),r['losses'][1],f32(r['losses'][3]+r['losses'][4]),r['losses'][2],f32(r['losses'][5]+r['losses'][6])) for r in rows]
            check(name+':stream%d_loss_FP32_identities'%index,all(a==b and c==d and e==f for a,b,c,d,e,f in identities))
            check(name+':stream%d_route_consistency'%index,all(r['route']['modality'] in (0,1) and r['route']['action'] in range(5) and r['route']['primary_tokens']==385 and r['route']['auxiliary_tokens']==(0 if r['route']['action']==0 else 385 if r['route']['action']==4 else 193) and r['route']['candidate_count']<=8 and (r['route']['request_index']==-1)==(r['route']['action']==0) for r in rows))
            if ai==0:references[index]=rows
            ref=references[index]
            discrete=lambda r:{k:v for k,v in r['route'].items() if k not in ('confidence','seconds')}
            cmp=dict(arm=name,stream=index,sequence=s['sequence'],raw_payload_sha256=s['steps_sha256'],prediction_route_sha256=s['prediction_route_sha256'],
                predictions_exact=all(r['box']==b['box'] for r,b in zip(rows,ref)),discrete_routes_exact=all(discrete(r)==discrete(b) for r,b in zip(rows,ref)),
                confidence_exact=all(r['route']['confidence']==b['route']['confidence'] for r,b in zip(rows,ref)),
                confidence_max_abs_delta=max(abs(r['route']['confidence']-b['route']['confidence']) for r,b in zip(rows,ref)),
                loss_values_exact=all(r['losses']==b['losses'] for r,b in zip(rows,ref)),
                loss_max_abs_delta=max(abs(x-y) for r,b in zip(rows,ref) for x,y in zip(r['losses'],b['losses'])),
                changed_confidence_rows=sum(r['route']['confidence']!=b['route']['confidence'] for r,b in zip(rows,ref)),
                changed_loss_rows=sum(r['losses']!=b['losses'] for r,b in zip(rows,ref)),
                update_frame_action_counts=dict(Counter(r['route']['action'] for r in rows)),
                pair_loss_nonzero_rows=sum(r['losses'][6]>0 for r in rows))
            stream_audit.append(cmp);stream_results.append(cmp)
            count+=len(rows);tracked+=rows[-1]['frame'];inits+=1
    check(name+':exact_fixed_work',sorted(all_indices)==list(range(8)) and count==arm['total_updates']==256 and tracked==1280 and inits==8)
    wall=arm['training_wall_seconds']
    check(name+':update_rate',arm['aggregate_updates_per_second']==count/wall)
    check(name+':track_rate',arm['aggregate_training_track_frames_per_second']==tracked/wall)
    release=max(w['training_finished_perf_counter'] for w in arm['workers'])-wall
    offsets=[w['training_finished_perf_counter']-w['training_seconds']-release for w in arm['workers']]
    check(name+':clock_ordering',min(offsets)>=0 and arm['training_and_validation_wall_seconds']>=wall and arm['spawn_to_join_seconds']>arm['training_and_validation_wall_seconds'])
    before,after=arm['coexisting_training_before'],arm['coexisting_training_after']
    interval=after['observed']-before['observed']
    check(name+':coexisting_observation_window',interval>wall and native['started']<before['observed']<after['observed']<native['completed'])
    background=[]
    for i,(first,last) in enumerate(zip(before['runs'],after['runs'])):
        check(name+':coexisting_identity'+str(i),first['task']==last['task']==plan['coexisting_runs'][i]['task'] and first['gpu']==last['gpu']==plan['coexisting_runs'][i]['gpu'] and last['logged_updates']>=first['logged_updates'])
        background.append(dict(gpu=first['gpu'],logged_increment=last['logged_updates']-first['logged_updates'],observation_seconds=interval,
            approximate_logged_updates_per_second=(last['logged_updates']-first['logged_updates'])/interval))
    nvml_before=list(map(int,arm['gpu_before_release_csv'].split(',')))
    nvml_after=list(map(int,arm['gpu_after_work_csv'].split(',')))
    check(name+':snapshot_fields',len(nvml_before)==len(nvml_after)==3 and nvml_before[1]==nvml_after[1]==24576 and 0<nvml_after[0]<24576 and 0<=nvml_after[2]<=100)
    summaries.append(dict(arm=name,additional_workers=n,total_GPU1_processes=n+1,updates=count,track_calls=tracked,initializations=inits,
        training_wall_seconds=wall,additional_updates_per_second=count/wall,training_track_frames_per_second=tracked/wall,
        validation_and_IPC_seconds=arm['training_and_validation_wall_seconds']-wall,
        spawn_to_join_seconds=arm['spawn_to_join_seconds'],spawn_to_join_updates_per_second=count/arm['spawn_to_join_seconds'],
        inferred_release_perf_counter=release,worker_start_offsets_seconds=offsets,
        worker_peak_allocated_bytes=[w['peak_allocated_bytes'] for w in arm['workers']],worker_peak_reserved_bytes=[w['peak_reserved_bytes'] for w in arm['workers']],
        resident_before_release_mib=nvml_before[0],resident_after_work_mib=nvml_after[0],total_mib=nvml_after[1],end_utilization_percent=nvml_after[2],
        background_logged_rates=background,all_stream_route_hashes_equal_first=all(x['predictions_exact'] and x['discrete_routes_exact'] and x['confidence_exact'] for x in stream_results),
        log_line=arm_logs[ai][0]))
check('16_distinct_workers',len(all_pids)==len(set(all_pids))==16)
check('1280_updates_6400_tracks_40_init',sum(a['updates'] for a in summaries)==1280 and sum(a['track_calls'] for a in summaries)==6400 and sum(a['initializations'] for a in summaries)==40)
check('no_saved_model_artifact',not any(p.suffix in ('.pth','.pt','.ckpt') for p in (ROOT/out).rglob('*')))
first,repeat=summaries[0]['additional_updates_per_second'],summaries[-1]['additional_updates_per_second']
published_summary=load('reports/TRAIN_CONCURRENCY_SUMMARY_20261026.json')
for calculated,reported in zip(summaries,published_summary['arms']):
    mapping={'arm':'arm','additional_workers':'additional_processes','total_GPU1_processes':'total_gpu1_processes',
        'updates':'optimizer_updates','training_wall_seconds':'training_wall_seconds',
        'additional_updates_per_second':'additional_aggregate_updates_per_second',
        'training_track_frames_per_second':'additional_training_track_frames_per_second',
        'spawn_to_join_seconds':'spawn_to_join_seconds','spawn_to_join_updates_per_second':'spawn_to_join_updates_per_second',
        'resident_after_work_mib':'gpu1_resident_after_work_mib','total_mib':'gpu1_total_mib',
        'end_utilization_percent':'gpu1_utilization_at_end_percent','worker_peak_allocated_bytes':'worker_peak_allocated_bytes',
        'worker_peak_reserved_bytes':'worker_peak_reserved_bytes','all_stream_route_hashes_equal_first':'all_stream_predictions_routes_equal_first'}
    check(calculated['arm']+':published_summary',all(calculated[k]==reported[v] for k,v in mapping.items()))
    check(calculated['arm']+':published_background_rates',all(c['gpu']==p['gpu'] and c['logged_increment']==p['logged_update_increment'] and c['approximate_logged_updates_per_second']==p['approximate_updates_per_second'] for c,p in zip(calculated['background_logged_rates'],reported['coexisting_formal_logged_rates'])))
check('published_summary_drift',published_summary['repeated_additional1_relative_drift_percent']==100*(repeat/first-1))
check('published_summary_best',published_summary['measured_highest_additional_throughput_arm']==max(summaries,key=lambda x:x['additional_updates_per_second'])['arm'])
report=dict(generated_at=datetime.now(timezone.utc).isoformat(),checks=checks,check_count=len(checks),failed=[x for x in checks if not x['passed']],
    plan_sha256=cfg['plan_sha256'],native=native,native_command_wall_seconds=native['completed']-native['started'],
    all_hashes=hashes,remote_source_count=len(remote['source_sha256']),remote_input_count=len(remote['input_sha256']),remote_finished_artifact_count=len(remote['artifacts_sha256']),
    arms=summaries,stream_comparisons=stream_audit,repeated_additional1_drift_percent=100*(repeat/first-1),
    additional2_over_both_additional1_baselines=[summaries[1]['additional_updates_per_second']/x for x in (first,repeat)],
    changed_confidence_streams=[dict(arm=x['arm'],stream=x['stream'],sequence=x['sequence'],changed_rows=x['changed_confidence_rows'],max_abs=x['confidence_max_abs_delta']) for x in stream_audit if not x['confidence_exact']],
    changed_loss_streams=[dict(arm=x['arm'],stream=x['stream'],sequence=x['sequence'],changed_rows=x['changed_loss_rows'],max_abs=x['loss_max_abs_delta']) for x in stream_audit if not x['loss_values_exact']],
    all_saved_update_boxes_exact=all(x['predictions_exact'] for x in stream_audit),all_saved_discrete_routes_exact=all(x['discrete_routes_exact'] for x in stream_audit),
    first_arm_update_frame_actions=dict(sum((Counter(x['update_frame_action_counts']) for x in stream_audit if x['arm']==plan['arms'][0]['name']),Counter())),
    gt_evidence_scope='Hash-bound TRAIN manifest provides current frame targets; raw teacher outputs/gradients/state tensors not archived by this profile.',
    clock_evidence_scope='Worker finish counters and elapsed durations retained; parent release counter inferred from saved wall duration,not separately archived. Spawn/join raw endpoints not retained.')
(TRACE/'RECOMPUTATION.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k not in ('checks','all_hashes','stream_comparisons')},indent=2))

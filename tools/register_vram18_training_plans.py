"""Register a finite cost/region grid for the user's four-card18GiB target."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    original=json.loads((ROOT/'reports/JOINT_CANDIDATE_EVALUATION_PROTOCOL_20261026.json').read_bytes())
    assert all(sha(ROOT/n)==h for n,h in original['source_sha256'].items())
    costs=[0.0,0.0125,0.025,0.05,0.075,0.1,0.15,0.2]
    extras=[]
    for objective in ['joint_quality','joint_gainpair']:
        for k in [4,8]:
            for cost in costs:
                if cost not in (0.0,0.05):extras.append((objective,k,cost))
    assert len(extras)==24
    placements={}
    for objective,g0,g1 in [('joint_quality',0,2),('joint_gainpair',1,3)]:
        for i,cost in enumerate([0.0125,0.025,0.075,0.1,0.15]):placements[(objective,8,cost)]=(g0,[4,6,6,8,8][i])
        for i,cost in enumerate([0.0125,0.025,0.075,0.1,0.15,0.2]):placements[(objective,4,cost)]=(g1,[4,4,4,6,6,8][i])
        placements[(objective,8,0.2)]=(g1,8)
    env=dict(OPENCV_FOR_THREADS_NUM='1',MALLOC_ARENA_MAX='2',MALLOC_MMAP_THRESHOLD_='131072',MALLOC_TRIM_THRESHOLD_='131072')
    new_tools=['tools/register_vram18_training_plans.py','tools/launch_vram18_training_wave.py','scripts/train_candidate_host_compact.py']
    grid_path='reports/VRAM18_TRAINING_GRID_20261027.json'
    grid=dict(status='PLANNED',objectives=['joint_quality','joint_gainpair'],candidate_counts=[4,8],trajectory_costs=costs,
        seed=2027,full_training_per_case=dict(sequences=882,rounds=2,updates=182182,tracked_frames=917106,initializes=1764),
        total_configurations=32,existing_configurations=8,new_configurations=24,target_tasks_per_gpu=8,target_resident_gib_approximately=18,
        rationale='Pre-registered interaction between candidate count and acquisition penalty changes causal training-state/query coverage. New cost values span near-zero, existing hinge/local-cost boundary and conservative STOP settings. No duplicate config, seed search, network layer/parameter/loss/LR/batch change. All original8 jobs untouched. Compact host allocator allowed only after paired actual numerical/memory proof. Staged4/6/8 percard with measured total throughput, original-job slowdown and RAM before expansion; no dummy occupancy.',
        allocator_environment=env,post_load_heap_release='One gc.collect+libc.malloc_trim(0), original trajectory executor unchanged.',evaluation='Full97LasHeRdev costs0/.05/.15, unchanged toolkit, audited freeze; eligible models same checkpoint/config across3fulltests. Failed fullDEV has no test. TotalP and serviceFPS included. Extended protocol is a separate sealed file; original8 protocol unchanged.')
    gp=ROOT/grid_path;assert not gp.exists();gp.write_text(json.dumps(grid,indent=2)+'\n')
    plans=[]
    for objective,k,cost in extras:
        base_path='reports/concurrent_'+objective+'_k4_cost0_plan_20261026.json'
        plan=json.loads((ROOT/base_path).read_bytes());gpu,phase=placements[(objective,k,cost)]
        tag=str(cost).replace('.','p');run=objective+'_k%d_cost%s_vram18_20261027'%(k,tag)
        path='reports/'+run+'_plan.json'
        plan.update(launch_trainer='scripts/train_candidate_host_compact.py',physical_gpu=gpu,cost_weight=cost,max_candidates=k,runs=dict(sanity=run+'_sanity',formal=run+'_formal'),tasks=dict(sanity='train_'+run+'_sanity',formal='train_'+run+'_formal'),
            source_review='reports/'+run+'_source_review.json',sanity_native_status='runs/pipeline_tasks/train_'+run+'_sanity/status.json',execution_env=env,
            registered_grid=grid_path,comparison_control=base_path,launch_wrapper='scripts/run_task_four_gpu.py',
            scope='User18GiB target finite32-grid: same architecture/three module duties/103218887P/seed2027/SOTep14/originalB6250/full882x2/182182updates/917106tracks/1764init. ONLY candidateK and trajectory acquisition-cost vary; host libc allocator settings separately validated for actual4step numerical equality and reducedPSS. No change to CNN/quality/gain loss, LR, batch, data order or first8 jobs. Real4step native0/frozen/replay/reload gate, formal reset originalB not M0. FullDEV/frozen3benchmarks before claims; no synthetic accuracy/parallelthroughput-as-onlineFPS. Stage4/6/8card admission requires actual ownership/RAM and profile proof; no preemption/restart.')
        for name in new_tools:plan['source_sha256'][name]=sha(ROOT/name)
        pp=ROOT/path;assert not pp.exists();pp.write_text(json.dumps(plan,indent=2)+'\n')
        plans.append(dict(path=path,sha256=sha(pp),phase=phase,gpu=gpu,objective=objective,max_candidates=k,cost_weight=cost))
    owners=json.loads((ROOT/'reports/FOUR_GPU_EIGHT_TRAIN_SIX_HOUR_NODE_PLAN_20261026.json').read_bytes())['tasks']
    index=dict(status='REGISTERED_SOURCE_REVIEW_PENDING',grid=grid_path,grid_sha256=sha(gp),plans=plans,original_owners=owners,
        source_review='reports/VRAM18_TRAINING_SOURCE_REVIEW_20261027.json',host_profile='reports/HOST_ALLOCATOR_PAIRED_TRIM_PROFILE_20261027.json',
        host_profile_plan='reports/HOST_ALLOCATOR_TRIM_PROFILE_PLAN_20261027.json',
        observer_source='tools/observe_vram18_training.py',observer_sha256=sha(ROOT/'tools/observe_vram18_training.py'),
        baseline_previous='reports/FOUR_GPU_EIGHT_TRAIN_FIRST_NODE_20261026.json',baseline_previous_sha256=sha(ROOT/'reports/FOUR_GPU_EIGHT_TRAIN_FIRST_NODE_20261026.json'),
        baseline_reference='reports/VRAM18_PRE_EXPANSION_BASELINE_20261027.json',baseline_reference_sha256=sha(ROOT/'reports/VRAM18_PRE_EXPANSION_BASELINE_20261027.json'),
        scope='Registration only, no new formal launch. Phase4 adds1/1/3/3, phase6 adds2each, phase8 adds2each; total32distinctconfigs, no original active file edits.')
    ip=ROOT/'reports/VRAM18_TRAINING_PLAN_INDEX_20261027.json';assert not ip.exists();ip.write_text(json.dumps(index,indent=2)+'\n')
    spec=dict(original);spec['training_plan_sha256']=dict(original['training_plan_sha256']);spec['source_sha256']=dict(original['source_sha256'])
    for name in new_tools:spec['source_sha256'][name]=sha(ROOT/name)
    for row in plans:spec['training_plan_sha256'][row['path']]=row['sha256']
    assert len(spec['training_plan_sha256'])==32
    spec['source_review']='reports/VRAM18_EVALUATION_SOURCE_REVIEW_20261027.json'
    spec['scope']=original['scope'].replace('eight registered','32registered')+' Expanded32grid retains same source/metric/runtime checks. Initial8 protocol unchanged. Allocation environment change has separate actual native memory/numeric admission; report native timings at actual load, no baseline speedup claim.'
    ep=ROOT/'reports/VRAM18_EVALUATION_PROTOCOL_20261027.json';assert not ep.exists();ep.write_text(json.dumps(spec,indent=2)+'\n')
    print(json.dumps(dict(status='REGISTERED_SOURCE_ONLY',new_plans=len(plans),models=32,source_files=len(spec['source_sha256']),index_sha256=sha(ip),evaluation_protocol_sha256=sha(ep))),flush=True)
if __name__=='__main__':main()

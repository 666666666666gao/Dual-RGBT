"""Assign only the eight unlaunched registered configurations to additional25."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    original_path = 'reports/VRAM18_TRAINING_PLAN_INDEX_20261027.json'
    original = json.loads((ROOT / original_path).read_bytes())
    copied = json.loads((ROOT / 'reports/SERVER25_REJOIN_RUNTIME_COPY_20261027.json').read_bytes())
    accepted_path = 'reports/SERVER25_ENVIRONMENT_ACCEPTANCE_20261027.json'
    accepted = json.loads((ROOT / accepted_path).read_bytes())
    assert accepted['status'] == 'ENVIRONMENT_ACCEPTED'
    unused_path = 'reports/SERVER26_UNLAUNCHED_GRID_REASSIGNMENT_20261027.json'
    unused = json.loads((ROOT / unused_path).read_bytes())
    assert unused['all_eight_absent'] and len(unused['tasks']) == 8
    plans = []
    for row in original['plans']:
        if row['phase'] != 8:
            continue
        template = 'reports/concurrent_' + row['objective'] + '_k4_cost0_plan_20261026.json'
        plan = json.loads((ROOT / template).read_bytes())
        tag = str(row['cost_weight']).replace('.', 'p')
        stem = row['objective'] + '_k%d_cost%s_server25_20261027' % (row['max_candidates'], tag)
        path = 'reports/' + stem + '_plan.json'
        plan.update(status='READY', physical_gpu=row['gpu'], cost_weight=row['cost_weight'],
                    max_candidates=row['max_candidates'],
                    runs=dict(sanity=stem+'_sanity', formal=stem+'_formal'),
                    tasks=dict(sanity='train_'+stem+'_sanity', formal='train_'+stem+'_formal'),
                    source_review='reports/'+stem+'_source_review.json',
                    sanity_native_status='runs/pipeline_tasks/train_'+stem+'_sanity/status.json',
                    execution_env=dict(OPENCV_FOR_THREADS_NUM='1'),
                    launch_wrapper='scripts/run_task_four_gpu.py',
                    launch_trainer='scripts/train_candidate_'+row['objective']+'_trajectory.py',
                    registered_grid='reports/SERVER25_REMAINING_GRID_20261027.json',
                    environment_acceptance=accepted_path,
                    scope='Additional server25, original registered but unlaunched configuration. Same architecture/three module duties/103218887 totalP/SOTep14/B6250/seed2027/full882x2/182182updates. Original native trainer and default allocator; no heap-trim wrapper, no changed loss/LR/batch/data order. Only provider paths, run identity and registered K/cost differ. Real M0 native0/frozen/reload admission; formal resets originalB, not M0. FullDEV and frozen3benchmarks still required; no accuracy/onlineFPS claim.')
        assert plan['launch_trainer'] in plan['source_sha256']
        for name in ['tools/server25_runtime.py', 'tools/register_server25_training.py',
                     'tools/launch_server25_training.py']:
            plan['source_sha256'][name] = sha(ROOT / name)
        plan['input_sha256'] = dict(copied['input_sha256'])
        plan['input_sha256'][accepted_path] = sha(ROOT / accepted_path)
        target = ROOT / path
        assert not target.exists()
        target.write_text(json.dumps(plan, indent=2) + '\n')
        plans.append(dict(path=path, sha256=sha(target), original_unlaunched_plan=row['path'],
                          original_unlaunched_sha256=row['sha256'], gpu=row['gpu']))
    assert len(plans) == 8
    index = dict(status='REGISTERED_PENDING_SOURCE_REVIEW', plans=plans,
                 original_index_sha256=sha(ROOT / original_path),
                 source_review='reports/SERVER25_TRAINING_SOURCE_REVIEW_20261027.json',
                 environment_acceptance=accepted_path, total_parameters=103218887,
                 scope='Eight additional25 jobs, two per GPU. The24 existing26 jobs are untouched. Reassignments use new plans; original26 plans/source remain immutable.')
    path = ROOT / 'reports/SERVER25_REMAINING_GRID_20261027.json'
    assert not path.exists()
    path.write_text(json.dumps(index, indent=2) + '\n')
    print(json.dumps(dict(status='REGISTERED', plans=len(plans), index_sha256=sha(path))))


if __name__ == '__main__':
    main()

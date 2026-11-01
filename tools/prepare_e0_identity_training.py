"""Create immutable M2 update-smoke/full-training proposals for source review."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from benchmark_inputs import sha

FOLDER = 'reports/e0_m2_training_p25_r2_20261101'


def main():
    collection_relative = (
        'reports/e0_m2_identity_p25_r1_20261101/runtime/'
        'E0_M2_IDENTITY_FULL_PLAN_P25_R1_20261101.json')
    collection_path = ROOT / collection_relative
    collection = json.loads(collection_path.read_bytes())
    sources = dict(collection['source_sha256'])
    for relative in ('scripts/train_e0_identity.py', 'tools/verify_e0_identity_collection.py',
                     'tests/test_train_e0_identity.py'):
        sources[relative] = sha(ROOT / relative)
    folder = ROOT / FOLDER
    folder.mkdir(exist_ok=False)
    for phase in ('smoke', 'full'):
        relative = FOLDER + '/%s_draft.json' % phase
        plan = dict(mode='protected_e0_offline_m2_identity_training', phase=phase,
            launch_allowed=False, split='train', test_eligible=False,
            scientific_result_audit_accepted=False, target_host='2025',
            target_root='/data2/gb/Dual-RGBT', device='cuda', assigned_gpu='2',
            task='train_protected_e0_m2_identity_%s_r2_p25_20261101' % phase,
            output='runs/protected_e0_offline_m2_identity_%s_r2_p25_20261101' % phase,
            reviewed_draft=relative,
            source_review='reports/E0_M2_IDENTITY_TRAINING_SOURCE_REVIEW_P25_R2_20261101.json',
            source_review_sha256=None, source_sha256=sources,
            collection_plan='reports/E0_M2_IDENTITY_FULL_PLAN_P25_R1_20261101.json',
            collection_plan_sha256=sha(collection_path),
            collection_acceptance='reports/E0_M2_IDENTITY_FULL_COLLECTION_ACCEPTANCE_P25_20261101.json',
            collection_acceptance_sha256=None,
            checkpoint_sha256=collection['checkpoint_sha256'],
            epochs=0 if phase == 'smoke' else 12,
            batch_sets=6 if phase == 'smoke' else 512,
            lr=3e-4, weight_decay=1e-4, seed=2027,
            update_smoke_task='train_protected_e0_m2_identity_smoke_r2_p25_20261101',
            update_smoke_output='runs/protected_e0_offline_m2_identity_smoke_r2_p25_20261101',
            update_smoke_plan=FOLDER + '/smoke_plan.json',
            update_smoke_plan_sha256=None, update_smoke_receipt_sha256=None,
            update_smoke_native_sha256=None,
            loss='logsumexp known candidates minus logsumexp all positive candidates; '
                 'unknown labels excluded, sets without positives excluded from updates',
            selection='TRAIN component holdout top1 positive fraction, upstream E0 saw holdout',
            retention='best/latest and last five numbered epochs',
            scope='Standalone fixed-anchor M2 candidate matching; no E0/M1/M3 update, '
                  'online recovery, benchmark result or independent generalization. '
                  'Smoke five updates on six real TRAIN prefix sets only. '
                  'Full requires all eight collection shards and real update smoke.')
        (ROOT / relative).write_text(json.dumps(plan, indent=2) + '\n')
    print(json.dumps({p.relative_to(ROOT).as_posix(): sha(p) for p in folder.glob('*.json')},
                     indent=2))


if __name__ == '__main__':
    main()

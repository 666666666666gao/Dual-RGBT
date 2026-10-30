"""Register separate complete exploratory-test jobs for retained checkpoints."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from benchmark_inputs import sha

REVIEW = 'reports/CACHED_GAIN_TEST_SOURCE_REVIEW_20261030.json'
MAPPING = 'reports/CACHED_PARENT_LOSS_SERVER26_TEST_MAPPING_20261030.json'
QUEUE = 'tools/queue_complete_test.py'
OUTPUT = 'reports/CACHED_GAIN_FULL_TEST_QUEUE_SERVER26_20261030.json'
FORMAL = 'runs/cached_parent_gain_all882_formal_20261030/COMPLETE.json'
NATIVE = 'runs/pipeline_tasks/train_cached_parent_gain_all882_formal_20261030/status.json'


def main():
    review_path, mapping_path = ROOT / REVIEW, ROOT / MAPPING
    review = json.loads(review_path.read_text())
    mapping = json.loads(mapping_path.read_text())
    assert review['status'] == 'SOURCE_PASS' and review['blocking_issues'] == 0
    assert mapping['status'] == 'TEST_MAPPING_VERIFIED'
    assert (mapping['sequences'], mapping['frames'], mapping['checked_image_paths']) == (
        245, 220703, 441406)
    training = json.loads((ROOT / FORMAL).read_text())
    native_path = ROOT / NATIVE
    native = json.loads(native_path.read_text())
    assert training['status'] == 'TRAIN_COMPLETE_PENDING_DEV'
    assert native['state'] == 'COMPLETE' and native['exit_code'] == 0
    source_names = list(review['source_sha256']) + [QUEUE, 'tools/prepare_cached_gain_full_test.py']
    source_hashes = {name: sha(ROOT / name) for name in source_names}
    source_hashes['tools/queue_complete_test.py'] = source_hashes.pop(QUEUE)
    source_hashes['tools/prepare_cached_gain_full_test.py'] = sha(
        ROOT / 'tools/prepare_cached_gain_full_test.py')
    jobs = []
    for checkpoint in training['retained']:
        epoch = checkpoint['epoch']
        base_path = ROOT / ('reports/cached_parent_gain_all882_e%03d_eval_smoke_20261030_plan.json' % epoch)
        plan = json.loads(base_path.read_text())
        assert plan['checkpoint_sha256'] == checkpoint['sha256']
        assert plan['training_result_sha256'] == sha(ROOT / FORMAL)
        assert plan['training_native_sha256'] == sha(native_path)
        plan['source_sha256'] = source_hashes
        plan['source_review'] = REVIEW
        plan['source_review_sha256'] = sha(review_path)
        plan.update(full_evaluation_allowed=True, exploratory_test=True,
                    runtime_provider=dict(ssh_alias='2026',
                                          project='/data2/gb/Dual-RGBT/snapshots/cached_parent_loss_20261030'),
                    dataset_mapping_receipt=MAPPING,
                    dataset_mapping_receipt_sha256=sha(mapping_path))
        run = 'cached_parent_gain_all882_e%03d_lasher_test_exploratory_20261030' % epoch
        plan_name = 'reports/' + run + '_plan.json'
        plan_path = ROOT / plan_name
        assert not plan_path.exists()
        plan_path.write_text(json.dumps(plan, indent=2) + '\n')
        jobs.append(dict(epoch=epoch, run=run, script='tools/run_cached_gain_server26_test.py',
                         plan=plan_name, plan_sha256=sha(plan_path),
                         checkpoint=checkpoint['checkpoint'],
                         checkpoint_sha256=checkpoint['sha256'],
                         score='reports/cached_parent_gain_all882_test_20261030/e%03d.json' % epoch))
    manifest = dict(status_file=OUTPUT, host='2026', split='LasHeR-test',
                    exploratory_test=True, test_eligible=False,
                    sequences=245, frames=220703, jobs=jobs,
                    source_sha256={name: sha(ROOT / name) for name in
                                   ('tools/prepare_cached_gain_full_test.py', QUEUE)})
    output_path = ROOT / OUTPUT
    assert not output_path.exists()
    output_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(status='PLANS_REGISTERED', jobs=len(jobs),
                          manifest_sha256=sha(output_path),
                          run_names=[job['run'] for job in jobs])))


if __name__ == '__main__':
    main()

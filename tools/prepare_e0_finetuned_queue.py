"""Prepare the requested final-five, three-benchmark E0 evaluation queue."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = 'reports/E0_EP20_FINAL_FIVE_EVALUATION_QUEUE_P25_20261101.json'


def main():
    preflight = json.loads((ROOT / 'reports/E0_FINETUNED_EVALUATION_PREFLIGHT_20261101.json').read_bytes())
    root = '/home/gaob/Dual-RGBT'
    review = 'reports/E0_FINETUNED_EVALUATION_FINAL_SOURCE_REVIEW_20261101.json'
    code = """
import hashlib,json
from pathlib import Path
r=Path('/home/gaob/Dual-RGBT')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
p=r/'reports/E0_FINETUNED_EVALUATION_FINAL_SOURCE_REVIEW_20261101.json'
review=json.loads(p.read_bytes())
assert sha(p)=='bbcd785dc68a12529e7690dfa4b901abd03862a5b3ad15d49307f6b6a80734fd'
assert review['stage']=='E0_finetuned_evaluation'
assert review['status']=='SOURCE_PASS' and review['blocking_issues']==0
assert all(sha(r/n)==h for n,h in review['source_sha256'].items())
config=r/'runs/pipeline_tasks/train_e0_executor_ep20_r1_20261101/status.json'
native=json.loads(config.read_bytes())
assert native['task']=='train_e0_executor_ep20_r1_20261101'
run=Path('/data2/gaob/Dual-RGBT/e0_executor_ep20_lasher_r1_20261101')
training=json.loads((run/'config.json').read_bytes())
assert training['stage']=='E0_executor_ep20_finetune'
assert training['epochs']==40 and training['policy_frozen'] is True
assert sha(r/'scripts/finetune_e0_ep20.py')==training['hashes']['entry']
print(json.dumps(dict(review_sha256=sha(p),source_sha256=review['source_sha256'],
 training_config_sha256=sha(run/'config.json'),training_entry_sha256=training['hashes']['entry'],
 factory=training['cadtrack'],factory_sha256=training['hashes']['cadtrack'],
 parent_sha256=training['hashes']['parent'])))
"""
    result = subprocess.run(
        ['ssh', '-o', 'ClearAllForwardings=yes', '-o', 'BatchMode=yes', '2025',
         root + '/conda-envs/dual_rgbt/bin/python -'],
        input=code.encode(), stdout=subprocess.PIPE, check=True)
    actual = json.loads(result.stdout)
    assert len(actual['source_sha256']) == 49
    assert actual['factory_sha256'] == preflight['factory_sha256']
    jobs = []
    for epoch in preflight['retained_epochs']:
        for dataset in preflight['read_only_remote_checks']['datasets']:
            name = dataset['benchmark'].lower()
            jobs.append(dict(epoch=epoch, benchmark=dataset['benchmark'],
                task='eval_e0_ep20_e%03d_%s_r1_p25_20261101' % (epoch, name),
                run='/data2/gaob/Dual-RGBT/e0_ep20_e%03d_%s_r1_p25_20261101' % (epoch, name),
                checkpoint=preflight['formal_training_output'] + '/epoch_%03d.pth' % epoch,
                manifest=root + '/' + dataset['manifest'],
                manifest_sha256=dataset['manifest_sha256'],
                sequences=dataset['sequences'], frames=dataset['frames'],
                score_output=root + '/reports/e0_ep20_e%03d_%s_full_r1_p25_20261101.json' % (epoch, name)))
    queue = dict(stage='E0_ep20_final_five_evaluation_queue', target_host='2025',
        target_root=root, status='WAITING_FOR_RETAINED_CHECKPOINTS',
        retained_epochs=preflight['retained_epochs'], jobs=jobs,
        training_task=preflight['training_task'],
        training_output=preflight['formal_training_output'],
        source_review=root + '/' + review, cost_weight=.05,
        exploratory_test=True, test_eligible=False,
        scientific_result_audit_accepted=False, **actual,
        scope='Fifteen complete exploratory evaluations of retained E0 epochs36-40. '
              'Not launched by queue preparation. Require matching EPOCH_COMPLETE '
              'before loading each file, actual checkpoint hash and native capacity. '
              'All retained checkpoints remain audit/evaluation dependencies. '
              'Report every result; TEST does not select training or final validation.')
    assert len(jobs) == 15
    path = ROOT / OUTPUT
    assert not path.exists()
    path.write_text(json.dumps(queue, indent=2) + '\n')
    print(json.dumps(dict(queue=OUTPUT, jobs=len(jobs),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        review_source_entries=len(actual['source_sha256'])), indent=2))


if __name__ == '__main__':
    main()

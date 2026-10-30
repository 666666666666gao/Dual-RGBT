"""Compare the complete dev selections without implying test improvement."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    rows = []
    for variant, directory, epochs in (
        ('parent_loss', 'cached_query_gain_parent_loss_dev_20261030', range(5, 10)),
        ('utility_focal', 'cached_query_gain_dev_20261030', range(6, 11)),
    ):
        for epoch in epochs:
            path = ROOT / 'reports' / directory / ('e%d.json' % epoch)
            report = json.loads(path.read_text())
            assert report['status'] == 'SCORED_COMPLETE'
            assert (report['split'], report['sequences'], report['frames']) == ('dev', 97, 54646)
            assert report['test_eligible'] is False
            assert sha(path.with_suffix('.npz')) == report['curves_sha256']
            rows.append(dict(variant=variant, epoch=epoch, report=path.relative_to(ROOT).as_posix(),
                             report_sha256=sha(path), native_status_sha256=report['native_status_sha256'],
                             verified_run_artifacts=len(report['run_artifacts_sha256']),
                             **{key: report[key] for key in ('PR', 'NPR', 'SR', 'fps', 'checkpoint_sha256')}))
    best = {}
    for variant in ('parent_loss', 'utility_focal'):
        best[variant] = max((row for row in rows if row['variant'] == variant),
                            key=lambda row: (row['SR'], row['PR'], row['NPR']))
    parent_sr = 56.04716073871899
    result = dict(status='COMPLETE_DEV_COMPARISON_NO_PROMOTION', rows=rows, best=best,
                  parent_SR=parent_sr,
                  parent_loss_best_minus_focal_SR_pp=best['parent_loss']['SR'] - best['utility_focal']['SR'],
                  parent_loss_best_minus_parent_SR_pp=best['parent_loss']['SR'] - parent_sr,
                  full_dev_sequences=97, full_dev_frames=54646, total_parameters=103218887,
                  test_eligible=False, promoted=False, retained_weights='Keep last five of each training run',
                  remote_verification='All five parent-loss native/config/392 run artifact/curve bindings verified on originating server in this continuation before retrieval. Local curve bindings rechecked here.',
                  limitations=[
                      'Same100TRAIN cache, seed2027, batch2866, LR0.00112 and stopping rule; realized epochs differ: parent9/18updates versus focal10/20updates.',
                      'Component holdout was seen by parent training; this is not unseen-system generalization.',
                      'Dev selects settings and weights. No full-test improvement is claimed.',
                      'Concurrent FPS differs by server and workload; it is not an isolated speed comparison.',
                  ])
    output = ROOT / 'reports/CACHED_LOSS_COMPLETE_DEV_COMPARISON_20261030.json'
    assert not output.exists()
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(best=best, parent_loss_best_minus_focal_SR_pp=result['parent_loss_best_minus_focal_SR_pp'],
                          parent_loss_best_minus_parent_SR_pp=result['parent_loss_best_minus_parent_SR_pp']), indent=2))


if __name__ == '__main__':
    main()

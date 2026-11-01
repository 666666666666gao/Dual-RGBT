"""Summarize the live E0 TRAIN CSV by epoch and executed route."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import time

import numpy as np


def digest(value):
    return hashlib.sha256(value).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--task-directory', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run, task, output = Path(args.run), Path(args.task_directory), Path(args.output)
    assert not output.exists()
    config_bytes = (run / 'config.json').read_bytes()
    config = json.loads(config_bytes)
    assert config['stage'] == 'E0_executor_ep20_finetune'
    assert config['policy_frozen'] is True and config['dev_enabled'] is config['test_enabled'] is False
    native = json.loads((task / 'status.json').read_bytes())
    csv_bytes = (run / 'steps.csv').read_bytes()
    rows = list(csv.DictReader(io.StringIO(csv_bytes.decode())))
    assert [int(row['step']) for row in rows] == list(range(len(rows)))
    log_bytes = (task / 'run.log').read_bytes()
    completed = {int(epoch): float(seconds) for epoch, seconds in
                 re.findall(r'EPOCH_COMPLETE\s+(\d+)\s+([0-9.eE+-]+)', log_bytes.decode())}
    steps_per_epoch = config['samples'] // config['batch_size']
    epochs = []
    for epoch in sorted({int(row['epoch']) for row in rows}):
        selected = [row for row in rows if int(row['epoch']) == epoch]
        loss = np.asarray([float(row['loss']) for row in selected])
        seconds = np.asarray([float(row['seconds']) for row in selected])
        assert np.isfinite(loss).all() and np.isfinite(seconds).all() and (seconds > 0).all()
        routes = []
        for modality in range(2):
            for action in range(5):
                values = [float(row['loss']) for row in selected
                          if int(row['modality']) == modality and int(row['action']) == action]
                assert values
                routes.append(dict(modality=modality, action=action, updates=len(values),
                                   mean_loss=float(np.mean(values))))
        assert sum(row['updates'] for row in routes) == len(selected)
        assert len(selected) <= steps_per_epoch
        if epoch in completed:
            assert len(selected) == steps_per_epoch
        decay = sum(epoch > milestone for milestone in config['milestones'])
        epochs.append(dict(epoch=epoch, updates=len(selected),
            samples=len(selected) * config['batch_size'],
            epoch_complete=epoch in completed,
            epoch_wall_seconds=completed.get(epoch),
            executor_lr=config['lr'] * config['gamma'] ** decay,
            backbone_lr=config['lr'] * .1 * config['gamma'] ** decay,
            mean_loss=float(loss.mean()), median_loss=float(np.median(loss)),
            loss_p90=float(np.quantile(loss, .9)),
            mean_logged_step_seconds=float(seconds.mean()), routes=routes))
    durations = list(completed.values())
    assert durations
    report = dict(status='E0_TRAIN_PROGRESS_SNAPSHOT',
        utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        run=str(run), task_directory=str(task), native_state=native['state'],
        wrapper_pid=native['pid'], epochs=epochs,
        completed_epochs=sorted(completed), total_logged_updates=len(rows),
        mean_completed_epoch_minutes=float(np.mean(durations)) / 60,
        planned_epochs=config['epochs'], retained_epochs=list(range(config['epochs'] - 4, config['epochs'] + 1)),
        estimated_remaining_hours_at_completed_epoch_mean=(
            (config['epochs'] - len(rows) / steps_per_epoch) * float(np.mean(durations)) / 3600),
        convergence_established=False, test_eligible=False,
        scientific_result_audit_accepted=False,
        snapshot_sha256={'config.json': digest(config_bytes), 'steps.csv': digest(csv_bytes),
                         'native_status.json': digest((task / 'status.json').read_bytes()),
                         'run.log': digest(log_bytes)},
        scope='TRAIN loss and elapsed-time diagnosis only. Each epoch uses different '
              'seeded pairs and all ten modality/action routes. Non-STOP objectives '
              'contain correction supervision and must not be compared as tracking '
              'accuracy. Step seconds exclude loader waiting and are not FPS. '
              'Runtime extrapolation is not convergence or held-out tracking evidence.')
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ('epochs', 'snapshot_sha256')}, indent=2), flush=True)
    for epoch in epochs:
        print(json.dumps({key: value for key, value in epoch.items() if key != 'routes'}), flush=True)


if __name__ == '__main__':
    main()

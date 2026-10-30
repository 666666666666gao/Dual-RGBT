"""Train and select size-tail epochs, retaining the last five scored weights."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from benchmark_inputs import sha
from joint_evaluation_contract import check_files


def stopping_epoch(scores, min_epochs=5, patience=3, min_delta=0.1):
    """Significant progress is measured against the previous progress anchor."""
    anchor, stale = None, 0
    for epoch, score in enumerate(scores, 1):
        if anchor is None or score > anchor + min_delta:
            anchor, stale = score, 0
        else:
            stale += 1
        if epoch >= min_epochs and stale >= patience:
            return epoch
    return None


def best_recent(rows):
    return max(rows[-5:], key=lambda row: (row['metrics']['SR'], row['metrics']['PR'], row['metrics']['NPR']))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--gpu', choices=('0', '1', '2', '3'), required=True)
    args = parser.parse_args()
    assert Path(args.name).name == args.name
    plan_path = ROOT / args.plan
    plan = json.loads(plan_path.read_text())
    assert plan['formal_launch_allowed'] is True and 5 <= plan['epochs'] <= 10
    check_files(ROOT, plan['source_sha256'])
    assert sha(ROOT / plan['parent_control_score']) == plan['parent_control_score_sha256']
    assert sha(ROOT / plan['diagnostic_score']) == plan['diagnostic_score_sha256']
    for path in [plan['parent_control_score'], plan['diagnostic_score']]:
        report = json.loads((ROOT / path).read_text())
        assert report['status'] == 'SCORED_COMPLETE' and (report['sequences'], report['frames']) == (97, 54646)
    directory = ROOT / 'runs' / args.name
    directory.mkdir(parents=True, exist_ok=False)
    with (directory / 'controller.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = dict(status='RUNNING', controller_pid=os.getpid(), plan_sha256=sha(plan_path),
            physical_gpu=args.gpu, poll_seconds=180, rows=[])
        python = str(ROOT / 'conda-envs/dual_rgbt/bin/python')
        cpu = str(ROOT / 'conda-envs/cpu-scoring/bin/python')
        env = dict(os.environ, OPENCV_FOR_THREADS_NUM='1', PYTHONUNBUFFERED='1')

        def save():
            state['updated'] = time.time()
            temporary = directory / 'status.tmp'
            temporary.write_text(json.dumps(state, indent=2) + '\n')
            temporary.replace(directory / 'status.json')

        def run(task, command):
            assert not (ROOT / 'runs/pipeline_tasks' / task).exists()
            wrapper = [python, str(ROOT / 'scripts/run_task_four_gpu.py'), '--task', task, '--gpu', args.gpu, '--'] + command
            with (directory / (task + '.wrapper.log')).open('wb') as handle:
                process = subprocess.Popen(wrapper, cwd=str(ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT)
            state.update(active_task=task, active_pid=process.pid)
            save()
            while process.poll() is None:
                state['next_check_epoch'] = time.time() + 180
                save()
                time.sleep(180)
            native = json.loads((ROOT / 'runs/pipeline_tasks' / task / 'status.json').read_text())
            assert process.returncode == 0 and native['state'] == 'COMPLETE' and native['exit_code'] == 0
            assert native['command'] == command

        def remove_checkpoint(row):
            path = ROOT / row['checkpoint']
            expected = ROOT / 'runs' / (args.name + '_train_e%03d' % row['epoch']) / ('epoch_%03d.pth' % row['epoch'])
            assert path == expected and sha(path) == row['checkpoint_sha256']
            path.unlink()
            row['checkpoint_retained'] = False

        save()
        for epoch in range(1, plan['epochs'] + 1):
            state.update(epoch=epoch, phase='WAITING_FOR_TRAIN_MEMORY')
            save()
            while True:
                used = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], universal_newlines=True)
                if int(used.splitlines()[int(args.gpu)]) <= plan['max_existing_gpu_mib']:
                    break
                state['next_check_epoch'] = time.time() + 180
                save()
                time.sleep(180)
            train_run = args.name + '_train_e%03d' % epoch
            task = 'train_' + train_run
            command = [python, str(ROOT / 'scripts/train_size_tail_epoch.py'), '--plan', args.plan,
                '--run', train_run, '--task', task, '--epoch', str(epoch)]
            if state['rows']:
                previous = state['rows'][-1]
                command += ['--resume', previous['checkpoint'], '--resume-sha256', previous['checkpoint_sha256']]
            state['phase'] = 'TRAIN'
            run(task, command)
            training = json.loads((ROOT / 'runs' / train_run / 'COMPLETE.json').read_text())
            checkpoint = 'runs/' + train_run + '/epoch_%03d.pth' % epoch
            assert training['strict_reload_bit_equal'] and sha(ROOT / checkpoint) == training['checkpoint_sha256']
            evaluation = json.loads((ROOT / plan['evaluation_template']).read_text())
            evaluation.update(mode='size_tail_diagnostic', checkpoint=checkpoint,
                checkpoint_sha256=training['checkpoint_sha256'], training_task=task,
                training_scope='One completed epoch in a resumable dev-selected training chain')
            evaluation_path = directory / ('eval_epoch_%03d.json' % epoch)
            evaluation_path.write_text(json.dumps(evaluation, indent=2) + '\n')
            eval_run = args.name + '_dev_e%03d' % epoch
            task = 'eval_' + eval_run
            state['phase'] = 'DEV'
            run(task, [python, str(ROOT / 'scripts/eval_size_tail.py'), '--plan', evaluation_path.relative_to(ROOT).as_posix(),
                '--run', eval_run, '--task', task, '--smoke-frames', '0'])
            score_path = directory / ('score_epoch_%03d.json' % epoch)
            state['phase'] = 'SCORE'
            save()
            with (directory / ('score_epoch_%03d.log' % epoch)).open('wb') as handle:
                subprocess.run([cpu, str(ROOT / 'tools/score_size_tail.py'), '--run', eval_run,
                    '--output', score_path.relative_to(ROOT).as_posix()], cwd=str(ROOT), env=env,
                    stdout=handle, stderr=subprocess.STDOUT, check=True)
            metrics = json.loads(score_path.read_text())
            assert metrics['status'] == 'SCORED_COMPLETE' and (metrics['sequences'], metrics['frames']) == (97, 54646)
            row = dict(epoch=epoch, checkpoint=checkpoint, checkpoint_sha256=training['checkpoint_sha256'],
                checkpoint_retained=True, score_path=score_path.relative_to(ROOT).as_posix(), score_sha256=sha(score_path),
                metrics={k: metrics[k] for k in ('PR', 'NPR', 'SR', 'fps')}, training=training)
            state['rows'].append(row)
            if len(state['rows']) > 5:
                remove_checkpoint(state['rows'][-6])
            save()
            if metrics['SR'] < plan['parent_metrics']['SR'] - plan['max_sr_drop_pp']:
                state.update(status='REJECTED_REGRESSION', phase='STOPPED',
                    stop_reason='Full-dev SR fell below the predeclared parent regression floor')
                save()
                (directory / 'REJECTION.json').write_text(json.dumps(dict(epoch=epoch, score=row,
                    parent_metrics=plan['parent_metrics'], max_sr_drop_pp=plan['max_sr_drop_pp'],
                    plan_sha256=state['plan_sha256']), indent=2) + '\n')
                return
            if stopping_epoch([row['metrics']['SR'] for row in state['rows']]) is not None:
                state['stop_reason'] = 'Three epochs without SR improvement greater than0.1pp; at least five observed'
                break
        check_files(ROOT, plan['source_sha256'])
        assert sha(plan_path) == state['plan_sha256']
        selected = best_recent(state['rows'])
        selection = dict(status='DEV_SELECTED', test_eligible=False, selected=selected,
            evaluated_epochs=[r['epoch'] for r in state['rows']], selection_window=[r['epoch'] for r in state['rows'][-5:]],
            ranking=['SR', 'PR', 'NPR'], parent_metrics=plan['parent_metrics'], protected_e1_metrics=plan['protected_e1_metrics'],
            stop_reason=state.get('stop_reason', 'Maximum planned epochs reached'), plan_sha256=sha(plan_path))
        (directory / 'SELECTION.json').write_text(json.dumps(selection, indent=2) + '\n')
        state.update(status='COMPLETE', phase='SELECTED', selected_epoch=selected['epoch'])
        save()
        print(json.dumps(selection, indent=2))

if __name__ == '__main__':
    main()

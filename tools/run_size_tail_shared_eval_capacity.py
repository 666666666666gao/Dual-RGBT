"""Continue the shared epoch chains using measured device and host capacity."""
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
from run_size_tail_epochs_keep_five import best_recent, stopping_epoch


def identity(state):
    return {key: state[key] for key in (
        'controller_pid', 'plan_sha256', 'physical_gpu', 'epoch', 'phase',
        'active_task', 'active_pid')}


def task_terminal(native, pid):
    assert native['pid'] == pid
    if native['state'] == 'RUNNING':
        assert (Path('/proc') / str(pid)).exists()
        return False
    assert native['state'] == 'COMPLETE' and native['exit_code'] == 0
    return True


def evaluation_capacity(gpu):
    cards = subprocess.check_output([
        'nvidia-smi', '--query-gpu=memory.free',
        '--format=csv,noheader,nounits'], universal_newlines=True).splitlines()
    free = int(cards[int(gpu)])
    available_kib = int(next(line.split()[1] for line in
        Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))
    # Observed per worker: 2.21 GiB device and 5.3 GiB resident host memory.
    # Only the two registered chains dispatch here. Reserve capacity for both
    # simultaneous admissions, even before nvidia-smi sees either new process.
    return free >= 6144 and available_kib >= 16 * 1024 * 1024


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--handoff', required=True)
    args = parser.parse_args()
    handoff_path = ROOT / args.handoff
    handoff = json.loads(handoff_path.read_text())
    review_path = ROOT / handoff['source_review']
    assert sha(review_path) == handoff['source_review_sha256']
    review = json.loads(review_path.read_text())
    assert review['status'] == 'SOURCE_PASS' and review['blocking_issues'] == 0
    check_files(ROOT, review['source_sha256'])
    plan_path = ROOT / handoff['plan']
    plan = json.loads(plan_path.read_text())
    assert sha(plan_path) == handoff['identity']['plan_sha256']
    assert plan['formal_launch_allowed'] and 5 <= plan['epochs'] <= 10
    check_files(ROOT, plan['source_sha256'])
    for key in ('parent_control_score', 'diagnostic_score'):
        assert sha(ROOT / plan[key]) == plan[key + '_sha256']
        score = json.loads((ROOT / plan[key]).read_text())
        assert score['status'] == 'SCORED_COMPLETE'
        assert (score['sequences'], score['frames']) == (97, 54646)
    directory = ROOT / 'runs' / handoff['chain']
    state_path = directory / 'status.json'
    state = json.loads(state_path.read_text())
    assert identity(state) == handoff['identity']
    assert state['phase'] in ('TRAIN', 'DEV', 'WAITING_FOR_EVAL_CAPACITY')
    native_path = ROOT / 'runs/pipeline_tasks' / state['active_task'] / 'status.json'
    native = json.loads(native_path.read_text())
    task_terminal(native, state['active_pid'])
    assert native['task'] == state['active_task']
    active_gpu = state['physical_gpu'] if state['phase'] != 'DEV' else state['active_gpu']
    assert native['physical_gpu'] == active_gpu
    if state['phase'] == 'WAITING_FOR_EVAL_CAPACITY':
        assert native['state'] == 'COMPLETE' and native['exit_code'] == 0
    assert native['command'][1] == str(ROOT / 'scripts' / (
        'eval_size_tail.py' if state['phase'] == 'DEV' else 'train_size_tail_epoch.py'))
    old_pid = state['controller_pid']
    assert (Path('/proc') / str(old_pid)).exists()
    ready = dict(pid=os.getpid(), handoff_sha256=sha(handoff_path),
                 identity=identity(state), status='READY_TO_ADOPT')
    ready_path = directory / 'shared_capacity_ready.json'
    assert not ready_path.exists()
    ready_temporary = directory / 'shared_capacity_ready.tmp'
    with ready_temporary.open('x') as handle:
        handle.write(json.dumps(ready, indent=2) + '\n')
    ready_temporary.replace(ready_path)
    deadline = time.time() + 20
    while (Path('/proc') / str(old_pid)).exists():
        assert time.time() < deadline
        time.sleep(.2)

    with (directory / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads(state_path.read_text())
        assert identity(state) == handoff['identity']
        adopted_phase, adopted_epoch = state['phase'], state['epoch']
        adopted_task, adopted_pid = state['active_task'], state['active_pid']
        state.update(controller_pid=os.getpid(), previous_controller_pid=old_pid,
                     evaluation_gpu=handoff['eval_gpu'], scheduler='shared_evaluation_capacity',
                     scheduler_source_sha256=review['source_sha256'],
                     scheduler_handoff_sha256=sha(handoff_path),
                     scheduler_review_sha256=sha(review_path))
        python = str(ROOT / 'conda-envs/dual_rgbt/bin/python')
        cpu = str(ROOT / 'conda-envs/cpu-scoring/bin/python')
        env = dict(os.environ, OPENCV_FOR_THREADS_NUM='1', PYTHONUNBUFFERED='1')

        def save():
            state['updated'] = time.time()
            temporary = directory / 'status.tmp'
            temporary.write_text(json.dumps(state, indent=2) + '\n')
            temporary.replace(state_path)

        def wait_task(task, pid):
            path = ROOT / 'runs/pipeline_tasks' / task / 'status.json'
            # The adopted wrapper publishes JSON non-atomically before exiting.
            while (Path('/proc') / str(pid)).exists():
                state['next_check_epoch'] = time.time() + 180
                save()
                time.sleep(180)
            native = json.loads(path.read_text())
            assert task_terminal(native, pid)
            return native

        def start_task(task, command, gpu):
            assert not (ROOT / 'runs/pipeline_tasks' / task).exists()
            wrapper = [python, str(ROOT / 'scripts/run_task_four_gpu.py'),
                       '--task', task, '--gpu', gpu, '--'] + command
            with (directory / (task + '.wrapper.log')).open('xb') as handle:
                process = subprocess.Popen(
                    wrapper, cwd=str(ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT)
            state.update(active_task=task, active_pid=process.pid, active_gpu=gpu)
            save()
            native_path = ROOT / 'runs/pipeline_tasks' / task / 'status.json'
            while process.poll() is None:
                state['next_check_epoch'] = time.time() + 180
                save()
                time.sleep(180)
            native = json.loads(native_path.read_text())
            assert process.returncode == 0 and task_terminal(native, process.pid)
            assert native['command'] == command and native['physical_gpu'] == gpu

        save()
        for epoch in range(adopted_epoch, plan['epochs'] + 1):
            first = epoch == adopted_epoch
            train_run = handoff['chain'] + '_train_e%03d' % epoch
            train_task = 'train_' + train_run
            eval_run = handoff['chain'] + '_dev_e%03d' % epoch
            eval_task = 'eval_' + eval_run
            if first:
                assert adopted_task == (eval_task if adopted_phase == 'DEV' else train_task)
                wait_task(adopted_task, adopted_pid)
            else:
                state.update(epoch=epoch, phase='WAITING_FOR_TRAIN_MEMORY')
                save()
                while True:
                    used = subprocess.check_output([
                        'nvidia-smi', '--query-gpu=memory.used',
                        '--format=csv,noheader,nounits'], universal_newlines=True).splitlines()
                    if int(used[int(state['physical_gpu'])]) <= plan['max_existing_gpu_mib']:
                        break
                    state['next_check_epoch'] = time.time() + 180
                    save()
                    time.sleep(180)
                previous = state['rows'][-1]
                command = [python, str(ROOT / 'scripts/train_size_tail_epoch.py'),
                           '--plan', handoff['plan'], '--run', train_run, '--task', train_task,
                           '--epoch', str(epoch), '--resume', previous['checkpoint'],
                           '--resume-sha256', previous['checkpoint_sha256']]
                state['phase'] = 'TRAIN'
                start_task(train_task, command, state['physical_gpu'])
            training = json.loads((ROOT / 'runs' / train_run / 'COMPLETE.json').read_text())
            checkpoint = 'runs/' + train_run + '/epoch_%03d.pth' % epoch
            assert training['strict_reload_bit_equal']
            assert sha(ROOT / checkpoint) == training['checkpoint_sha256']
            if not (first and adopted_phase == 'DEV'):
                state['phase'] = 'WAITING_FOR_EVAL_CAPACITY'
                save()
                while not evaluation_capacity(handoff['eval_gpu']):
                    state['next_check_epoch'] = time.time() + 180
                    save()
                    time.sleep(180)
                evaluation = json.loads((ROOT / plan['evaluation_template']).read_text())
                evaluation.update(mode='size_tail_diagnostic', checkpoint=checkpoint,
                                  checkpoint_sha256=training['checkpoint_sha256'],
                                  training_task=train_task,
                                  training_scope='Unchanged epoch plan; inference shares a GPU')
                evaluation_path = directory / ('eval_epoch_%03d.json' % epoch)
                with evaluation_path.open('x') as handle:
                    handle.write(json.dumps(evaluation, indent=2) + '\n')
                state['phase'] = 'DEV'
                start_task(eval_task, [python, str(ROOT / 'scripts/eval_size_tail.py'),
                           '--plan', evaluation_path.relative_to(ROOT).as_posix(),
                           '--run', eval_run, '--task', eval_task, '--smoke-frames', '0'],
                           handoff['eval_gpu'])
            score_path = directory / ('score_epoch_%03d.json' % epoch)
            assert not score_path.exists()
            state['phase'] = 'SCORE'
            save()
            with (directory / ('score_epoch_%03d.log' % epoch)).open('xb') as handle:
                subprocess.run([cpu, str(ROOT / 'tools/score_size_tail.py'),
                                '--run', eval_run, '--output', score_path.relative_to(ROOT).as_posix()],
                               cwd=str(ROOT), env=env, stdout=handle,
                               stderr=subprocess.STDOUT, check=True)
            metrics = json.loads(score_path.read_text())
            assert metrics['status'] == 'SCORED_COMPLETE'
            assert (metrics['sequences'], metrics['frames']) == (97, 54646)
            state['rows'].append(dict(
                epoch=epoch, checkpoint=checkpoint, checkpoint_sha256=training['checkpoint_sha256'],
                checkpoint_retained=True, score_path=score_path.relative_to(ROOT).as_posix(),
                score_sha256=sha(score_path),
                metrics={key: metrics[key] for key in ('PR', 'NPR', 'SR', 'fps')},
                training=training))
            if len(state['rows']) > 5:
                old = state['rows'][-6]
                path = ROOT / old['checkpoint']
                expected = ROOT / 'runs' / (handoff['chain'] + '_train_e%03d' % old['epoch'])
                assert path.parent == expected and path.name == 'epoch_%03d.pth' % old['epoch']
                assert sha(path) == old['checkpoint_sha256']
                path.unlink()
                old['checkpoint_retained'] = False
            save()
            if metrics['SR'] < plan['parent_metrics']['SR'] - plan['max_sr_drop_pp']:
                state.update(status='REJECTED_REGRESSION', phase='STOPPED',
                             stop_reason='Full-dev SR fell below the predeclared parent regression floor')
                save()
                (directory / 'REJECTION.json').write_text(json.dumps(dict(
                    epoch=epoch, score=state['rows'][-1], parent_metrics=plan['parent_metrics'],
                    max_sr_drop_pp=plan['max_sr_drop_pp'], plan_sha256=state['plan_sha256']),
                    indent=2) + '\n')
                return
            if stopping_epoch([row['metrics']['SR'] for row in state['rows']]) is not None:
                state['stop_reason'] = 'Three epochs without SR improvement greater than0.1pp; at least five observed'
                break
        check_files(ROOT, plan['source_sha256'])
        check_files(ROOT, review['source_sha256'])
        assert sha(plan_path) == state['plan_sha256']
        selected = best_recent(state['rows'])
        selection = dict(
            status='DEV_SELECTED', test_eligible=False, selected=selected,
            evaluated_epochs=[row['epoch'] for row in state['rows']],
            selection_window=[row['epoch'] for row in state['rows'][-5:]],
            ranking=['SR', 'PR', 'NPR'], parent_metrics=plan['parent_metrics'],
            protected_e1_metrics=plan['protected_e1_metrics'],
            stop_reason=state.get('stop_reason', 'Maximum planned epochs reached'),
            plan_sha256=sha(plan_path))
        with (directory / 'SELECTION.json').open('x') as handle:
            handle.write(json.dumps(selection, indent=2) + '\n')
        state.update(status='COMPLETE', phase='SELECTED', selected_epoch=selected['epoch'])
        save()


if __name__ == '__main__':
    main()

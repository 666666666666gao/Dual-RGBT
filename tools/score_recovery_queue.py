"""Verify and score completed recovery dev runs without changing frozen R4 files."""
import argparse
import fcntl
import json
from pathlib import Path
import os
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from benchmark_inputs import sha
from joint_candidate_results import official_metrics
from joint_evaluation_contract import verify_runs


def score(entry, toolkit_protocol, output):
    base = json.loads(toolkit_protocol.read_text())
    plan_path = entry['plan']
    plan = json.loads((ROOT / plan_path).read_text())
    spec = json.loads((ROOT / plan['protocol']).read_text())
    assert plan['status'] == 'RECOVERY_READY' and plan['test_eligible'] is False
    assert spec['initialization'] == base['initialization']
    manifest_path = spec['datasets']['LasHeR']['dev']
    assert manifest_path == base['datasets']['LasHeR']['dev']
    assert sha(ROOT / manifest_path) == base['input_sha256'][manifest_path]
    plan, spec, manifest, configs, timing, artifacts, tokens = verify_runs(
        ROOT, plan_path, manifest_path, [entry['run']])
    toolkit = ROOT / 'vendor/cadtrack/toolkit'
    for name, digest in base['toolkit_code_sha256'].items():
        assert sha(toolkit / name) == digest, name
    assert not output.exists() and not output.with_suffix('.npz').exists()
    report = official_metrics(toolkit, 'LasHeR', 'dev', [ROOT / 'runs' / entry['run']], configs, output)
    report.update(status='SCORED_COMPLETE', evaluation_class='RECOVERY_DEV_DIAGNOSTIC',
        training_native_status='FAILED', test_eligible=False, benchmark='LasHeR',
        candidate_id=plan['candidate_id'], initialization=spec['initialization'],
        training_plan=plan['training_plan'], evaluation_plan_sha256=sha(ROOT / plan_path),
        manifest_sha256=sha(ROOT / manifest_path), source_sha256=spec['source_sha256'],
        run_names=[entry['run']], run_artifacts_sha256=artifacts,
        curves_sha256=sha(output.with_suffix('.npz')), total_parameters=configs[0]['total_parameters'],
        max_candidates=plan['max_candidates'], input_sizes=configs[0]['input_sizes'],
        gpu_names=[configs[0]['gpu_name']], physical_gpus=[configs[0]['physical_gpu']],
        execution_host=socket.gethostname(), mean_heavy_tokens=tokens,
        toolkit_protocol_sha256=sha(toolkit_protocol), toolkit_code_sha256=base['toolkit_code_sha256'],
        scoring_script_sha256=sha(Path(__file__)),
        timing_scope='Concurrent execution; service FPS is not aggregate throughput or a paired speedup.')
    output.write_text(json.dumps(report, indent=2) + '\n')
    return {k: report[k] for k in ('candidate_id', 'cost_weight', 'PR', 'NPR', 'SR', 'fps')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--queue-state', required=True)
    parser.add_argument('--toolkit-protocol', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    directory = ROOT / args.output_dir
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / 'scoring.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while True:
        queue = json.loads((ROOT / args.queue_state).read_text())
        reports = []
        for item in queue['completed']:
            entry = item['entry']
            output = directory / (entry['run'] + '.json')
            if not output.exists():
                score(entry, ROOT / args.toolkit_protocol, output)
            report = json.loads(output.read_text())
            assert report['status'] == 'SCORED_COMPLETE' and report['test_eligible'] is False
            assert report['evaluation_plan_sha256'] == sha(ROOT / entry['plan'])
            reports.append(dict(path=output.relative_to(ROOT).as_posix(),
                **{k: report[k] for k in ('candidate_id', 'cost_weight', 'PR', 'NPR', 'SR', 'fps')}))
        terminal = queue['status'] in ('COMPLETE', 'COMPLETE_WITH_FAILURES')
        state = dict(status='COMPLETE' if terminal else 'WATCHING', pid=os.getpid(),
            updated=time.time(), poll_seconds=180, queue_state=args.queue_state,
            scored=reports, failed_evaluations=len(queue['failed']),
            scope='Local queue only; combine server25 and server26 for the complete 24-task recovery sweep.')
        (directory / 'scoring_status.json').write_text(json.dumps(state, indent=2) + '\n')
        print(json.dumps(dict(scored=len(reports), status=state['status'])), flush=True)
        if args.once or terminal:
            break
        time.sleep(180)
    lock.close()


if __name__ == '__main__':
    main()

"""Freeze one full-dev choice before any of the three benchmark tests."""
import argparse
import json
from pathlib import Path
import sys
import subprocess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from benchmark_inputs import frozen_selection, sha, validate_selection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--variant', choices=('e0', 'e1'), required=True)
    parser.add_argument('--dev-reports', nargs='+', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--toolkit', required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_bytes())
    assert plan['status'] == 'READY'
    assert all(sha(ROOT / name) == digest for name, digest in plan['source_sha256'].items())
    paths = [Path(path).resolve() for path in args.dev_reports]
    reports = [json.loads(path.read_bytes()) for path in paths]
    for path, report in zip(paths, reports):
        recomputed_path = path.with_name(path.stem + '_freeze_recomputed.json')
        subprocess.run([sys.executable, str(ROOT / 'tools/score_benchmark_tracking.py'),
            '--plan', args.plan, '--manifest', args.manifest, '--toolkit', args.toolkit,
            '--runs'] + [str(ROOT / 'runs' / name) for name in report['run_names']] +
            ['--output', str(recomputed_path)], check=True)
        recomputed = json.loads(recomputed_path.read_bytes())
        assert recomputed == report
    best = max(reports, key=lambda report: (report['SR'], report['PR'], report['NPR'], -report['mean_ms']))
    selection = dict(selection_split='LasHeR_dev97', dev_sequences=97, dev_frames=54646,
        variant=args.variant, checkpoint_sha256=best['checkpoint_sha256'], evaluation_plan_sha256=sha(args.plan),
        configurations={benchmark: dict(cost_weight=best['cost_weight'], max_candidates=best['max_candidates'])
                        for benchmark in plan['datasets']},
        dev_receipts=[dict(path=str(path.relative_to(ROOT)).replace('\\', '/'), sha256=sha(path)) for path in paths],
        criterion='Highest full-dev SR, then PR, NPR, lower measured mean latency; fixed final C round2.')
    output = ROOT / plan['models'][args.variant]['frozen_selection']
    assert not output.exists()
    validate_selection(ROOT, plan, args.variant, best['checkpoint_sha256'], sha(args.plan), selection)
    output.write_bytes((json.dumps(selection, indent=2) + '\n').encode())
    digest = sha(output)
    frozen_selection(ROOT, plan, args.variant, best['checkpoint_sha256'], sha(args.plan), digest)
    print(json.dumps(dict(status='FROZEN', path=str(output), selection_sha256=digest)), flush=True)


if __name__ == '__main__':
    main()

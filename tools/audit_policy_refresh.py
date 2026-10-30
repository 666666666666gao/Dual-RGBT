"""Check complete refresh coverage before the existing sampled-query audit."""
import argparse
import json
from pathlib import Path

from summarize_frozen_query_cache import main as summarize, sha

ROOT = Path(__file__).resolve().parents[1]


def verify_coverage(config, receipt):
    plan = config['plan']
    expected = plan['fit_sequences'] + plan['holdout_sequences']
    if config['smoke']:
        expected = expected[:1]
    assert config['selected_sequences'] == expected
    assert receipt['sequences'] == len(expected)
    assert receipt['frames'] == (plan['smoke_frames'] if config['smoke'] else plan['expected_frames'])
    assert receipt['frozen_state_tensors'] == 265
    if config['smoke']:
        assert receipt['observer_policy_trajectory_equal']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run = ROOT / 'runs' / args.run
    config = json.loads((run / 'config.json').read_text())
    receipt = json.loads((run / ('SMOKE.json' if config['smoke'] else 'COMPLETE.json')).read_text())
    verify_coverage(config, receipt)
    summarize()
    output = ROOT / args.output
    report = json.loads(output.read_text())
    report.update(base_auditor_sha256=report['source_sha256'], source_sha256=sha(Path(__file__)),
                  complete_planned_coverage=True,
                  scope='Frozen learned-policy TRAIN-state samples; exact planned coverage verified. Not exhaustive recovery recall or benchmark accuracy.')
    output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()

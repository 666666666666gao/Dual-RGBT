"""Reseal not-yet-started evaluation protocols with primary/query evidence capture."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not (ROOT / 'reports/joint_candidate_evaluation').exists()
    archive = ROOT / '.aris/traces/experiment-audit/2026-10-27_query_evidence_capture/pre_change'
    names = ['reports/JOINT_CANDIDATE_EVALUATION_PROTOCOL_20261026.json',
             'reports/VRAM18_EVALUATION_PROTOCOL_20261027.json',
             'reports/SERVER25_JOINT_EVALUATION_PROTOCOL_20261027.json']
    changed = ['scripts/run_joint_candidate_evaluation.py', 'scripts/joint_evaluation_contract.py']
    added = ['scripts/query_evidence_capture.py', 'tools/register_query_evidence_capture.py']
    for name in names:
        assert sha(ROOT / name) == sha(archive / name)
        spec = json.loads((ROOT / name).read_bytes())
        for source, digest in spec['source_sha256'].items():
            if source not in changed:
                assert sha(ROOT / source) == digest, source
        for source in changed + added:
            spec['source_sha256'][source] = sha(ROOT / source)
        spec['evidence_schema'] = 'primary_executed_box_and_selected_gain_v1'
        spec['scope'] += (' Evaluation-only capture retains computed primary boxes and selected gains; '
                          'no additional encoder, routing/weight/training change or GT in tracker. '
                          'Offline signed IoU gain requires primary and executed final boxes. '
                          'STOP opportunity and unexecuted alternatives remain unobserved.')
        (ROOT / name).write_bytes((json.dumps(spec, indent=2)+'\n').encode())
    print(json.dumps({name:sha(ROOT/name) for name in names}))


if __name__ == '__main__':
    main()

"""Seal the eight known training identities and unchanged metric/input protocol."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from benchmark_inputs import sha


def main():
    index = json.loads((ROOT / 'reports/CONCURRENT_TRAIN_PLAN_INDEX_20261026.json').read_bytes())
    training_paths = ['reports/global_candidate_joint_quality_trajectory_plan_20261026.json',
                      'reports/global_candidate_joint_gainpair_trajectory_plan_20261026.json'] + [p['path'] for p in index['plans']]
    assert len(training_paths) == len(set(training_paths)) == 8
    sources, training_hashes = {}, {}
    for name in training_paths:
        plan = json.loads((ROOT / name).read_bytes())
        assert plan['status'] == 'READY' and plan['max_candidates'] in (4, 8)
        assert plan['expected_formal_counts'] == dict(updates=182182, tracked_frames=917106, initializes=1764)
        training_hashes[name] = sha(ROOT / name)
        for path, digest in plan['source_sha256'].items():
            assert path not in sources or sources[path] == digest
            sources[path] = digest
    for path in ['scripts/joint_evaluation_contract.py', 'scripts/run_joint_candidate_evaluation.py',
                 'tools/joint_candidate_results.py', 'tools/register_joint_evaluation_protocol.py']:
        sources[path] = sha(ROOT / path)
    assert all(sha(ROOT / path) == digest for path, digest in sources.items())
    manifests = json.loads((ROOT / 'reports/JOINT_EVALUATION_STATIC_MANIFESTS_20261026.json').read_bytes())['manifests']
    datasets, sizes = {}, {}
    inputs = {}
    for path, row in manifests.items():
        datasets.setdefault(row['benchmark'], {})[row['split']] = path
        sizes.setdefault(row['benchmark'], {})[row['split']] = dict(sequences=row['sequences'], frames=row['frames'])
        inputs[path] = row['sha256']
    base = json.loads((ROOT / training_paths[0]).read_bytes())
    for path in [base['factory_checkpoint'], base['checkpoint'], 'checkpoints/CADTrack_ep0014.pth.tar']:
        inputs[path] = base['input_sha256'][path]
    for path in ['reports/RANKSTOP_DEV_FREEZE_DECISION_20261026.json',
                 'reports/SERVER26_MODEL_TOTAL_PARAMETERS_20261026.json']:
        inputs[path] = sha(ROOT / path)
    old = json.loads((ROOT / 'reports/benchmark_evaluation_plan.json').read_bytes())
    result = dict(status='READY', seed=2027,
        source_review='reports/JOINT_CANDIDATE_EVALUATION_SOURCE_REVIEW_20261026.json',
        source_sha256=sources, input_sha256=inputs, training_plan_sha256=training_hashes,
        toolkit_code_sha256=old['toolkit_code_sha256'], datasets=datasets, split_sizes=sizes,
        formal_counts=dict(updates=182182, tracked_frames=917106, initializes=1764),
        total_parameters=103218887, dev_cost_candidates=[0.0, 0.05, 0.15],
        reference_decision='reports/RANKSTOP_DEV_FREEZE_DECISION_20261026.json',
        initialization=dict(label='CADTrack without its three paper modules; SOT backbone; epoch14',
                            pretrained_sha256=base['protected_checkpoint_sha256']),
        selection_criterion='Highest full97dev SR, then PR, NPR, lower executed mean heavy-token count, higher cost coefficient. No latency tie-break from unequal concurrent loads.',
        scope='Source-only evaluation recipe for eight registered joint-controller finals. prepare rejects running/native-failed/sanity/partial/old8-head finals before touching checkpoint. Require full882x2 native0,265finite tensors/exact251frozen+14changed versus originalB,source/input metadata and actual finalSHA. One real11frame train-smoke with strictreload, then three complete97dev costs. Fresh DEV_RESULT_AUDIT_PASS(0blockers) must bind planSHA/checkpointSHA/exact3reportSHA map. Eligibility strictly exceeds oldE1 fullDEV SR56.43314371195134; failing candidates have no test. Freeze one checkpoint/config for LasHeR245/220703,RGBT210210/104706,RGBT234234/116650; official supplied toolkit unchanged incl inheritedafterrain/init/rounding quirks. GPU0/1/2/3 only with separate currentownership/capacity admission. Per-frame serviceFPS includesinit+decode,excludesload/writes/verification,notparallel aggregate throughput. Totalparameters only, nottrainable count. No new NN execution/checkpoint/result is asserted by protocol registration.')
    path = ROOT / 'reports/JOINT_CANDIDATE_EVALUATION_PROTOCOL_20261026.json'
    assert not path.exists()
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(status='REGISTERED_SOURCE_REVIEW_PENDING', protocol=path.relative_to(ROOT).as_posix(),
        protocol_sha256=sha(path), source_files=len(sources), registered_models=len(training_paths))), flush=True)


if __name__ == '__main__':
    main()

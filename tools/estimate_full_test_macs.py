"""Reweight existing MAC measurements with verified full-test route counts."""
import argparse
import ast
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def query_macs(source):
    tree = ast.parse(source)
    assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Attribute) and t.attr == 'query_gain'
                           for t in node.targets)]
    assert len(assignments) == 1
    layers = assignments[0].value.args
    assert [node.func.attr for node in layers] == ['Linear', 'GELU', 'Linear']
    dimensions = [tuple(ast.literal_eval(arg) for arg in node.args)
                  for node in (layers[0], layers[2])]
    assert dimensions == [(955, 64), (64, 1)]
    return sum(a * b for a, b in dimensions)


def shape_estimates(measured, mixture, per_query):
    table = {(r['modality'], r['action'], r['candidate_count']): r['conv_matrix_macs']
             for r in measured}
    assert len(table) == len(measured)

    def estimate(modality, action, candidates):
        # Two local requests plus two requests per global candidate.
        return table[(modality, action, 8)] + 2 * (candidates - 8) * per_query

    for shape, value in table.items():
        assert estimate(*shape) == value, shape
    rows = []
    for (modality, action, candidates), frames in sorted(mixture.items()):
        assert modality in (0, 1) and action in range(5) and 0 <= candidates <= 8
        shape = (modality, action, candidates)
        rows.append(dict(modality=modality, action=action, candidate_count=candidates,
            frames=frames, conv_matrix_macs=estimate(*shape),
            provenance='existing_profiled_shape' if shape in table else 'analytic_shape_estimate'))
    return rows


def analyze(root, profile_path, report_path, run):
    profile = json.loads(profile_path.read_bytes())
    report = json.loads(report_path.read_bytes())
    assert profile['status'] == 'PROFILE_COMPLETE'
    assert report['status'] == 'SCORED_COMPLETE'
    assert (report['split'], report['sequences'], report['frames']) == ('test', 245, 220703)
    assert report['exploratory_test'] is True and report['test_eligible'] is False
    assert profile['checkpoint_sha256'] == report['parent_checkpoint_sha256']
    assert profile['total_parameters'] == report['total_parameters'] == 103218887
    assert sha(report_path.with_suffix('.npz')) == report['curves_sha256']
    sources = {name: digest for name, digest in profile['source_sha256'].items()
               if name.startswith(('dual_rgbt/', 'dual_rgbt_evidence/', 'dual_rgbt_global/', 'lib/'))
               or name == 'experiments/cadtrack/cadtrack.yaml'}
    assert len(sources) == 38
    for name, digest in sources.items():
        assert report['source_sha256'][name] == sha(root / name) == digest, name
    per_query = query_macs((root / 'dual_rgbt_evidence/policy.py').read_text())
    directory, used = root / 'runs' / run, {}

    def checked(relative):
        path = directory / relative
        digest = sha(path)
        assert report['run_artifacts_sha256'][relative] == digest, relative
        used[path.relative_to(root).as_posix()] = digest
        return path

    config = json.loads(checked('config.json').read_bytes())
    complete = json.loads(checked('COMPLETE.json').read_bytes())
    assert config['run'] == run and config['split'] == 'test' and config['smoke_frames'] == 0
    assert config['checkpoint_sha256'] == complete['checkpoint_sha256'] == report['checkpoint_sha256']
    assert config['source_sha256'] == report['source_sha256']
    assert config['max_candidates'] == 8 and config['cost_weight'] == report['cost_weight'] == .15
    assert complete['config_sha256'] == sha(directory / 'config.json')
    assert complete['optimizer_updates'] == complete['backward_calls'] == 0
    native_path = root / 'runs/pipeline_tasks' / config['task'] / 'status.json'
    native = json.loads(native_path.read_bytes())
    assert sha(native_path) == report['native_status_sha256']
    assert native['state'] == 'COMPLETE' and native['exit_code'] == 0
    assert native['command'] == config['native_command']
    assert native['task'] == complete['task'] == config['task']
    names = config['sequences']
    assert len(names) == len(set(names)) == 245
    assert set(names) == {row['sequence'] for row in report['per_sequence']}
    progress = json.loads(checked('progress.json').read_bytes())
    lengths = {row['sequence']: row['frames'] for row in progress}
    assert len(progress) == len(lengths) == 245 and set(lengths) == set(names)
    assert sum(lengths.values()) == complete['frames'] == 220703
    mixture = Counter()
    for name in names:
        with checked('traces/' + name + '.csv').open(newline='') as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) + 1 == lengths[name]
        assert [int(row['frame']) for row in rows] == list(range(1, len(rows) + 1))
        mixture.update((int(r['modality']), int(r['action']), int(r['candidate_count'])) for r in rows)
    tracked = sum(mixture.values())
    assert tracked == 220458
    shapes = shape_estimates(profile['measured_cases'], mixture, per_query)
    total = sum(row['frames'] * row['conv_matrix_macs'] for row in shapes)
    derived = [r for r in shapes if r['provenance'] == 'analytic_shape_estimate']
    return dict(status='ANALYTIC_FULL_TEST_MACS_ESTIMATE', run=run,
        benchmark='LasHeR', split='test', sequences=245, frames=220703,
        tracked_frames=tracked, total_parameters=report['total_parameters'],
        exploratory_test=True, test_eligible=False, fresh_gpu_profile=False,
        source_profile=profile_path.relative_to(root).as_posix(),
        source_profile_sha256=sha(profile_path),
        source_report=report_path.relative_to(root).as_posix(), source_report_sha256=sha(report_path),
        checkpoint_sha256=report['checkpoint_sha256'],
        measured_parent_checkpoint_sha256=profile['checkpoint_sha256'],
        native_status_sha256=sha(native_path), core_source_sha256=sources,
        used_artifacts_sha256=used, estimator_sha256=sha(Path(__file__)),
        per_query_linear_macs=per_query, per_candidate_linear_macs=2 * per_query,
        existing_profiled_cases_checked=len(profile['measured_cases']),
        observed_shapes=len(shapes), analytically_derived_shapes=len(derived),
        analytically_derived_frames=sum(r['frames'] for r in derived),
        mean_tracking_macs_estimate=total / tracked,
        mean_macs_including_initialization_estimate=(
            total + 245 * profile['initialization']['conv_matrix_macs']) / 220703,
        shape_estimates=shapes, optimizer_updates=0, backward_calls=0,
        scope='CPU analytic reweighting, NOT a new GPU measurement. Reuses parent profiled '
              'action-at-K8 shapes and adds 2*(K-8)*(955*64+64) linear MACs. Formula exactly '
              'matches all existing measured cases; missing shapes remain analytically estimated. '
              'The overlay changes gain weights, not the verified core computation shapes. '
              'Counts profiler-supported convolution/matrix MACs only; excludes normalization, '
              'softmax, sampling, decoding, pointwise arithmetic and other unsupported operations. '
              'Initialization reuses the parent measurement. No FPS, accuracy improvement, '
              'selection decision or independent scientific acceptance.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = ROOT / args.output
    assert not output.exists()
    result = analyze(ROOT, ROOT / args.profile, ROOT / args.report, args.run)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in (
        'status', 'observed_shapes', 'analytically_derived_shapes',
        'analytically_derived_frames', 'mean_tracking_macs_estimate')}, indent=2))


if __name__ == '__main__':
    main()

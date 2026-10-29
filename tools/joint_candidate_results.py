"""Prepare, score and freeze completed joint-controller evaluations."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from benchmark_inputs import groundtruth, sha
from joint_evaluation_contract import check_files, evaluation_command, evaluation_plan, protocol, run_name
from joint_evaluation_contract import checked_dev_report, selected_configuration, tensor_witness, training_metadata, verify_runs


def prepare(args):
    spec = protocol(ROOT, args.protocol)
    check_files(ROOT, spec['input_sha256'])
    training, config, complete, native_path = training_metadata(ROOT, spec, args.training_plan)
    witness = tensor_witness(ROOT, training, config, complete)
    candidate = training['runs']['formal']
    directory = ROOT / 'runs' / candidate
    inputs = dict(spec['input_sha256'])
    for path in [ROOT / args.training_plan, ROOT / training['source_review'], native_path,
                 directory / 'config.json', directory / 'COMPLETE.json', directory / 'latest.pth',
                 ROOT / spec['source_review']]:
        inputs[path.relative_to(ROOT).as_posix()] = sha(path)
    plan = dict(status='READY', candidate_id=candidate, protocol=args.protocol,
        initialization=spec['initialization'],
        protocol_sha256=sha(ROOT / args.protocol), training_plan=args.training_plan,
        checkpoint='runs/' + candidate + '/latest.pth', checkpoint_sha256=complete['checkpoint_sha256'],
        training_config_sha256=sha(directory / 'config.json'), factory_checkpoint=training['factory_checkpoint'],
        max_candidates=training['max_candidates'], source_sha256=spec['source_sha256'], input_sha256=inputs,
        tensor_witness=witness,
        frozen_selection='reports/joint_candidate_evaluation/' + candidate + '/frozen_selection.json',
        eligibility_decision='reports/joint_candidate_evaluation/' + candidate + '/test_eligibility.json')
    output = ROOT / args.output
    assert not output.exists()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2)+'\n')
    print(json.dumps(dict(status='READY', path=args.output, candidate_id=candidate,
        checkpoint_sha256=plan['checkpoint_sha256'], tensor_witness=witness)), flush=True)


def official_metrics(toolkit, benchmark, split, runs, configs, output):
    """Only the supplied toolkit computes precision/success curves."""
    sys.path.insert(0, str(toolkit))
    if benchmark == 'LasHeR':
        import score_tracking
        sys.argv = ['score_tracking.py', '--toolkit', str(toolkit), '--runs'] + [str(p) for p in runs] + ['--output', str(output)]
        score_tracking.main()
        return json.loads(output.read_bytes())
    from rgbt import RGBT210, RGBT234
    from rgbt.dataset.basedataset import TrackerResult
    from rgbt.metrics.metrics import PR, SR, MPR, MSR
    dataset = RGBT210() if benchmark == 'RGBT210' else RGBT234()
    assert set(dataset.seqs_name) == {name for cfg in configs for name in cfg['sequences']}
    metrics = [('PR', PR()), ('SR', SR())] if benchmark == 'RGBT210' else [('MPR', MPR()), ('MSR', MSR())]
    curves = {key: [] for key, _ in metrics}
    rows = []
    for run, cfg in zip(runs, configs):
        result = TrackerResult('Dual-RGBT-joint', str(run / 'predictions'), cfg['sequences'], '', 'ltwh')
        chunk = {}
        for key, metric in metrics:
            _, array = metric(dataset, result, cfg['sequences'])
            curves[key].extend(array.tolist())
            chunk[key] = array
        for index, name in enumerate(cfg['sequences']):
            row = dict(sequence=name)
            row.update({key:100*float(array[index, 20] if key in ('PR', 'MPR') else array[index].mean())
                        for key, array in chunk.items()})
            rows.append(row)
    arrays = {key: np.asarray(value) for key, value in curves.items()}
    np.savez_compressed(output.with_suffix('.npz'), **arrays)
    return dict(benchmark=benchmark, split=split, per_sequence=rows,
        **{key:100*float(array[:, 20].mean() if key in ('PR', 'MPR') else array.mean()) for key, array in arrays.items()})


def score(args):
    result = verify_runs(ROOT, args.plan, args.manifest, args.runs)
    plan, spec, manifest, configs, timing, artifacts, mean_tokens = result
    toolkit = Path(args.toolkit)
    assert all(sha(toolkit / name) == digest for name, digest in spec['toolkit_code_sha256'].items())
    if manifest['benchmark'] != 'LasHeR':
        assert all(sha(toolkit / name.replace('\\', '/')) == digest for name, digest in manifest['annotation_file_sha256'].items())
    sys.path.insert(0, str(toolkit))
    from rgbt import LasHeR, RGBT210, RGBT234
    dataset = (LasHeR() if manifest['benchmark'] == 'LasHeR' and manifest['split'] == 'test' else
               RGBT210() if manifest['benchmark'] == 'RGBT210' else
               RGBT234() if manifest['benchmark'] == 'RGBT234' else None)
    if dataset is not None:
        for sequence in manifest['sequences']:
            visible, infrared = groundtruth(sequence, manifest['benchmark'])
            if manifest['benchmark'] == 'LasHeR':
                assert np.array_equal(visible.round(0), np.asarray(dataset[sequence['name']]))
            else:
                for modality, gt in (('visible', visible), ('infrared', infrared)):
                    assert np.array_equal(gt, np.asarray(dataset[sequence['name']][modality], dtype=np.float32))
    output = ROOT / args.output
    assert not output.exists() and not output.with_suffix('.npz').exists()
    output.parent.mkdir(parents=True, exist_ok=True)
    report = official_metrics(toolkit, manifest['benchmark'], manifest['split'],
                              [ROOT / 'runs' / name for name in args.runs], configs, output)
    report.update(status='SCORED_COMPLETE', candidate_id=plan['candidate_id'],
        initialization=plan['initialization'], training_plan=plan['training_plan'],
        benchmark=manifest['benchmark'], split=manifest['split'], sequences=manifest['sequence_count'],
        frames=manifest['frame_count'], checkpoint_sha256=plan['checkpoint_sha256'],
        evaluation_plan_sha256=sha(ROOT / args.plan), manifest_sha256=sha(ROOT / args.manifest),
        source_sha256=spec['source_sha256'], run_names=args.runs, run_artifacts_sha256=artifacts,
        curves_sha256=sha(output.with_suffix('.npz')), cost_weight=configs[0]['cost_weight'],
        max_candidates=plan['max_candidates'], total_parameters=configs[0]['total_parameters'],
        gpu_names=sorted({cfg['gpu_name'] for cfg in configs}),
        physical_gpus=sorted({cfg['physical_gpu'] for cfg in configs}), input_sizes=configs[0]['input_sizes'],
        mean_heavy_tokens=mean_tokens, mean_ms=1000*float(timing.mean()),
        p95_ms=1000*float(np.quantile(timing, .95)), fps=len(timing)/float(timing.sum()),
        initialization_included=True, image_decoding_included=True, model_loading_excluded=True,
        fps_definition='Frames divided by summed per-frame service time across shards; not parallel aggregate throughput.',
        frozen_selection_sha256=configs[0]['selection_sha256'], annotation_source=manifest['annotation_source'],
        timing_scope='Actual execution load may include other jobs; no paired baseline speedup claim.')
    if plan.get('status') == 'RECOVERY_READY':
        report.update(evaluation_class='RECOVERY_DEV_DIAGNOSTIC', training_native_status='FAILED', test_eligible=False)
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('per_sequence', 'run_artifacts_sha256', 'source_sha256')}), flush=True)


def freeze(args):
    plan, spec, _ = evaluation_plan(ROOT, args.plan)
    assert plan.get('status') != 'RECOVERY_READY'
    review = json.loads((ROOT / args.result_review).read_bytes())
    assert review['status'] == 'DEV_RESULT_AUDIT_PASS' and review['blocking_issues'] == 0
    assert review['evaluation_plan_sha256'] == sha(ROOT / args.plan)
    assert review['checkpoint_sha256'] == plan['checkpoint_sha256']
    assert len(args.dev_reports) == len(set(args.dev_reports)) == 3
    reports = []
    for path in args.dev_reports:
        report = checked_dev_report(ROOT, args.plan, plan, spec, ROOT / path)
        assert review['dev_reports_sha256'][path] == sha(ROOT / path)
        result = verify_runs(ROOT, args.plan, spec['datasets']['LasHeR']['dev'], report['run_names'])
        assert result[5] == report['run_artifacts_sha256'] and result[6] == report['mean_heavy_tokens']
        assert np.isclose(report['fps'], len(result[4])/float(result[4].sum()), rtol=1e-12)
        reports.append(report)
    assert set(review['dev_reports_sha256']) == set(args.dev_reports)
    assert sorted(r['cost_weight'] for r in reports) == spec['dev_cost_candidates']
    best = max(reports, key=lambda r: (r['SR'], r['PR'], r['NPR'], -r['mean_heavy_tokens'], r['cost_weight']))
    reference = json.loads((ROOT / spec['reference_decision']).read_bytes())
    decision_path = ROOT / plan['eligibility_decision']
    assert not decision_path.exists() and not (ROOT / plan['frozen_selection']).exists()
    decision_path.parent.mkdir(parents=True, exist_ok=True)
    decision = dict(status='ELIGIBLE' if best['SR'] > reference['reference_dev']['SR'] else 'REJECT_NO_TEST',
        candidate_id=plan['candidate_id'], checkpoint_sha256=plan['checkpoint_sha256'],
        best_dev={k: best[k] for k in ('PR', 'NPR', 'SR', 'cost_weight')}, reference_dev=reference['reference_dev'],
        result_review=args.result_review, result_review_sha256=sha(ROOT / args.result_review), test_launched=False)
    if decision['status'] == 'ELIGIBLE':
        selection = dict(status='FROZEN', candidate_id=plan['candidate_id'], selection_split='LasHeR_dev97',
            checkpoint_sha256=plan['checkpoint_sha256'], evaluation_plan_sha256=sha(ROOT / args.plan),
            criterion=spec['selection_criterion'], result_review=args.result_review,
            result_review_sha256=sha(ROOT / args.result_review),
            dev_receipts=[dict(path=path, sha256=sha(ROOT / path)) for path in args.dev_reports],
            configurations={name:dict(cost_weight=best['cost_weight'], max_candidates=plan['max_candidates'])
                            for name in spec['datasets']})
        (ROOT / plan['frozen_selection']).write_text(json.dumps(selection, indent=2)+'\n')
        selected_configuration(ROOT, args.plan, sha(ROOT / plan['frozen_selection']))
        decision['selection_sha256'] = sha(ROOT / plan['frozen_selection'])
    decision_path.write_text(json.dumps(decision, indent=2)+'\n')
    print(json.dumps(decision), flush=True)


def commands(args):
    plan, spec, training = evaluation_plan(ROOT, args.plan)
    if plan.get('status') == 'RECOVERY_READY':
        assert (args.benchmark, args.split) == ('LasHeR', 'dev')
        assert args.selection_sha256 is None
    assert 1 <= args.shards <= spec['split_sizes'][args.benchmark][args.split]['sequences']
    assert args.split == 'test' or args.selection_sha256 is None
    if args.split == 'test':
        selection = selected_configuration(ROOT, args.plan, args.selection_sha256)
        costs = [selection['configurations'][args.benchmark]['cost_weight']]
    elif args.split == 'dev':
        assert args.benchmark == 'LasHeR'
        costs = spec['dev_cost_candidates']
    else:
        assert args.benchmark == 'LasHeR' and args.shards == 1
        costs = [training['cost_weight']]
    result = []
    for cost in costs:
        for shard in range(args.shards):
            run = run_name(plan, args.benchmark, args.split, cost, shard, args.shards)
            cfg = dict(plan=args.plan, benchmark=args.benchmark, split=args.split, task='eval_' + run,
                run=run, cost_weight=cost, shard=shard, shards=args.shards,
                smoke_frames=11 if args.split == 'train_smoke' else 0, selection_sha256=args.selection_sha256)
            result.append(dict(task=cfg['task'], run=run, command=evaluation_command(cfg, spec),
                required_environment=dict(OPENCV_FOR_THREADS_NUM='1'),
                scope='Command registration only; physical GPU ownership/capacity must be checked before launch.'))
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='operation', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--protocol', required=True)
    p.add_argument('--training-plan', required=True)
    p.add_argument('--output', required=True)
    p = sub.add_parser('score')
    p.add_argument('--plan', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--toolkit', required=True)
    p.add_argument('--runs', nargs='+', required=True)
    p.add_argument('--output', required=True)
    p = sub.add_parser('freeze')
    p.add_argument('--plan', required=True)
    p.add_argument('--dev-reports', nargs=3, required=True)
    p.add_argument('--result-review', required=True)
    p = sub.add_parser('commands')
    p.add_argument('--plan', required=True)
    p.add_argument('--benchmark', choices=('LasHeR', 'RGBT210', 'RGBT234'), required=True)
    p.add_argument('--split', choices=('train_smoke', 'dev', 'test'), required=True)
    p.add_argument('--shards', type=int, required=True)
    p.add_argument('--selection-sha256')
    args = parser.parse_args()
    dict(prepare=prepare, score=score, freeze=freeze, commands=commands)[args.operation](args)


if __name__ == '__main__':
    main()

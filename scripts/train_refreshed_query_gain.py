"""Refit the parent gain head on verified frozen-E6 TRAIN-state features."""
import argparse
import copy
import csv
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark_inputs import sha
from joint_evaluation_contract import check_files, evaluation_plan
from run_benchmark_tracking import build_model, freeze


def objective(prediction, gain, cost, variant):
    regression = F.smooth_l1_loss(prediction, gain)
    mask = gain.abs() > .01
    margin = F.relu(.05 - gain.sign() * prediction)[mask].mean() if mask.any() else regression.new_zeros(())
    base = regression + .25 * margin
    if variant == 'parent_loss':
        return base
    assert variant == 'utility_focal'
    # Output stays in IoU-gain units; only the training loss uses logits.
    logits = (prediction - cost) / .1
    target = (gain > cost).to(prediction.dtype)
    probability = logits.sigmoid()
    p_correct = probability * target + (1 - probability) * (1 - target)
    alpha = .75 * target + .25 * (1 - target)
    focal = alpha * (1 - p_correct).square() * F.binary_cross_entropy_with_logits(
        logits, target, reduction='none')
    return base + .1 * focal.mean()


def diagnostics(prediction, gain, primary_iou, cost, sequences):
    """STOP versus the four sampled requests; not full candidate or trajectory recall."""
    utility = prediction - cost
    chosen = utility.argmax(1)
    queried = utility.max(1) > 0
    index = np.arange(len(chosen))
    actual = gain[index, chosen] * queried
    final = primary_iou + actual
    query_iou = primary_iou[:, None] + gain
    recoverable = (primary_iou < .1) & (query_iou.max(1) >= .5)
    selected_utility = (gain[index, chosen] - cost[chosen]) * queried
    oracle = np.maximum(0, (gain - cost).max(1))
    return dict(states=len(chosen), queries=int(queried.sum()),
        mean_gain_mae=float(np.abs(prediction - gain).mean()),
        mean_sampled_utility_regret=float((oracle - selected_utility).mean()),
        sampled_recoverable_states=int(recoverable.sum()),
        recovery_sequences=len({s for s, yes in zip(sequences, recoverable) if yes}),
        recoveries=int(((primary_iou < .1) & (final >= .5)).sum()),
        destroys_good=int(((primary_iou >= .5) & (final < .1)).sum()),
        negative_utility_queries=int((queried & (selected_utility < 0)).sum()))


def check_cache_policy(config, receipt, audit, plan):
    assert config['plan']['mode'] == 'frozen_policy_train_refresh'
    assert config['split'] == 'train' and config['smoke'] is False
    assert config['parent_checkpoint_sha256'] == plan['parent_checkpoint_sha256']
    assert config['checkpoint_sha256'] == receipt['checkpoint_sha256'] == plan['cache_checkpoint_sha256']
    assert config['plan']['overlay_sha256'] == plan['cache_checkpoint_sha256']
    assert config['plan']['checkpoint_sha256'] == plan['cache_checkpoint_sha256']
    assert config['plan']['seed'] == plan['seed']
    assert config['plan']['cost_weight'] == 0.15 and config['plan']['label_stride'] == 10
    assert audit['complete_planned_coverage'] is True
    expected = config['plan']['fit_sequences'] + config['plan']['holdout_sequences']
    assert config['selected_sequences'] == expected and receipt['sequences'] == len(expected)
    assert receipt['frames'] == config['plan']['expected_frames']


def load_caches(plan):
    data = {p: dict(features=[], gains=[], primary=[], recorded=[], sequences=[]) for p in ('fit', 'holdout')}
    seen, input_hashes = set(), {}
    for entry in plan['caches']:
        audit_path = ROOT / entry['audit']
        assert sha(audit_path) == entry['audit_sha256']
        audit = json.loads(audit_path.read_text())
        assert audit['status'] == 'VERIFIED_CACHE_DIAGNOSTIC' and not audit['smoke']
        run = ROOT / 'runs' / entry['run']
        receipt_path, config_path = run / 'COMPLETE.json', run / 'config.json'
        native_path = ROOT / 'runs/pipeline_tasks' / entry['task'] / 'status.json'
        assert sha(receipt_path) == audit['receipt_sha256'] and sha(native_path) == audit['native_sha256']
        config, receipt, native = [json.loads(p.read_text()) for p in (config_path, receipt_path, native_path)]
        assert native['state'] == 'COMPLETE' and native['exit_code'] == 0
        assert native['command'] == config['native_command'] and audit['run'] == entry['run']
        check_cache_policy(config, receipt, audit, plan)
        assert receipt['status'] == 'CACHE_COMPLETE' and receipt['frozen_state_tensors'] == 265
        assert receipt['optimizer_updates'] == receipt['backward_calls'] == 0
        assert receipt['teacher_preserves_tracker_state']
        for name, digest in receipt['artifacts_sha256'].items():
            path = run / name
            assert sha(path) == digest
            input_hashes[str(path.relative_to(ROOT))] = digest
        for path in (audit_path, receipt_path, native_path):
            input_hashes[str(path.relative_to(ROOT))] = sha(path)
        for name in config['selected_sequences']:
            assert name not in seen
            seen.add(name)
            partition = 'fit' if name in config['plan']['fit_sequences'] else 'holdout'
            features = np.load(run / (name + '_features.npy'), allow_pickle=False)
            with (run / (name + '_labels.csv')).open(newline='') as handle:
                rows = list(csv.DictReader(handle))
            assert features.shape == (len(rows), 955) and features.dtype == np.float32
            indices = [i for i, row in enumerate(rows) if row['purpose'] == 'sampled']
            sampled = [rows[i] for i in indices]
            assert len(sampled) % 4 == 0
            for i in range(0, len(sampled), 4):
                group = sampled[i:i+4]
                assert [int(row['kind']) for row in group] == [0, 1, 2, 3]
                assert len({row['frame'] for row in group}) == 1
                assert all(row['partition'] == partition and row['sequence'] == name for row in group)
                assert len({row['primary_iou'] for row in group}) == 1
            target = data[partition]
            count = len(sampled) // 4
            target['features'].append(features[indices].reshape(count, 4, 955))
            target['gains'].append(np.asarray([float(row['gain']) for row in sampled], np.float32).reshape(count, 4))
            target['recorded'].append(np.asarray([float(row['predicted_gain']) for row in sampled], np.float32).reshape(count, 4))
            target['primary'].append(np.asarray([float(sampled[i]['primary_iou']) for i in range(0, len(sampled), 4)], np.float32))
            target['sequences'].extend([name] * count)
    for partition, values in data.items():
        for key in ('features', 'gains', 'recorded', 'primary'):
            values[key] = np.concatenate(values[key])
            assert np.isfinite(values[key]).all()
    assert len(seen) == plan['expected_sequences']
    assert seen == set(plan['fit_sequences'] + plan['holdout_sequences'])
    assert set(data['fit']['sequences']) == set(plan['fit_sequences'])
    assert set(data['holdout']['sequences']) == set(plan['holdout_sequences'])
    assert not set(data['fit']['sequences']) & set(data['holdout']['sequences'])
    return data, input_hashes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--task', required=True)
    args = parser.parse_args()
    path = ROOT / args.plan
    plan = json.loads(path.read_text())
    assert plan['mode'] in ('smoke', 'formal') and plan['launch_allowed'] is True
    assert plan['test_eligible'] is False
    check_files(ROOT, plan['source_sha256'])
    review_path = ROOT / plan['source_review']
    assert sha(review_path) == plan['source_review_sha256']
    review = json.loads(review_path.read_text())
    assert review['status'] == 'SOURCE_PASS'
    assert review['source_sha256']['scripts/train_refreshed_query_gain.py'] == plan['source_sha256']['scripts/train_refreshed_query_gain.py']
    native_path = ROOT / 'runs/pipeline_tasks' / args.task / 'status.json'
    native = json.loads(native_path.read_text())
    assert native['state'] == 'RUNNING' and native['pid'] == os.getppid()
    assert native['command'] == [sys.executable] + sys.argv
    assert native['physical_gpu'] == os.environ['CUDA_VISIBLE_DEVICES']
    assert os.environ['CUDA_DEVICE_ORDER'] == 'PCI_BUS_ID'
    parent, _, _ = evaluation_plan(ROOT, plan['parent_evaluation_plan'])
    assert parent['checkpoint_sha256'] == plan['parent_checkpoint_sha256']
    check_files(ROOT, plan['cache_policy_inputs_sha256'])
    manifest_path = ROOT / plan['training_manifest']
    assert sha(manifest_path) == plan['training_manifest_sha256']
    manifest = json.loads(manifest_path.read_text())
    selected_names = set(plan['fit_sequences'] + plan['holdout_sequences'])
    assert len(selected_names) == plan['expected_sequences']
    assert selected_names <= {row['name'] for row in manifest['train']}
    assert not selected_names & ({row['name'] for row in manifest['dev']} | set(manifest['test_sequences']))
    overlay = torch.load(ROOT / plan['cache_policy_overlay'], map_location='cpu')
    assert sha(ROOT / plan['cache_policy_overlay']) == plan['cache_checkpoint_sha256']
    assert overlay['stage'] == 'cached_query_gain' and overlay['epoch'] == 6
    assert overlay['config']['parent_checkpoint_sha256'] == parent['checkpoint_sha256']
    behavior_native = json.loads((ROOT / plan['cache_policy_native']).read_text())
    behavior_result = json.loads((ROOT / plan['cache_policy_result']).read_text())
    assert behavior_native['state'] == 'COMPLETE' and behavior_native['exit_code'] == 0
    assert behavior_native['command'] == overlay['config']['native_command']
    assert behavior_result['status'] == 'TRAIN_COMPLETE_PENDING_DEV' and behavior_result['strict_reload_bit_equal']
    assert dict(epoch=6, checkpoint=plan['cache_policy_overlay'],
                sha256=plan['cache_checkpoint_sha256']) in behavior_result['retained']
    if plan['mode'] == 'formal':
        assert plan['formal_launch_allowed'] is True
        smoke_path = ROOT / plan['smoke_result']
        assert sha(smoke_path) == plan['smoke_result_sha256']
        smoke = json.loads(smoke_path.read_text())
        assert smoke['status'] == 'SANITY_PASS' and smoke['strict_reload_bit_equal']
        assert smoke['changed_state_tensors'] == ['query_gain.0.weight', 'query_gain.0.bias', 'query_gain.2.weight', 'query_gain.2.bias']
        smoke_config = json.loads((smoke_path.parent / 'config.json').read_text())
        smoke_native = json.loads((ROOT / 'runs/pipeline_tasks' / smoke_config['task'] / 'status.json').read_text())
        assert smoke_native['state'] == 'COMPLETE' and smoke_native['exit_code'] == 0
        assert smoke_native['command'] == smoke_config['native_command']
        for key in ('source_sha256', 'parent_checkpoint_sha256', 'cache_checkpoint_sha256',
                    'caches', 'fit_sequences', 'holdout_sequences', 'variant', 'batch_states', 'lr', 'seed'):
            assert smoke_config['plan'][key] == plan[key]
    data, input_hashes = load_caches(plan)
    input_hashes.update(plan['cache_policy_inputs_sha256'])
    input_hashes[plan['training_manifest']] = plan['training_manifest_sha256']
    torch.manual_seed(plan['seed'])
    np.random.seed(plan['seed'])
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    model = build_model('e1', ROOT / parent['factory_checkpoint'])
    parent_net = torch.load(ROOT / parent['checkpoint'], map_location='cpu')['net']
    model.load_state_dict(parent_net, strict=True)
    freeze(model)
    head = copy.deepcopy(model.query_gain)
    for parameter in head.parameters():
        parameter.requires_grad_(True)
    assert sum(p.numel() for p in head.parameters()) == 61249
    head.eval()
    tensors = {part: torch.from_numpy(values['features']).cuda() for part, values in data.items()}
    gains = torch.from_numpy(data['fit']['gains']).cuda()
    cost = np.asarray([193/385, 193/385, 193/385, 1], np.float32) * .15
    cost_tensor = torch.from_numpy(cost).cuda()

    @torch.no_grad()
    def measure(partition):
        prediction = head(tensors[partition]).squeeze(-1).cpu().numpy()
        values = data[partition]
        return diagnostics(prediction, values['gains'], values['primary'], cost, values['sequences']), prediction

    baseline, max_errors = {}, {}
    behavior_head = copy.deepcopy(head)
    behavior_head.load_state_dict(overlay['head'], strict=True)
    for part in data:
        baseline[part], predicted = measure(part)
        with torch.no_grad():
            recorded = behavior_head(tensors[part]).squeeze(-1).cpu().numpy()
        max_errors[part] = float(np.abs(recorded - data[part]['recorded']).max())
        assert max_errors[part] < 1e-5
    del behavior_head, overlay
    if plan['mode'] == 'formal':
        for part in data:
            assert baseline[part]['sampled_recoverable_states'] >= plan['minimum_recovery_states'][part]
            assert baseline[part]['recovery_sequences'] >= plan['minimum_recovery_sequences'][part]
    directory = ROOT / 'runs' / args.run
    directory.mkdir(parents=True, exist_ok=False)
    config = dict(plan=plan, plan_sha256=sha(path), native_command=native['command'], task=args.task,
        parent_checkpoint_sha256=parent['checkpoint_sha256'], input_sha256=input_hashes,
        baseline=baseline, cache_policy_prediction_max_abs_error=max_errors,
        sampling='Uniform over states, four equally weighted query families; selected_extra excluded. No candidate-population reweighting.',
        scope='Same parent initialization; only TRAIN visited-state cache changes. E6-head outputs verify recorded features, not initialization. Component holdout was seen by parent training; not benchmark accuracy.')
    (directory / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    optimizer = torch.optim.AdamW(head.parameters(), lr=plan['lr'], weight_decay=1e-4)
    rng = np.random.RandomState(plan['seed'])
    history, retained = [], []
    best, stale, updates = float('inf'), 0, 0
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, plan['max_epochs'] + 1):
        order = rng.permutation(len(gains))
        losses = []
        for start in range(0, len(order), plan['batch_states']):
            indices = torch.as_tensor(order[start:start + plan['batch_states']], device='cuda')
            prediction = head(tensors['fit'][indices]).squeeze(-1)
            loss = objective(prediction, gains[indices], cost_tensor, plan['variant'])
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
            optimizer.step()
            losses.append(loss.item())
            updates += 1
        measurements = {part: measure(part)[0] for part in data}
        row = dict(epoch=epoch, updates=updates, mean_train_loss=float(np.mean(losses)), **measurements)
        history.append(row)
        checkpoint = directory / ('epoch_%03d.pth' % epoch)
        state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        torch.save(dict(stage='cached_query_gain', epoch=epoch, head=state, optimizer=optimizer.state_dict(), config=config), checkpoint)
        probe = copy.deepcopy(head)
        probe.load_state_dict(torch.load(checkpoint, map_location='cpu')['head'], strict=True)
        with torch.no_grad():
            assert torch.equal(head(tensors['fit'][:2]), probe(tensors['fit'][:2]))
        del probe
        retained.append(dict(epoch=epoch, checkpoint=str(checkpoint.relative_to(ROOT)), sha256=sha(checkpoint)))
        if len(retained) > 5:
            old = retained.pop(0)
            owned = ROOT / old['checkpoint']
            assert owned.parent == directory and owned.name == 'epoch_%03d.pth' % old['epoch']
            assert sha(owned) == old['sha256']
            owned.unlink()
        score = row['holdout']['mean_sampled_utility_regret']
        improved = score < best - plan['min_delta']
        if improved:
            best, stale = score, 0
        else:
            stale += 1
        (directory / 'progress.json').write_text(json.dumps(history, indent=2) + '\n')
        print(json.dumps(row), flush=True)
        if epoch >= plan['min_epochs'] and stale >= plan['patience']:
            break
    model.query_gain.load_state_dict(head.state_dict(), strict=True)
    changed = [k for k, value in model.state_dict().items() if not torch.equal(value.cpu(), parent_net[k])]
    assert changed == ['query_gain.0.weight', 'query_gain.0.bias', 'query_gain.2.weight', 'query_gain.2.bias']
    assert all(p.grad is None for p in model.parameters())
    check_files(ROOT, input_hashes)
    check_files(ROOT, dict(parent['source_sha256'], **plan['source_sha256']))
    assert sha(path) == config['plan_sha256'] and sha(ROOT / parent['checkpoint']) == parent['checkpoint_sha256']
    result = dict(status='SANITY_PASS' if plan['mode'] == 'smoke' else 'TRAIN_COMPLETE_PENDING_DEV',
        epochs=len(history), updates=updates, trainable_parameters=61249,
        total_parameters=sum(p.numel() for p in model.parameters()), changed_state_tensors=changed,
        strict_reload_bit_equal=True, retained=retained, elapsed_seconds=time.time()-started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        plan_sha256=config['plan_sha256'], baseline=baseline, final=history[-1], test_eligible=False,
        scope='Only query_gain changed from the original parent. Retain last five weights and evaluate each on full dev; test remains frozen-selection only.')
    (directory / 'COMPLETE.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

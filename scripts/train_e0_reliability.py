"""Offline M1 training from complete protected-E0 TRAIN observations."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from benchmark_inputs import sha
from dual_rgbt.e0_reliability import E0Reliability, STATE_NAMES, localization_targets


def load_collection(root, plan):
    """Accept eight complete paired shards, never the three-prefix smoke."""
    assert len(plan['caches']) == 8
    fit, held = set(plan['fit_sequences']), set(plan['holdout_sequences'])
    assert len(fit) == 706 and len(held) == 176 and fit.isdisjoint(held)
    seen, shards, used = set(), set(), {}
    data = {part: dict(features=[], iou=[]) for part in ('fit', 'holdout')}
    for entry in plan['caches']:
        directory = root / 'runs' / entry['run']
        receipt_path = directory / 'COMPLETE.json'
        assert sha(receipt_path) == entry['receipt_sha256']
        receipt = json.loads(receipt_path.read_bytes())
        config = json.loads((directory / 'config.json').read_bytes())
        native_path = root / 'runs/pipeline_tasks' / entry['task'] / 'status.json'
        assert sha(native_path) == entry['native_sha256']
        native = json.loads(native_path.read_bytes())
        assert native['state'] == 'COMPLETE' and native['exit_code'] == 0
        assert native['command'] == config['native_command']
        assert receipt['task'] == config['task'] == entry['task']
        assert config['run'] == entry['run'] and config['split'] == receipt['split'] == 'train'
        assert receipt['status'] == 'PAIRED_TRAIN_CACHE_COMPLETE'
        assert receipt['checkpoint_sha256'] == plan['checkpoint_sha256']
        assert config['plan']['checkpoint_sha256'] == plan['checkpoint_sha256']
        assert set(config['plan']['fit_sequences']) == fit
        assert set(config['plan']['holdout_sequences']) == held
        assert config['plan']['full_collection_allowed'] is True
        assert config['plan']['cost_weight'] == 0.
        assert config['plan']['matmul_tf32'] is config['plan']['cudnn_tf32'] is False
        manifest_relative = config['plan']['manifest']
        manifest_path = root / manifest_relative
        assert sha(manifest_path) == config['plan']['input_sha256'][manifest_relative]
        used[manifest_relative] = sha(manifest_path)
        manifest = json.loads(manifest_path.read_bytes())
        sequences = {row['name']: row for row in manifest['train']}
        assert len(sequences) == 882 and set(sequences) == fit | held
        assert set(sequences).isdisjoint(set(manifest['test_sequences']) |
                                         {row['name'] for row in manifest['dev']})
        assert config['selected_sequences'] == [row['name'] for row in manifest['train']][config['shard']::8]
        assert receipt['optimizer_updates'] == receipt['backward_calls'] == 0
        assert receipt['frozen_state_tensors'] == 265
        assert receipt['all_online_states_bit_equal'] is True
        assert receipt['feature_width'] == 914
        required = {'config.json', 'progress.json'}
        required.update(name + suffix for name in config['selected_sequences']
                        for suffix in ('_features.npy', '_labels.csv'))
        assert required <= set(receipt['artifacts_sha256'])
        assert config['shard'] not in shards
        shards.add(config['shard'])
        for relative, digest in receipt['artifacts_sha256'].items():
            path = directory / relative
            assert sha(path) == digest
            used[path.relative_to(root).as_posix()] = digest
        for path in (receipt_path, native_path):
            used[path.relative_to(root).as_posix()] = sha(path)
        progress = json.loads((directory / 'progress.json').read_bytes())
        assert [row['sequence'] for row in progress] == config['selected_sequences']
        by_name = {row['sequence']: row for row in progress}
        states = 0
        for name in config['selected_sequences']:
            assert name in fit | held and name not in seen
            seen.add(name)
            partition = 'fit' if name in fit else 'holdout'
            features = np.load(directory / (name + '_features.npy'), allow_pickle=False)
            with (directory / (name + '_labels.csv')).open(newline='') as handle:
                rows = list(csv.DictReader(handle))
            assert features.shape == (len(rows), 914) and len(rows) > 0
            assert features.dtype == np.float32 and np.isfinite(features).all()
            assert all(row['sequence'] == name and row['partition'] == partition for row in rows)
            frames = [int(row['frame']) for row in rows]
            sequence = sequences[name]
            expected_frames = [frame for frame in sequence['valid_frames'] if frame > 0 and frame % 5 == 0]
            assert frames == expected_frames
            assert by_name[name]['frames'] == len(sequence['rgb'])
            assert by_name[name]['matched_tracked_frames'] == len(sequence['rgb']) - 1
            assert by_name[name]['all_online_states_bit_equal'] is True
            assert by_name[name]['feature_states'] == len(rows)
            iou = np.asarray([float(row['final_iou']) for row in rows], np.float32)
            assert np.isfinite(iou).all() and ((iou >= 0) & (iou <= 1)).all()
            data[partition]['features'].append(features)
            data[partition]['iou'].append(iou)
            states += len(rows)
        assert states == receipt['feature_states']
        assert len(config['selected_sequences']) == receipt['sequences']
        assert receipt['frames'] == sum(row['frames'] for row in progress)
        assert receipt['tracked_frames'] == sum(row['matched_tracked_frames'] for row in progress)
    assert shards == set(range(8)) and seen == fit | held
    for part in data.values():
        part['features'] = torch.from_numpy(np.concatenate(part['features']))
        part['iou'] = torch.from_numpy(np.concatenate(part['iou']))
        assert (torch.bincount(localization_targets(part['iou']), minlength=3) > 0).all()
    return data, used


def fit_normalization(features):
    mean = features.mean(0)
    scale = features.std(0, unbiased=False)
    # Modality/coverage channels can be constant on fit; center without scaling.
    scale[scale == 0] = 1
    return mean, scale


@torch.no_grad()
def metrics(head, features, iou, batch):
    target = localization_targets(iou)
    confusion = torch.zeros(3, 3, dtype=torch.long)
    loss_sum = 0.
    for start in range(0, len(features), batch):
        labels = target[start:start + batch]
        logits = head(features[start:start + batch])
        loss_sum += F.cross_entropy(logits, labels, reduction='sum').item()
        confusion += torch.bincount(
            (labels * 3 + logits.argmax(1)).cpu(), minlength=9).reshape(3, 3)
    counts = confusion.sum(1)
    assert (counts > 0).all()
    return dict(states=len(target), cross_entropy=loss_sum / len(target),
        class_counts=dict(zip(STATE_NAMES, counts.tolist())),
        confusion_true_rows_predicted_columns=confusion.tolist(),
        macro_recall=float((confusion.diag().double() / counts).mean()),
        lost_recall=float(confusion[2, 2].double() / counts[2]),
        reliable_interrupted_fraction=float(1 - confusion[0, 0].double() / counts[0]))


def train_head(data, plan, output, plan_sha256, used):
    """Train only the standalone head; main admits real collection inputs."""
    torch.manual_seed(plan['seed'])
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    mean, scale = fit_normalization(data['fit']['features'])
    for part in data.values():
        part['features'] = ((part['features'] - mean) / scale).to(plan['device'])
        part['iou'] = part['iou'].to(plan['device'])
    head = E0Reliability().to(plan['device'])
    optimizer = torch.optim.AdamW(head.parameters(), lr=plan['lr'], weight_decay=plan['weight_decay'])
    output.mkdir(parents=True, exist_ok=False)
    (output / 'config.json').write_text(json.dumps(
        dict(plan=plan, plan_sha256=plan_sha256, inputs_sha256=used), indent=2) + '\n')
    history, best, updates = [], float('-inf'), 0
    batch = plan['batch_states']
    for epoch in range(1, plan['epochs'] + 1):
        head.train()
        order = torch.randperm(len(data['fit']['iou']), device=plan['device'])
        for start in range(0, len(order), batch):
            index = order[start:start + batch]
            optimizer.zero_grad()
            logits = head(data['fit']['features'][index])
            loss = F.cross_entropy(logits, localization_targets(data['fit']['iou'][index]))
            loss.backward()
            optimizer.step()
            updates += 1
        head.eval()
        result = {part: metrics(head, values['features'], values['iou'], batch)
                  for part, values in data.items()}
        history.append(dict(epoch=epoch, optimizer_updates=updates, **result))
        checkpoint = dict(net={key: value.detach().cpu().clone() for key, value in head.state_dict().items()},
            feature_mean=mean, feature_scale=scale, epoch=epoch, metrics=result,
            plan_sha256=plan_sha256, protected_e0_checkpoint_sha256=plan['checkpoint_sha256'],
            selection_partition='train_component_holdout', total_parameters=117507)
        torch.save(checkpoint, output / ('epoch_%03d.pth' % epoch))
        torch.save(checkpoint, output / 'latest.pth')
        if result['holdout']['macro_recall'] > best:
            best = result['holdout']['macro_recall']
            torch.save(checkpoint, output / 'best.pth')
        if epoch > 5:
            (output / ('epoch_%03d.pth' % (epoch - 5))).unlink()
        (output / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
        print(json.dumps(history[-1]), flush=True)
    restored = torch.load(output / 'latest.pth', map_location='cpu')
    reloaded = E0Reliability().to(plan['device']).eval()
    reloaded.load_state_dict(restored['net'], strict=True)
    assert torch.equal(restored['feature_mean'], mean)
    assert torch.equal(restored['feature_scale'], scale)
    with torch.no_grad():
        probe = data['holdout']['features'][:batch]
        assert torch.equal(head(probe), reloaded(probe))
    receipt = dict(status='OFFLINE_M1_TRAINING_COMPLETE', split='train',
        epochs=plan['epochs'], optimizer_updates=updates, total_parameters=117507,
        strict_reload_bit_equal=True,
        protected_e0_loaded=False, online_policy_changed=False, test_eligible=False,
        best_holdout_macro_recall=best,
        scope='Localization-overlap reliability only, not target identity or recovery. '
              'Fit-only normalization; upstream E0 saw component holdout. '
              'No benchmark improvement or independent generalization claim.',
        artifacts_sha256={p.name: sha(p) for p in output.iterdir() if p.is_file()})
    (output / 'COMPLETE.json').write_text(json.dumps(receipt, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    path = ROOT / args.plan
    plan = json.loads(path.read_bytes())
    assert plan['mode'] == 'protected_e0_offline_m1_training' and plan['launch_allowed'] is True
    assert plan['split'] == 'train' and plan['test_eligible'] is False
    assert plan['checkpoint_sha256'] == 'ee60704e7fe1e281584430332dea7807a335bd4d723a769e4783217b0ac6f56c'
    assert plan['epochs'] >= 5 and plan['batch_states'] > 0 and plan['lr'] > 0
    for relative, digest in plan['source_sha256'].items():
        assert sha(ROOT / relative) == digest
    assert plan['source_sha256']['scripts/train_e0_reliability.py'] == sha(Path(__file__))
    review_path = ROOT / plan['source_review']
    assert sha(review_path) == plan['source_review_sha256']
    review = json.loads(review_path.read_bytes())
    assert review['status'] == 'SOURCE_PASS' and review['blocking_issues'] == 0
    assert review['source_sha256'] == plan['source_sha256']
    data, used = load_collection(ROOT, plan)
    train_head(data, plan, ROOT / args.output, sha(path), used)


if __name__ == '__main__':
    main()

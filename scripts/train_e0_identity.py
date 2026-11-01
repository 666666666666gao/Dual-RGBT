"""Train only M2's identity head from fully verified executed TRAIN candidates."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'tools')]
from benchmark_inputs import sha
from dual_rgbt.e0_identity import E0CandidateIdentity
from collect_e0_identity_train import check_admission, sampled_frames
from verify_e0_identity_collection import image_sizes, load_collection, verify_arrays

import numpy as np


def multi_positive_loss(logits, labels):
    """Ignore unknown candidates and reward the total probability of all positives."""
    assert logits.shape == labels.shape and labels.ndim == 2
    assert ((labels == -1) | (labels == 0) | (labels == 1)).all()
    assert (labels == 1).any(1).all()
    assert torch.isfinite(logits[labels == 1]).all()
    observed = logits.masked_fill(labels == -1, float('-inf'))
    positive = logits.masked_fill(labels != 1, float('-inf'))
    return (torch.logsumexp(observed, 1) - torch.logsumexp(positive, 1)).mean()


@torch.no_grad()
def metrics(head, data, batch):
    loss_sum, supervised_count, positive_picks = 0., 0, 0
    ignored_picks, negative_picks, no_match_true, no_match_correct = 0, 0, 0, 0
    candidate_positive, candidate_correct = 0, 0
    for start in range(0, len(data['labels']), batch):
        values = {key: value[start:start + batch] for key, value in data.items()}
        labels = values['labels']
        logits = head(values['anchor'], values['features'], values['coverage'])
        supervised = (labels == 1).any(1)
        count = int(supervised.sum().item())
        if count:
            loss_sum += multi_positive_loss(logits[supervised], labels[supervised]).item() * count
        chosen = logits.argmax(1)
        chosen_label = labels.gather(1, chosen[:, None]).squeeze(1)
        positive_picks += int(((chosen_label == 1) & supervised).sum().item())
        ignored_picks += int(((chosen_label == -1) & supervised).sum().item())
        negative_picks += int(((chosen_label == 0) & supervised).sum().item())
        no_match = labels[:, 0] == 1
        no_match_true += int(no_match.sum().item())
        no_match_correct += int((no_match & (chosen == 0)).sum().item())
        has_candidate = (labels[:, 1:] == 1).any(1)
        candidate_positive += int(has_candidate.sum().item())
        candidate_correct += int((has_candidate & (chosen_label == 1) & (chosen > 0)).sum().item())
        supervised_count += count
    assert supervised_count > 0
    assert positive_picks + ignored_picks + negative_picks == supervised_count
    return dict(candidate_sets=len(data['labels']), supervised_sets=supervised_count,
        ambiguous_sets=len(data['labels']) - supervised_count,
        multi_positive_loss=loss_sum / supervised_count,
        supervised_top1_positive_fraction=positive_picks / supervised_count,
        supervised_ignored_picks=ignored_picks, supervised_negative_picks=negative_picks,
        sets_with_positive_candidate=candidate_positive,
        correct_candidate_picks=candidate_correct,
        no_match_sets=no_match_true, correct_no_match_picks=no_match_correct)


def train_head(data, plan, output, plan_sha, acceptance_sha):
    torch.manual_seed(plan['seed'])
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    data = {part: {key: value.to(plan['device']) for key, value in fields.items()}
            for part, fields in data.items()}
    fit = data['fit']
    eligible = torch.nonzero((fit['labels'] == 1).any(1)).flatten()
    assert len(eligible) > 0
    head = E0CandidateIdentity().to(plan['device'])
    optimizer = torch.optim.AdamW(head.parameters(), lr=plan['lr'],
                                  weight_decay=plan['weight_decay'])
    output.mkdir(parents=True, exist_ok=False)
    (output / 'config.json').write_text(json.dumps(dict(
        plan=plan, plan_sha256=plan_sha, acceptance_sha256=acceptance_sha), indent=2) + '\n')
    history, updates, best, started = [], 0, float('-inf'), time.time()
    batch = plan['batch_sets']
    for epoch in range(1, plan['epochs'] + 1):
        epoch_started = time.time()
        head.train()
        order = eligible[torch.randperm(len(eligible), device=plan['device'])]
        for start in range(0, len(order), batch):
            index = order[start:start + batch]
            optimizer.zero_grad()
            logits = head(fit['anchor'][index], fit['features'][index], fit['coverage'][index])
            loss = multi_positive_loss(logits, fit['labels'][index])
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
            updates += 1
        head.eval()
        result = {part: metrics(head, fields, batch) for part, fields in data.items()}
        history.append(dict(epoch=epoch, optimizer_updates=updates,
                            epoch_seconds=time.time() - epoch_started, **result))
        checkpoint = dict(net={key: value.detach().cpu().clone()
                               for key, value in head.state_dict().items()},
            epoch=epoch, metrics=result, plan_sha256=plan_sha,
            acceptance_sha256=acceptance_sha,
            protected_e0_checkpoint_sha256=plan['checkpoint_sha256'],
            selection_partition='train_component_holdout',
            total_parameters=sum(p.numel() for p in head.parameters()))
        torch.save(checkpoint, output / ('epoch_%03d.pth' % epoch))
        torch.save(checkpoint, output / 'latest.pth')
        score = result['holdout']['supervised_top1_positive_fraction']
        if score > best:
            best = score
            torch.save(checkpoint, output / 'best.pth')
        if epoch > 5:
            (output / ('epoch_%03d.pth' % (epoch - 5))).unlink()
        (output / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
        print(json.dumps(history[-1]), flush=True)
    restored = E0CandidateIdentity().to(plan['device']).eval()
    restored.load_state_dict(torch.load(output / 'latest.pth', map_location='cpu')['net'], strict=True)
    with torch.no_grad():
        probe = {key: value[:batch] for key, value in data['holdout'].items()}
        assert torch.equal(head(probe['anchor'], probe['features'], probe['coverage']),
                           restored(probe['anchor'], probe['features'], probe['coverage']))
    receipt = dict(status='OFFLINE_M2_IDENTITY_TRAINING_COMPLETE', split='train',
        plan_sha256=plan_sha, acceptance_sha256=acceptance_sha,
        epochs=plan['epochs'], optimizer_updates=updates,
        total_parameters=sum(p.numel() for p in head.parameters()),
        best_component_holdout_top1_positive_fraction=best,
        strict_reload_bit_equal=True, protected_e0_loaded=False,
        online_policy_changed=False, test_eligible=False,
        scientific_result_audit_accepted=False, wall_seconds=time.time() - started,
        scope='Standalone overlap-supervised candidate matching. Ignored labels '
              'excluded from loss; multiple positives supported. No E0 update, '
              'online recovery, semantic identity proof or benchmark result. '
              'Upstream E0 saw the component holdout.',
        artifacts_sha256={p.name: sha(p) for p in output.iterdir() if p.is_file()})
    (output / 'COMPLETE.json').write_text(json.dumps(receipt, indent=2) + '\n')


def real_update_smoke(root, plan, output, plan_sha):
    """A few updates on already collected TRAIN prefixes, never a benchmark."""
    path = root / plan['collection_plan']
    assert sha(path) == plan['collection_plan_sha256']
    collection = json.loads(path.read_bytes())
    check_admission(root, collection)
    directory = root / 'runs' / collection['smoke_run']
    manifest = json.loads((root / collection['manifest']).read_bytes())
    sequences = {row['name']: row for row in manifest['train']}
    config = json.loads((directory / 'config.json').read_bytes())
    fields = {key: [] for key in ('anchor', 'features', 'coverage', 'labels')}
    for name in config['selected_sequences']:
        with np.load(directory / (name + '.npz'), allow_pickle=False) as arrays:
            sizes = image_sizes(sequences[name], sampled_frames(sequences[name], 31))
            result = verify_arrays(arrays, sequences[name], 31, sizes)
            fields['anchor'].append(np.repeat(arrays['anchor'][None], result['candidate_sets'], axis=0))
            for key in ('features', 'coverage', 'labels'):
                fields[key].append(arrays[key].copy())
    data = {key: torch.from_numpy(np.concatenate(values)).to(plan['device'])
            for key, values in fields.items()}
    assert len(data['labels']) == 6
    eligible = (data['labels'] == 1).any(1)
    assert eligible.any()
    torch.manual_seed(plan['seed'])
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    head = E0CandidateIdentity().to(plan['device'])
    before = {key: value.detach().clone() for key, value in head.state_dict().items()}
    optimizer = torch.optim.AdamW(head.parameters(), lr=plan['lr'],
                                  weight_decay=plan['weight_decay'])
    for _ in range(5):
        optimizer.zero_grad()
        logits = head(data['anchor'], data['features'], data['coverage'])
        loss = multi_positive_loss(logits[eligible], data['labels'][eligible])
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
        optimizer.step()
    assert any(not torch.equal(before[key], value) for key, value in head.state_dict().items())
    output.mkdir(parents=True, exist_ok=False)
    (output / 'config.json').write_text(json.dumps(dict(plan=plan, plan_sha256=plan_sha), indent=2) + '\n')
    torch.save(dict(net={key: value.detach().cpu().clone()
                         for key, value in head.state_dict().items()}), output / 'smoke.pth')
    restored = E0CandidateIdentity().to(plan['device']).eval()
    restored.load_state_dict(torch.load(output / 'smoke.pth', map_location='cpu')['net'], strict=True)
    head.eval()
    with torch.no_grad():
        assert torch.equal(head(data['anchor'], data['features'], data['coverage']),
                           restored(data['anchor'], data['features'], data['coverage']))
    receipt = dict(status='M2_IDENTITY_REAL_UPDATE_SMOKE_COMPLETE', smoke_only=True,
        split='train', test_eligible=False, plan_sha256=plan_sha,
        candidate_sets=6, supervised_sets=int(eligible.sum().item()),
        optimizer_updates=5, strict_reload_bit_equal=True, parameters_changed=True,
        total_parameters=sum(p.numel() for p in head.parameters()),
        protected_e0_loaded=False, online_policy_changed=False,
        scientific_result_audit_accepted=False,
        metrics=metrics(head, data, 6),
        artifacts_sha256={p.name: sha(p) for p in output.iterdir() if p.is_file()},
        scope='Five standalone identity-head updates on six real TRAIN prefix sets. '
              'Not full-collection acceptance, a retained scientific head or a benchmark.')
    (output / 'COMPLETE.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)


def verify_update_smoke(root, plan):
    directory = root / plan['update_smoke_output']
    receipt_path = directory / 'COMPLETE.json'
    native_path = root / 'runs/pipeline_tasks' / plan['update_smoke_task'] / 'status.json'
    assert sha(receipt_path) == plan['update_smoke_receipt_sha256']
    assert sha(native_path) == plan['update_smoke_native_sha256']
    receipt = json.loads(receipt_path.read_bytes())
    native = json.loads(native_path.read_bytes())
    config = json.loads((directory / 'config.json').read_bytes())
    smoke_plan_path = root / plan['update_smoke_plan']
    assert sha(smoke_plan_path) == plan['update_smoke_plan_sha256'] == config['plan_sha256']
    smoke_plan = json.loads(smoke_plan_path.read_bytes())
    assert smoke_plan == config['plan']
    assert smoke_plan['phase'] == 'smoke' and smoke_plan['launch_allowed'] is True
    assert smoke_plan['source_sha256'] == plan['source_sha256']
    assert smoke_plan['source_review_sha256'] == plan['source_review_sha256']
    assert smoke_plan['collection_plan_sha256'] == plan['collection_plan_sha256']
    assert smoke_plan['task'] == native['task'] == plan['update_smoke_task']
    assert smoke_plan['output'] == plan['update_smoke_output']
    assert native['state'] == 'COMPLETE' and native['exit_code'] == 0
    assert native['physical_gpu'] == smoke_plan['assigned_gpu']
    assert native['command'] == [str(root / 'conda-envs/dual_rgbt/bin/python'),
        str(root / 'scripts/train_e0_identity.py'), '--plan', plan['update_smoke_plan'],
        '--output', plan['update_smoke_output'], '--task', plan['update_smoke_task']]
    assert receipt['status'] == 'M2_IDENTITY_REAL_UPDATE_SMOKE_COMPLETE'
    assert receipt['plan_sha256'] == config['plan_sha256']
    assert receipt['smoke_only'] is receipt['strict_reload_bit_equal'] is receipt['parameters_changed'] is True
    assert receipt['optimizer_updates'] == 5 and receipt['candidate_sets'] == 6
    assert set(receipt['artifacts_sha256']) == {'config.json', 'smoke.pth'}
    for relative, digest in receipt['artifacts_sha256'].items():
        assert sha(directory / relative) == digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--task', required=True)
    args = parser.parse_args()
    path = ROOT / args.plan
    plan_sha = sha(path)
    plan = json.loads(path.read_bytes())
    assert plan['mode'] == 'protected_e0_offline_m2_identity_training'
    assert plan['phase'] in ('smoke', 'full')
    assert plan['launch_allowed'] is True and plan['split'] == 'train'
    assert plan['test_eligible'] is False and plan['device'] == 'cuda'
    assert plan['task'] == args.task and plan['output'] == args.output
    assert plan['batch_sets'] > 0 and plan['lr'] > 0
    assert plan['target_root'] == str(ROOT) and plan['target_host'] == '2025'
    assert plan['checkpoint_sha256'] == 'ee60704e7fe1e281584430332dea7807a335bd4d723a769e4783217b0ac6f56c'
    for relative, digest in plan['source_sha256'].items():
        assert sha(ROOT / relative) == digest
    assert plan['source_sha256']['scripts/train_e0_identity.py'] == sha(Path(__file__))
    review_path = ROOT / plan['source_review']
    assert sha(review_path) == plan['source_review_sha256']
    review = json.loads(review_path.read_bytes())
    assert review['stage'] == 'M2_identity_training'
    assert review['status'] == 'SOURCE_PASS' and review['blocking_issues'] == 0
    assert review['source_sha256'] == plan['source_sha256']
    draft_path = ROOT / plan['reviewed_draft']
    assert sha(draft_path) == review['reviewed_drafts'][plan['reviewed_draft']]
    draft = json.loads(draft_path.read_bytes())
    assert draft['launch_allowed'] is False and set(draft) == set(plan)
    variable = {'launch_allowed', 'source_review_sha256', 'collection_acceptance_sha256',
                'update_smoke_receipt_sha256', 'update_smoke_native_sha256',
                'update_smoke_plan_sha256'}
    assert all(plan[key] == value for key, value in draft.items() if key not in variable)
    native = json.loads((ROOT / 'runs/pipeline_tasks' / args.task / 'status.json').read_bytes())
    assert native['state'] == 'RUNNING' and native['pid'] == os.getppid()
    assert native['task'] == args.task
    assert native['physical_gpu'] == plan['assigned_gpu'] == os.environ['CUDA_VISIBLE_DEVICES']
    assert native['command'] == [sys.executable] + sys.argv
    if plan['phase'] == 'smoke':
        assert plan['epochs'] == 0 and plan['batch_sets'] == 6
        real_update_smoke(ROOT, plan, ROOT / args.output, plan_sha)
    else:
        assert plan['epochs'] >= 5
        verify_update_smoke(ROOT, plan)
        acceptance_path = ROOT / plan['collection_acceptance']
        assert sha(acceptance_path) == plan['collection_acceptance_sha256']
        acceptance = json.loads(acceptance_path.read_bytes())
        data, actual = load_collection(ROOT, plan['collection_plan'], plan['collection_plan_sha256'])
        assert acceptance == actual
        train_head(data, plan, ROOT / args.output, plan_sha, sha(acceptance_path))
    assert sha(path) == plan_sha
    for relative, digest in plan['source_sha256'].items():
        assert sha(ROOT / relative) == digest


if __name__ == '__main__':
    main()

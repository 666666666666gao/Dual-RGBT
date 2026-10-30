"""Adapt only the existing final size convolution; preserve model architecture."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dual_rgbt.data import LasHeRPairs
from dual_rgbt.losses import localization_loss
from dual_rgbt_global.candidate_policy import build_candidate_policy
from joint_evaluation_contract import evaluation_plan

PREFIX = 'executor.box_head.conv5_size.'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_epoch(run, state):
    path = run / ('epoch_%03d.pth' % state['epoch'])
    assert not path.exists()
    torch.save(state, path)
    checkpoints = sorted(run.glob('epoch_*.pth'))
    for old in checkpoints[:-5]:
        old.unlink()
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--sanity', action='store_true')
    args = parser.parse_args()
    plan = json.loads((ROOT / args.plan).read_text())
    assert sha(Path(__file__)) == plan['trainer_sha256']
    parent, _, _ = evaluation_plan(ROOT, plan['parent_evaluation_plan'])
    assert parent['checkpoint_sha256'] == plan['parent_checkpoint_sha256']
    manifest = ROOT / plan['manifest']
    assert sha(manifest) == plan['manifest_sha256']
    data = json.loads(manifest.read_text())
    assert len(data['train']) == 882 and len(data['dev']) == 97
    assert not {s['name'] for s in data['train']} & {s['name'] for s in data['dev']}
    assert not {s['name'] for s in data['train']} & set(data['test_sequences'])
    torch.manual_seed(plan['seed'])
    np.random.seed(plan['seed'])
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    run = ROOT / 'runs' / args.run
    run.mkdir(parents=True, exist_ok=False)
    config = dict(plan=plan, plan_sha256=sha(ROOT / args.plan), sanity=args.sanity)
    (run / 'config.json').write_text(json.dumps(config, indent=2))
    model = build_candidate_policy(ROOT / parent['factory_checkpoint']).cuda().eval()
    model.load_state_dict(torch.load(ROOT / parent['checkpoint'], map_location='cpu')['net'], strict=True)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(PREFIX))
    before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    params = [p for p in model.parameters() if p.requires_grad]
    assert len(params) == 2
    optimizer = torch.optim.AdamW(params, lr=plan['lr'], weight_decay=1e-4)
    batch_size = plan['batch_size']
    epochs = 1 if args.sanity else plan['epochs']
    samples = 4 * batch_size if args.sanity else plan['samples_per_epoch']
    assert samples % batch_size == 0
    torch.cuda.reset_peak_memory_stats()
    summaries = []
    start = time.time()
    for epoch in range(1, epochs + 1):
        dataset = LasHeRPairs(manifest, 'train', samples, plan['seed'] + (epoch - 1) * samples)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        losses = []
        for step, batch in enumerate(loader):
            templates, search, target = [batch[k].cuda() for k in ('templates', 'search', 'target')]
            modality = step % 2
            output = model.executor.primary(templates, search, modality)
            old_score, old_offset = output['score'].detach().clone(), output['offset'].detach().clone()
            loss, _ = localization_loss(output['box'], target)
            loss = loss.mean()
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
            optimizer.step()
            if args.sanity:
                with torch.no_grad():
                    updated = model.executor.primary(templates, search, modality)
                assert torch.equal(old_score, updated['score'])
                assert torch.equal(old_offset, updated['offset'])
            losses.append(loss.item())
            print(json.dumps(dict(epoch=epoch, step=step, loss=loss.item(), modality=modality)), flush=True)
        current = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        changed = [k for k in before if not torch.equal(before[k], current[k])]
        assert len(changed) == 2 and all(k.startswith(PREFIX) for k in changed)
        path = save_epoch(run, dict(net=current, epoch=epoch, optimizer=optimizer.state_dict(),
            stage='size_tail_adaptation', config=config))
        summaries.append(dict(epoch=epoch, mean_train_loss=float(np.mean(losses)),
            checkpoint=path.name, checkpoint_sha256=sha(path)))
        (run / 'epochs.json').write_text(json.dumps(summaries, indent=2))
    with torch.no_grad():
        expected = model.executor.primary(templates, search, modality)
        model.load_state_dict(torch.load(path, map_location='cpu')['net'], strict=True)
        actual = model.executor.primary(templates, search, modality)
        assert all(torch.equal(expected[k], actual[k]) for k in ('box', 'score', 'size', 'offset'))
    result = dict(status='SANITY_PASS' if args.sanity else 'TRAINING_COMPLETE_PENDING_DEV',
        epochs=epochs, updates=epochs * samples // batch_size, changed_state_tensors=changed,
        total_parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in params),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(), elapsed_seconds=time.time()-start,
        strict_reload_bit_equal=True, checkpoint_sha256=sha(path),
        scope='Train split only. Same-input score/offset invariance is not closed-loop route invariance. No dev/test metric claim.')
    assert sha(Path(__file__)) == plan['trainer_sha256']
    (run / 'COMPLETE.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

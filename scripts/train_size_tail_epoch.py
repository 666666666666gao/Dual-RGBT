"""One resumable size-tail epoch, with a native receipt for every checkpoint."""
import argparse
import json
import os
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
from joint_evaluation_contract import check_files, evaluation_plan
from train_size_tail import PREFIX, sha, save_epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--resume')
    parser.add_argument('--resume-sha256')
    parser.add_argument('--sanity', action='store_true')
    args = parser.parse_args()
    plan_path = ROOT / args.plan
    plan = json.loads(plan_path.read_text())
    check_files(ROOT, plan['source_sha256'])
    assert 1 <= args.epoch <= plan['epochs']
    native = json.loads((ROOT / 'runs/pipeline_tasks' / args.task / 'status.json').read_text())
    assert native['state'] == 'RUNNING' and native['pid'] == os.getppid()
    assert native['command'] == [sys.executable] + sys.argv
    assert native['physical_gpu'] == os.environ['CUDA_VISIBLE_DEVICES']
    if not args.sanity:
        assert plan['formal_launch_allowed'] is True
        gate = ROOT / plan['parent_control_score']
        assert sha(gate) == plan['parent_control_score_sha256']
        result = json.loads(gate.read_text())
        assert result['status'] == 'SCORED_COMPLETE' and result['evaluation_class'] == 'parent_control'
        assert (result['sequences'], result['frames']) == (97, 54646)
        assert result['checkpoint_sha256'] == plan['parent_checkpoint_sha256']
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
    model = build_candidate_policy(ROOT / parent['factory_checkpoint']).cuda().eval()
    parent_net = torch.load(ROOT / parent['checkpoint'], map_location='cpu')['net']
    model.load_state_dict(parent_net, strict=True)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(PREFIX))
    params = [p for p in model.parameters() if p.requires_grad]
    assert len(params) == 2 and sum(p.numel() for p in params) == 66
    optimizer = torch.optim.AdamW(params, lr=plan['lr'], weight_decay=1e-4)
    previous_updates = 0
    if args.epoch == 1:
        assert args.resume is None and args.resume_sha256 is None
    else:
        resume = ROOT / args.resume
        assert sha(resume) == args.resume_sha256
        previous = torch.load(resume, map_location='cpu')
        assert previous['epoch'] == args.epoch - 1 and previous['stage'] == 'size_tail_adaptation'
        assert previous['config']['plan_sha256'] == sha(plan_path)
        assert previous['config']['sanity'] == args.sanity
        previous_native = json.loads((ROOT / 'runs/pipeline_tasks' / previous['config']['task'] / 'status.json').read_text())
        assert previous_native['state'] == 'COMPLETE' and previous_native['exit_code'] == 0
        previous_complete = json.loads((resume.parent / 'COMPLETE.json').read_text())
        assert previous_complete['checkpoint_sha256'] == args.resume_sha256
        assert previous_complete['strict_reload_bit_equal']
        model.load_state_dict(previous['net'], strict=True)
        optimizer.load_state_dict(previous['optimizer'])
        previous_updates = previous_complete['updates']
        assert all(group['lr'] == plan['lr'] for group in optimizer.param_groups)
        assert all(int(optimizer.state[p]['step']) == previous_updates for p in params)
        del previous
    run = ROOT / 'runs' / args.run
    run.mkdir(parents=True, exist_ok=False)
    config = dict(plan=plan, plan_sha256=sha(plan_path), sanity=args.sanity, task=args.task,
        epoch=args.epoch, resume=args.resume, resume_sha256=args.resume_sha256)
    (run / 'config.json').write_text(json.dumps(config, indent=2))
    batch_size = plan['batch_size']
    samples = 4 * batch_size if args.sanity else plan['samples_per_epoch']
    assert samples % batch_size == 0
    dataset = LasHeRPairs(manifest, 'train', samples, plan['seed'] + (args.epoch - 1) * samples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    losses = []
    for step, batch in enumerate(loader):
        templates, search, target = [batch[k].cuda() for k in ('templates', 'search', 'target')]
        modality = (previous_updates + step) % 2
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
            assert torch.equal(old_score, updated['score']) and torch.equal(old_offset, updated['offset'])
        losses.append(loss.item())
        print(json.dumps(dict(epoch=args.epoch, step=step, loss=loss.item(), modality=modality)), flush=True)
    current = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    changed = [k for k in parent_net if not torch.equal(parent_net[k], current[k])]
    assert set(changed) == {PREFIX + 'weight', PREFIX + 'bias'}
    updates = previous_updates + samples // batch_size
    assert all(int(optimizer.state[p]['step']) == updates for p in params)
    path = save_epoch(run, dict(net=current, epoch=args.epoch, optimizer=optimizer.state_dict(),
        stage='size_tail_adaptation', config=config))
    with torch.no_grad():
        expected = model.executor.primary(templates, search, modality)
        model.load_state_dict(torch.load(path, map_location='cpu')['net'], strict=True)
        actual = model.executor.primary(templates, search, modality)
        assert all(torch.equal(expected[k], actual[k]) for k in ('box', 'score', 'size', 'offset'))
    result = dict(status='SANITY_PASS' if args.sanity else 'EPOCH_COMPLETE_PENDING_DEV', epoch=args.epoch,
        updates=updates, epoch_updates=samples // batch_size, previous_updates=previous_updates,
        changed_state_tensors=changed, trainable_parameters=66,
        total_parameters=sum(p.numel() for p in model.parameters()), mean_train_loss=float(np.mean(losses)),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        elapsed_seconds=time.time()-started, strict_reload_bit_equal=True, checkpoint_sha256=sha(path),
        optimizer_steps_verified=True, resume_sha256=args.resume_sha256,
        scope='Train split only. Dev metrics, not crop loss, determine stopping and selection.')
    check_files(ROOT, plan['source_sha256'])
    assert sha(plan_path) == config['plan_sha256']
    (run / 'COMPLETE.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

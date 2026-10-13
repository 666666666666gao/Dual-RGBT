"""Real-data execution check; does not evaluate the baseline or benchmark test set."""
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from dual_rgbt.data import LasHeRPairs
from dual_rgbt.model import DualRGBT
from dual_rgbt.losses import executor_loss


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES'] in ('0', '1')
    assert sys.prefix == str(root / 'conda-envs/dual_rgbt'), sys.prefix
    torch.manual_seed(2027)
    torch.cuda.manual_seed_all(2027)
    torch.backends.cudnn.benchmark = False
    start = time.perf_counter()
    model = DualRGBT().cuda()
    for module in (model.preview, model.quality, model.gain):
        module.requires_grad_(False)
    model.eval()  # Deterministic kernel check, not an accuracy/training claim.
    batch = next(iter(DataLoader(LasHeRPairs(root / 'data/manifest.json', 'train', 2), batch_size=2)))
    templates, search, target = [batch[k].cuda() for k in ('templates', 'search', 'target')]
    search.requires_grad_()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
    trace = []
    hooks = [block.attn.register_forward_pre_hook(lambda module, args: trace.append(args[0].shape[1]))
             for block in model.backbone.blocks]
    checks = []
    for modality in range(2):
        for action in range(5):
            trace.clear()
            before = model.backbone.blocks[11].attn.qkv.weight.detach().clone()
            result = model.forward_action(templates, search, modality, action)
            expected = [385] * 12
            if action:
                expected += ([385] if action == 4 else [193]) * 12
            assert trace == expected, (modality, action, trace)
            loss = executor_loss(result, target)
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True)
            search.grad = None
            loss.backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            assert all(torch.isfinite(g).all() for g in gradients)
            assert any(g.abs().sum() > 0 for g in gradients)
            assert model.backbone.blocks[11].attn.qkv.weight.grad.abs().sum() > 0
            if action:
                # The new spatial correction must actually use auxiliary pixels.
                assert search.grad[:, (1 - modality) * 3:(2 - modality) * 3].abs().sum() > 0
            optimizer.step()
            assert not torch.equal(before, model.backbone.blocks[11].attn.qkv.weight)
            checks.append({'modality': modality, 'action': action, 'loss': float(loss.item()),
                           'attention_token_trace': list(trace), 'optimizer_updated': True})
    for hook in hooks:
        hook.remove()
    with torch.no_grad():
        reference = model.forward_action(templates, search, 0, 1)['box']
    checkpoint = root / 'runs/m0_spatial_executor.pth'
    checkpoint.parent.mkdir(exist_ok=True)
    torch.save({'net': model.state_dict()}, checkpoint)
    clone = DualRGBT().cuda().eval()
    clone.load_state_dict(torch.load(checkpoint, map_location='cpu')['net'], strict=True)
    with torch.no_grad():
        reloaded = clone.forward_action(templates, search, 0, 1)['box']
    assert torch.equal(reference, reloaded), (reference - reloaded).abs().max().item()
    # STOP is tested in the actual policy too: setting a positive request cost
    # above all finite predicted gains must skip every heavy auxiliary call.
    trace.clear()
    hook = clone.backbone.blocks[0].attn.register_forward_pre_hook(
        lambda module, args: trace.append(args[0].shape[1]))
    clone.predict(templates[:1], search[:1], search.new_zeros(1, 8), cost_weight=1e6)
    hook.remove()
    assert trace == [385], trace
    report = {'status': 'PASS', 'seed': 2027, 'checks': checks, 'strict_reload_bit_equal': True,
              'policy_stop_heavy_trace': trace, 'python_prefix': sys.prefix,
              'torch': torch.__version__, 'cuda': torch.version.cuda,
              'device': torch.cuda.get_device_name(0), 'physical_gpu': os.environ['CUDA_VISIBLE_DEVICES'],
              'total_parameters': sum(p.numel() for p in model.parameters()),
              'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
              'peak_memory_bytes': torch.cuda.max_memory_allocated(),
              'elapsed_seconds': time.perf_counter() - start,
              'sequences': batch['sequence'], 'frames': batch['frame'].tolist(),
              'scope': 'Execution check only: no PR/NPR/SR or latency improvement claim.'}
    (root / 'reports/sanity.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

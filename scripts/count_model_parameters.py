"""CPU-only parameter inventory, without forward passes or baseline scoring."""
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from dual_rgbt.model import DualRGBT
from lib.config.cadtrack.config import cfg
from lib.models.cadtrack.cadtrack import build_cadtrack


def main():
    output = root / 'reports/model_parameter_counts.json'
    assert not output.exists()
    checkpoint = root / 'runs/stage_c_complete_20261013/latest.pth'
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == '94be6239f7f6864cc193fff7c21d67c9479924c9432039c6ff16cd0d1223f15f'
    baseline_path = Path('/home/gaob/CADTrack_ep0014.pth.tar')
    baseline_before_sha256 = hashlib.sha256(baseline_path.read_bytes()).hexdigest()
    ours = DualRGBT(checkpoint=str(baseline_path)).eval()
    ours.load_state_dict(torch.load(checkpoint, map_location='cpu')['net'], strict=True)
    baseline = build_cadtrack(cfg, training=False).eval()
    baseline.load_state_dict(torch.load(baseline_path, map_location='cpu')['net'], strict=True)
    assert all(p.device.type == 'cpu' for m in (ours, baseline) for p in m.parameters())
    assert not torch.cuda.is_initialized()
    count = lambda m: sum(p.numel() for p in m.parameters())
    components = {name: count(getattr(ours, name)) for name in
                  ('backbone', 'decode', 'box_head', 'preview', 'quality', 'gain')}
    original = {'backbone': count(baseline.backbone), 'decode': count(baseline.decode_fuse_search),
                'box_head': count(baseline.box_head)}
    assert all(components[k] == value for k, value in original.items())
    assert count(ours) == sum(components.values())
    assert count(baseline) == sum(original.values())
    for name in ('backbone', 'decode', 'box_head', 'preview'):
        getattr(ours, name).requires_grad_(False)
    trainable_c = sum(p.numel() for p in ours.parameters() if p.requires_grad)
    assert trainable_c == components['quality'] + components['gain']
    report = dict(baseline_total_parameters=count(baseline), ours_total_parameters=count(ours),
                  ours_components=components, baseline_components=original,
                  parameter_delta=count(ours) - count(baseline), stage_c_trainable_parameters=trainable_c,
                  baseline_before_sha256=baseline_before_sha256,
                  checkpoints_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in (baseline_path, checkpoint)},
                  source_sha256={p: hashlib.sha256((root / p).read_bytes()).hexdigest()
                                 for p in ('scripts/count_model_parameters.py', 'dual_rgbt/model.py',
                                           'lib/models/cadtrack/cadtrack.py')},
                  neural_forward=False, optimizer_updates=0, cuda_context=False,
                  scope='Static instantiated CPU model parameter inventory with strict checkpoint loads. '
                        'Buffers excluded; no historical baseline identity attestation, baseline accuracy, '
                        'FLOPs, latency or independent-module claim.')
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

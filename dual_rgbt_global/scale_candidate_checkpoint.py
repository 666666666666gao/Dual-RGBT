"""Restore the candidate geometry configuration together with its trained state."""
from pathlib import Path

import torch

from dual_rgbt_global.scale_candidate_policy import build_scale_candidate_policy


def load_scale_candidate_checkpoint(checkpoint_path, project_root):
    saved = torch.load(checkpoint_path, map_location='cpu')
    config = saved['config']
    assert saved['stage'] == config['stage'] == 'scale_candidate_b'
    candidate = config['candidate_config']
    model = build_scale_candidate_policy(Path(project_root) / config['factory_checkpoint'],
        candidate['local_radius_fraction'], candidate['wide_radius_fraction'])
    model.load_state_dict(saved['net'], strict=True)
    return model, config

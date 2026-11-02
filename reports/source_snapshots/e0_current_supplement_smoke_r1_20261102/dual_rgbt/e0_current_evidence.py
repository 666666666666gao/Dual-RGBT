"""Observe the current E0 proposal in the original identity feature space."""
import numpy as np
import torch

from dual_rgbt.e0_identity import primary_candidate_evidence


@torch.no_grad()
def current_proposal_evidence(model, fixed_templates, search, geometry, current_box):
    """Use the preceding-state crop and fixed initial templates; no GT or commits."""
    assert not model.training
    assert fixed_templates.shape == (1, 2, 6, 128, 128)
    assert search.shape == (1, 6, 256, 256)
    box = np.asarray(current_box, dtype=np.float32)
    geometry = np.asarray(geometry, dtype=np.float32)
    assert box.shape == (4,) and geometry.shape == (3,)
    assert np.isfinite(box).all() and np.isfinite(geometry).all()
    assert (box[2:] > 0).all() and geometry[2] > 0
    left, top, side = geometry
    centre = (box[:2] + box[2:] / 2 - [left, top]) / side
    normalized = np.concatenate([centre, box[2:] / side]).astype(np.float32)
    normalized_tensor = torch.from_numpy(normalized[None]).to(search)
    descriptors, coverage = [], []
    for modality in range(2):
        start = 3 * modality
        features, _ = model.encode(fixed_templates[:, :, start:start + 3],
                                   search[:, start:start + 3])
        descriptor, covered = primary_candidate_evidence(features, normalized_tensor)
        descriptors.append(descriptor[0])
        coverage.append(covered[0])
    return dict(features=torch.stack(descriptors).detach().clone(),
                coverage=torch.stack(coverage).detach().clone(),
                normalized_box=normalized.copy())

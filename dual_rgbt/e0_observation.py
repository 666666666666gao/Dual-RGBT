"""Read-only deep ROI observations of the original E0 prediction path."""
import torch
from torch.nn import functional as F


def roi_descriptor(features, box):
    """Pool a 3x3 ROI from the primary 16x16 grid; retain coverage explicitly."""
    assert features.shape == (1, 256, 768) and box.shape == (1, 4)
    points = (torch.arange(3, device=features.device, dtype=features.dtype) + .5) / 3 - .5
    x = box[:, :1] + points[None] * box[:, 2:3]
    y = box[:, 1:2] + points[None] * box[:, 3:4]
    grid = torch.stack([x[:, None].expand(-1, 3, -1),
                        y[:, :, None].expand(-1, -1, 3)], -1)
    covered = ((grid >= 0) & (grid < 1)).all(-1)
    spatial = features.transpose(1, 2).reshape(1, 768, 16, 16)
    sampled = F.grid_sample(spatial, grid * 2 - 1, mode='bilinear',
                            padding_mode='zeros', align_corners=False)
    sampled = sampled * covered[:, None].to(features.dtype)
    return sampled.mean((2, 3)), covered.to(features.dtype).mean((1, 2))[:, None]


class E0Observation:
    """Call the untouched E0 predictor once, return the same result object."""

    def __init__(self, model):
        self.model = model
        self.original = model.predict
        self.enabled = False
        self.calls = 0
        self.record = None
        model.predict = self.predict

    def begin(self, enabled):
        self.enabled = enabled
        self.calls = 0
        self.record = None

    @torch.no_grad()
    def predict(self, templates, search, history, cost_weight, window=None):
        output = self.original(templates, search, history, cost_weight, window)
        self.calls += 1
        if self.enabled:
            primary = output['primary']
            roi, coverage = roi_descriptor(primary['features'], output['box'])
            score = primary['score'].flatten(1)
            probability = score / score.sum(1, keepdim=True)
            statistics = torch.stack([score.max(1)[0], score.mean(1),
                -(probability * probability.log()).sum(1) / 5.545177], 1)
            identity = score.new_zeros(1, 2)
            identity[:, output['modality']] = 1
            preview = output['preview']
            features = torch.cat(preview['pools'] + preview['descriptors'] +
                [history, roi, coverage, statistics, output['box'], identity], 1)
            assert features.shape == (1, 914) and torch.isfinite(features).all()
            confidence = (output['score'] if output['action'] else primary['score']).max()
            self.record = dict(features=features.detach().cpu().numpy()[0].copy(),
                primary_box=primary['box'].detach().cpu().numpy()[0].copy(),
                final_normalized_box=output['box'].detach().cpu().numpy()[0].copy(),
                primary_confidence=float(primary['score'].max().item()),
                final_confidence=float(confidence.item()),
                primary_roi_coverage=float(coverage.item()))
        return output

    def close(self):
        self.model.predict = self.original

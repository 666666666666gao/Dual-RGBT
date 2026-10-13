"""Cheap preview -> expected executed IoU benefit -> one raw auxiliary region."""
import torch
from torch import nn
from torch.nn import functional as F

from dual_rgbt.model import DualRGBT
from dual_rgbt_evidence.readout import LocalEvidenceReader


ACTION_NAMES = ('STOP', 'LOCAL_MAIN', 'LOCAL_ALTERNATE', 'LOCAL_EXTERNAL', 'WIDE')


class EvidencePolicy(nn.Module):
    def __init__(self, executor):
        super().__init__()
        # Load the old executor strictly before constructing this new model.
        # Its old gain head controls different actions/labels and is not retained.
        del executor.gain
        self.executor = executor
        self.query_gain = nn.Sequential(nn.Linear(955, 64), nn.GELU(), nn.Linear(64, 1))

    def preview_state(self, templates, main_preview, broad_preview, history):
        state = self.executor.preview_state(templates, main_preview, history)
        features, maps = [], []
        for modality in range(2):
            feature = self.executor.preview(broad_preview[:, modality * 3:(modality + 1) * 3])
            features.append(feature)
            maps.append(F.cosine_similarity(feature, state['descriptors'][modality][:, :, None, None], dim=1))
        state.update(broad_features=features, broad_maps=maps)
        return state

    def query_geometries(self, primary, preview, modality, main_geometry, broad_geometry):
        """Four concrete requests, derived before any heavy auxiliary encoding."""
        score = primary['score'].flatten(1)
        first = score.argmax(1)
        cells = torch.arange(256, device=score.device)[None]
        nearby = ((cells % 16 - first[:, None] % 16).abs() <= 2) & (
                  (cells // 16 - first[:, None] // 16).abs() <= 2)
        second = score.masked_fill(nearby, -1).argmax(1)
        gather = second[:, None, None].expand(-1, 2, -1)
        offsets = primary['offset'].flatten(2).gather(2, gather).squeeze(2)
        second_centre = (torch.stack([second % 16, second // 16], dim=1).to(offsets.dtype) + offsets) / 16
        external = preview['broad_maps'][1 - modality].flatten(1).argmax(1)
        external_centre = (torch.stack([external % 8, external // 8], dim=1).to(offsets.dtype) + .5) / 8
        centres = [primary['box'][:, :2] * main_geometry[:, 2:3] + main_geometry[:, :2],
                   second_centre * main_geometry[:, 2:3] + main_geometry[:, :2],
                   external_centre * broad_geometry[:, 2:3] + broad_geometry[:, :2]]
        side = main_geometry[:, 2:3] / 2
        local = [torch.cat([centre - side / 2, side], dim=1) for centre in centres]
        return torch.stack(local + [broad_geometry], dim=1)

    def gain_predictions(self, primary, preview, modality, main_geometry, broad_geometry, queries):
        score = primary['score'].flatten(1)
        probabilities = score / score.sum(1, keepdim=True)
        statistics = torch.stack([score.max(1)[0], score.mean(1),
                                  -(probabilities * probabilities.log()).sum(1) / 5.545177], dim=1)
        identity = score.new_zeros(score.shape[0], 2)
        identity[:, modality] = 1
        context = torch.cat(preview['pools'] + preview['descriptors'] +
                            [preview['history'], primary['features'].mean(1), statistics,
                             primary['box'], identity], dim=1)
        # A 3x3 cheap preview read describes each requested source region.
        points = (torch.arange(3, device=score.device, dtype=score.dtype) + .5) / 3
        ratio = queries[:, :, 2] / broad_geometry[:, None, 2]
        shift = (queries[:, :, :2] - broad_geometry[:, None, :2]) / broad_geometry[:, None, 2:3]
        x = 2 * (points[None, None] * ratio[:, :, None] + shift[:, :, :1]) - 1
        y = 2 * (points[None, None] * ratio[:, :, None] + shift[:, :, 1:2]) - 1
        grid = torch.stack([x[:, :, None].expand(-1, -1, 3, -1),
                            y[:, :, :, None].expand(-1, -1, -1, 3)], dim=-1)
        grid = grid.reshape(score.shape[0], 12, 3, 2)
        values = F.grid_sample(preview['broad_features'][1 - modality], grid,
                               mode='bilinear', padding_mode='zeros', align_corners=False)
        covered = ((grid[..., 0] >= -1) & (grid[..., 0] < 1) &
                   (grid[..., 1] >= -1) & (grid[..., 1] < 1))
        values = values * covered[:, None].to(values.dtype)
        regional = values.reshape(score.shape[0], 32, 4, 3, 3).mean((3, 4)).transpose(1, 2)
        relevance = F.cosine_similarity(regional, preview['descriptors'][1 - modality][:, None], dim=2)
        intersection_left = torch.maximum(queries[:, :, :2], main_geometry[:, None, :2])
        intersection_right = torch.minimum(queries[:, :, :2] + queries[:, :, 2:3],
                                           main_geometry[:, None, :2] + main_geometry[:, None, 2:3])
        coverage = (intersection_right - intersection_left).clamp(min=0).prod(2) / queries[:, :, 2].square()
        relative = torch.cat([(queries[:, :, :2] - main_geometry[:, None, :2]) / main_geometry[:, None, 2:3],
                              queries[:, :, 2:3] / main_geometry[:, None, 2:3]], dim=2)
        kind = F.one_hot(torch.arange(4, device=score.device), 4).to(score.dtype)[None].expand(score.shape[0], -1, -1)
        costs = self.executor.token_cost[1:][None, :, None].expand(score.shape[0], -1, -1)
        inputs = torch.cat([context[:, None].expand(-1, 4, -1), relative, costs, kind,
                            regional, relevance[:, :, None], coverage[:, :, None]], dim=2)
        return torch.cat([score.new_zeros(score.shape[0], 1), self.query_gain(inputs).squeeze(2)], dim=1)

    def query(self, templates, image, primary, main_geometry, query_geometry, modality):
        return LocalEvidenceReader(self.executor)(templates, image, primary['features'],
                                                  main_geometry, query_geometry, modality)


def build_policy(executor_checkpoint):
    executor = DualRGBT('/home/gaob/CADTrack_ep0014.pth.tar')
    executor.load_state_dict(torch.load(executor_checkpoint, map_location='cpu')['net'], strict=True)
    return EvidencePolicy(executor)

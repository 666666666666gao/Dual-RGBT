"""Score a cheap region pool, then request at most one auxiliary encoding."""
import torch
from torch.nn import functional as F

from dual_rgbt.model import DualRGBT
from dual_rgbt_global.policy import GlobalEvidencePolicy


class CandidateEvidencePolicy(GlobalEvidencePolicy):
    def candidate_queries(self, primary, preview, modality, main_geometry, broad_geometry,
                          image_sizes, max_candidates):
        assert primary['box'].shape[0] == 1
        main = self.query_geometries(primary, preview, modality, main_geometry,
                                     broad_geometry, image_sizes)[:, :2]
        cells = torch.arange(1024, device=main.device)
        normalized = (torch.stack([cells % 32, cells // 32], 1).to(main.dtype) + .5) / 32
        coordinates = normalized * broad_geometry[0, 2] + broad_geometry[0, :2]
        valid = ((coordinates >= 0).all(1) & (coordinates[:, 0] < image_sizes[0, 1]) &
                 (coordinates[:, 1] < image_sizes[0, 0]))
        remaining = preview['broad_maps'][1 - modality][0].flatten().masked_fill(~valid, -torch.inf)
        indices = []
        radius = float(main_geometry[0, 2].item()) / 4
        # Same greedy separation as the measured candidate diagnostic; lists may exhaust.
        for _ in range(max_candidates):
            peak = int(remaining.argmax().item())
            if not torch.isfinite(remaining[peak]).item():
                break
            indices.append(peak)
            distance = (coordinates - coordinates[peak]).square().sum(1)
            remaining = remaining.masked_fill(distance < radius ** 2, -torch.inf)
        regions = [main[:, 0], main[:, 1]]
        kinds, ranks = [0, 1], [-1, -1]
        for rank, index in enumerate(indices):
            for kind, side in ((2, main_geometry[:, 2:3] / 2), (3, main_geometry[:, 2:3])):
                regions.append(torch.cat([coordinates[index][None] - side / 2, side], 1))
                kinds.append(kind)
                ranks.append(rank)
        return dict(geometries=torch.stack(regions, 1),
                    kinds=torch.tensor([kinds], device=main.device, dtype=torch.long),
                    ranks=torch.tensor([ranks], device=main.device, dtype=torch.long),
                    candidate_indices=torch.tensor(indices, device=main.device, dtype=torch.long))

    def candidate_costs(self, kinds):
        costs = self.executor.token_cost[kinds + 1]
        return torch.cat([costs.new_zeros(costs.shape[0], 1), costs], 1)

    def candidate_gains(self, primary, preview, modality, main_geometry, broad_geometry, queries, kinds):
        batch, count = queries.shape[:2]
        assert kinds.shape == (batch, count)
        score = primary['score'].flatten(1)
        probabilities = score / score.sum(1, keepdim=True)
        statistics = torch.stack([score.max(1)[0], score.mean(1),
                                  -(probabilities * probabilities.log()).sum(1) / 5.545177], 1)
        identity = score.new_zeros(batch, 2)
        identity[:, modality] = 1
        context = torch.cat(preview['pools'] + preview['descriptors'] +
                            [preview['history'], primary['features'].mean(1), statistics,
                             primary['box'], identity], 1)
        points = (torch.arange(3, device=score.device, dtype=score.dtype) + .5) / 3
        ratio = queries[:, :, 2] / broad_geometry[:, None, 2]
        shift = (queries[:, :, :2] - broad_geometry[:, None, :2]) / broad_geometry[:, None, 2:3]
        x = 2 * (points[None, None] * ratio[:, :, None] + shift[:, :, :1]) - 1
        y = 2 * (points[None, None] * ratio[:, :, None] + shift[:, :, 1:2]) - 1
        grid = torch.stack([x[:, :, None].expand(-1, -1, 3, -1),
                            y[:, :, :, None].expand(-1, -1, -1, 3)], -1).reshape(batch, 3 * count, 3, 2)
        values = F.grid_sample(preview['broad_features'][1 - modality], grid,
                               mode='bilinear', padding_mode='zeros', align_corners=False)
        covered = ((grid[..., 0] >= -1) & (grid[..., 0] < 1) &
                   (grid[..., 1] >= -1) & (grid[..., 1] < 1))
        values = values * covered[:, None].to(values.dtype)
        regional = values.reshape(batch, 32, count, 3, 3).mean((3, 4)).transpose(1, 2)
        relevance = F.cosine_similarity(regional, preview['descriptors'][1 - modality][:, None], dim=2)
        left = torch.maximum(queries[:, :, :2], main_geometry[:, None, :2])
        right = torch.minimum(queries[:, :, :2] + queries[:, :, 2:3],
                              main_geometry[:, None, :2] + main_geometry[:, None, 2:3])
        coverage = (right - left).clamp(min=0).prod(2) / queries[:, :, 2].square()
        relative = torch.cat([(queries[:, :, :2] - main_geometry[:, None, :2]) / main_geometry[:, None, 2:3],
                              queries[:, :, 2:3] / main_geometry[:, None, 2:3]], 2)
        costs = self.executor.token_cost[kinds + 1][:, :, None]
        kind_features = F.one_hot(kinds, 4).to(score.dtype)
        inputs = torch.cat([context[:, None].expand(-1, count, -1), relative, costs, kind_features,
                            regional, relevance[:, :, None], coverage[:, :, None]], 2)
        assert inputs.shape[2] == 955
        return torch.cat([score.new_zeros(batch, 1), self.query_gain(inputs).squeeze(2)], 1)


def build_candidate_policy(executor_checkpoint):
    executor = DualRGBT('/home/gaob/CADTrack_ep0014.pth.tar')
    executor.load_state_dict(torch.load(executor_checkpoint, map_location='cpu')['net'], strict=True)
    return CandidateEvidencePolicy(executor)

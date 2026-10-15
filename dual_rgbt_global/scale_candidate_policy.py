"""Use separate spatial separation budgets for local and wide requests."""
import torch

from dual_rgbt.model import DualRGBT
from dual_rgbt_global.candidate_policy import CandidateEvidencePolicy


def separated_indices(scores, coordinates, radius, capacity):
    remaining = scores.clone()
    indices = []
    for _ in range(capacity):
        peak = int(remaining.argmax().item())
        if not torch.isfinite(remaining[peak]).item():
            break
        indices.append(peak)
        distance = (coordinates - coordinates[peak]).square().sum(1)
        remaining = remaining.masked_fill(distance < radius ** 2, -torch.inf)
    return indices


class ScaleCandidateEvidencePolicy(CandidateEvidencePolicy):
    def __init__(self, executor, local_radius_fraction, wide_radius_fraction):
        super().__init__(executor)
        assert local_radius_fraction > 0 and wide_radius_fraction > 0
        self.local_radius_fraction = float(local_radius_fraction)
        self.wide_radius_fraction = float(wide_radius_fraction)

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
        scores = preview['broad_maps'][1 - modality][0].flatten().masked_fill(~valid, -torch.inf)
        side = float(main_geometry[0, 2].item())
        local = separated_indices(scores, coordinates, side * self.local_radius_fraction, max_candidates)
        wide = separated_indices(scores, coordinates, side * self.wide_radius_fraction, max_candidates)
        regions = [main[:, 0], main[:, 1]]
        kinds, ranks, request_cells = [0, 1], [-1, -1], [-1, -1]
        for rank in range(max(len(local), len(wide))):
            for kind, indices, ratio in ((2, local, .5), (3, wide, 1.)):
                if rank < len(indices):
                    index = indices[rank]
                    query_side = main_geometry[:, 2:3] * ratio
                    regions.append(torch.cat([coordinates[index][None] - query_side / 2, query_side], 1))
                    kinds.append(kind)
                    ranks.append(rank)
                    request_cells.append(index)
        # A cell may serve both families; this is the unique spatial-cell count.
        union = list(dict.fromkeys(local + wide))
        return dict(geometries=torch.stack(regions, 1),
            kinds=torch.tensor([kinds], device=main.device, dtype=torch.long),
            ranks=torch.tensor([ranks], device=main.device, dtype=torch.long),
            candidate_indices=torch.tensor(union, device=main.device, dtype=torch.long),
            request_cell_indices=torch.tensor([request_cells], device=main.device, dtype=torch.long),
            family_candidate_counts=torch.tensor([[len(local), len(wide)]], device=main.device, dtype=torch.long))


def build_scale_candidate_policy(executor_checkpoint, local_radius_fraction, wide_radius_fraction):
    executor = DualRGBT('/home/gaob/CADTrack_ep0014.pth.tar')
    executor.load_state_dict(torch.load(executor_checkpoint, map_location='cpu')['net'], strict=True)
    return ScaleCandidateEvidencePolicy(executor, local_radius_fraction, wide_radius_fraction)

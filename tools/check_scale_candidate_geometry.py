"""Exercise actual policy tensor geometry on archived scores, without a neural model."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dual_rgbt_global.candidate_policy import CandidateEvidencePolicy
from dual_rgbt_global.policy import GlobalEvidencePolicy
from dual_rgbt_global.scale_candidate_policy import ScaleCandidateEvidencePolicy


class GeometryOnly:
    query_geometries = GlobalEvidencePolicy.query_geometries
    local_radius_fraction = .25
    wide_radius_fraction = .5


def reference(scores, coordinates, radius, capacity):
    selected = []
    for index in np.argsort(-scores, kind='stable'):
        if not np.isfinite(scores[index]):
            break
        if all(np.sum((coordinates[index] - coordinates[old]) ** 2, dtype=np.float32)
               >= np.float32(radius ** 2) for old in selected):
            selected.append(int(index))
            if len(selected) == capacity:
                break
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    assert not output.exists()
    torch.set_num_threads(1)
    inputs = np.load(args.inputs, allow_pickle=False)
    count = len(inputs['scores'])
    assert count == 1752 and inputs['scores'].shape == (count, 1024)
    geometry = GeometryOnly()
    primary = dict(box=torch.tensor([[.5, .5, .2, .2]]),
                   score=torch.arange(1, 257, dtype=torch.float32).reshape(1, 1, 16, 16),
                   offset=torch.full((1, 2, 16, 16), .5))
    same_fields = ('geometries', 'kinds', 'ranks', 'candidate_indices')
    changed_pools = 0
    maximum_unique_cells = 0
    for state in range(count):
        scores = inputs['scores'][state]
        preview = dict(broad_maps=[torch.from_numpy(scores.copy()).reshape(1, 32, 32)] * 2)
        main = torch.from_numpy(inputs['main_geometry'][state:state + 1])
        broad = torch.from_numpy(inputs['broad_geometry'][state:state + 1])
        sizes = torch.from_numpy(inputs['image_sizes'][state:state + 1])
        values = (primary, preview, 0, main, broad, sizes, 8)
        legacy = CandidateEvidencePolicy.candidate_queries(geometry, *values)
        expected_indices = inputs['default_indices'][state]
        expected_indices = expected_indices[expected_indices >= 0].tolist()
        assert legacy['candidate_indices'].tolist() == expected_indices
        geometry.wide_radius_fraction = .25
        control = ScaleCandidateEvidencePolicy.candidate_queries(geometry, *values)
        assert all(torch.equal(legacy[name], control[name]) for name in same_fields)
        geometry.wide_radius_fraction = .5
        mixed = ScaleCandidateEvidencePolicy.candidate_queries(geometry, *values)
        assert torch.equal(mixed['geometries'][:, :2], legacy['geometries'][:, :2])
        cells = (np.stack([np.arange(1024) % 32, np.arange(1024) // 32], 1).astype(np.float32) + .5) / 32
        coordinates = cells * inputs['broad_geometry'][state, 2] + inputs['broad_geometry'][state, :2]
        valid = ((coordinates >= 0).all(1) & (coordinates[:, 0] < sizes[0, 1].item()) &
                 (coordinates[:, 1] < sizes[0, 0].item()))
        masked = np.where(valid, scores, -np.inf)
        selected_families = []
        for kind, fraction in ((2, .25), (3, .5)):
            selected = reference(masked, coordinates, float(main[0, 2].item()) * fraction, 8)
            selected_families.append(selected)
            keep = mixed['kinds'][0] == kind
            assert mixed['request_cell_indices'][0, keep].tolist() == selected
            assert mixed['ranks'][0, keep].tolist() == list(range(len(selected)))
            side = main[0, 2] * (.5 if kind == 2 else 1.)
            expected = torch.cat([torch.from_numpy(coordinates[selected]) - side / 2,
                                  side.expand(len(selected), 1)], 1)
            assert torch.equal(mixed['geometries'][0, keep], expected)
        local, wide = selected_families
        assert mixed['family_candidate_counts'].tolist() == [[len(local), len(wide)]]
        union = list(dict.fromkeys(local + wide))
        assert mixed['candidate_indices'].tolist() == union
        assert len(mixed['kinds'][0]) == 2 + len(local) + len(wide) <= 18
        assert mixed['request_cell_indices'][0, :2].tolist() == [-1, -1]
        changed_pools += int(local != wide)
        maximum_unique_cells = max(maximum_unique_cells, len(union))
        one_values = (primary, preview, 0, main, broad, sizes, 1)
        one = ScaleCandidateEvidencePolicy.candidate_queries(geometry, *one_values)
        old_one = CandidateEvidencePolicy.candidate_queries(geometry, *one_values)
        assert all(torch.equal(one[name], old_one[name]) for name in same_fields)
    assert not torch.cuda.is_initialized()
    result = dict(status='CPU_GEOMETRY_PASS', states=count, changed_family_pools=changed_pools,
        maximum_unique_cells=maximum_unique_cells, maximum_queries=18,
        original_radius_pool_bit_equal=True, one_candidate_legacy_bit_equal=True,
        independent_numpy_family_geometry_bit_equal=True, torch_version=torch.__version__,
        cuda_initialized=False, neural_model_constructed=False, neural_forward=False,
        scope='Actual Torch CPU proposal geometry on archived auxiliary score maps and synthetic primary geometry tensors; no weights, learning, tracking metrics, or latency evidence.')
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

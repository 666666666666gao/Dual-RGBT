"""Read a requested auxiliary crop with shared weights and a local spatial head.

The caller selects and crops the raw auxiliary region before calling this
module. Query geometry is independent of the main search geometry, so a query
may read and locate outside the main search crop. This is an executor component,
not a trained scheduling policy or a complete online tracker.
"""
import torch
from torch import nn
from torch.nn import functional as F


def sample_main_features(features, main_geometry, query_geometry, grid_size):
    """Project already computed main tokens to query-token centres.

    Geometry is [left, top, source_side] in original-image pixels, [B,3].
    Missing main coverage has zero features; auxiliary pixels remain available
    from the independently requested query crop.
    """
    assert features.ndim == 3 and features.shape[1:] == (256, 768)
    assert main_geometry.shape == query_geometry.shape == (features.shape[0], 3)
    assert grid_size in (8, 16)
    positions = (torch.arange(grid_size, device=features.device, dtype=features.dtype) + .5) / grid_size
    ratio = query_geometry[:, 2] / main_geometry[:, 2]
    shift = (query_geometry[:, :2] - main_geometry[:, :2]) / main_geometry[:, 2:3]
    x = 2 * (positions[None] * ratio[:, None] + shift[:, :1]) - 1
    y = 2 * (positions[None] * ratio[:, None] + shift[:, 1:2]) - 1
    grid = torch.stack([x[:, None, :].expand(-1, grid_size, -1),
                        y[:, :, None].expand(-1, -1, grid_size)], dim=-1)
    dense = features.transpose(1, 2).reshape(features.shape[0], 768, 16, 16)
    sampled = F.grid_sample(dense, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    covered = ((grid[..., 0] >= -1) & (grid[..., 0] < 1) &
               (grid[..., 1] >= -1) & (grid[..., 1] < 1))
    return sampled * covered[:, None].to(features.dtype)


def box_from_maps(score, size, offset):
    """CENTER head decoding for the actual 8x8 or 16x16 query grid."""
    side = score.shape[-1]
    assert side in (8, 16) and score.shape[-2] == side
    _, index = score.flatten(1).max(dim=1, keepdim=True)
    gather = index[:, None].expand(-1, 2, -1)
    sizes = size.flatten(2).gather(2, gather).squeeze(2)
    offsets = offset.flatten(2).gather(2, gather).squeeze(2)
    centres = torch.cat([index % side, index // side], dim=1).to(offsets.dtype)
    return torch.cat([(centres + offsets) / side, sizes], dim=1)


def pixel_xywh(box, geometry):
    """Map a query-relative cxcywh proposal to original-image xywh.

    This is the raw proposal. The online caller applies the original clip_box
    margin10 once, then labels and stores that executed box in the same frame.
    No target coordinate is truncated to the main or query crop.
    """
    centre = box[:, :2] * geometry[:, 2:3] + geometry[:, :2]
    size = box[:, 2:] * geometry[:, 2:3]
    return torch.cat([centre - size / 2, size], dim=1)


class LocalEvidenceReader(nn.Module):
    """Use the supplied executor's backbone, decode and box-head parameters.

    128px local queries execute 129 template/CLS + 64 search tokens and an
    8x8 readout. A 256px wider query executes 385 tokens and a 16x16 readout.
    Source-area expansion does not require enlarging the heavy input budget.
    All weights are shared with the supplied model; this adds no parameters.
    Both paths require their own real-data training before use in a tracker.
    """

    def __init__(self, shared_model):
        super().__init__()
        self.model = shared_model

    def encode_query(self, templates, image):
        side = image.shape[-1]
        assert image.shape[-2] == side and side in (128, 256)
        assert templates.shape[1:] == (2, 3, 128, 128)
        if side == 256:
            indices = torch.arange(256, device=image.device)[None].expand(image.shape[0], -1)
            return self.model.encode(templates, image, indices)
        trunk = self.model.backbone
        template_tokens = [trunk.patch_embed(templates[:, i]) + trunk.pos_embed_z
                           for i in range(templates.shape[1])]
        # A local query has its own canonical coordinates. Preserve distinct
        # pretrained positions for its 8x8 tokens; do not scatter into main ROI.
        positions = trunk.pos_embed_x.reshape(1, 16, 16, trunk.embed_dim)[:, 4:12, 4:12]
        pixels = F.unfold(image, kernel_size=16, stride=16).transpose(1, 2)
        projection = trunk.patch_embed.proj
        search_tokens = F.linear(pixels, projection.weight.flatten(1), projection.bias)
        search_tokens = search_tokens + positions.reshape(1, 64, trunk.embed_dim)
        tokens = torch.cat([trunk.cls_token.expand(image.shape[0], -1, -1)] +
                           template_tokens + [search_tokens], dim=1)
        tokens = trunk.pos_drop(tokens)
        for block in trunk.blocks:
            tokens = tokens + block.drop_path(block.attn(block.norm1(tokens)))
            tokens = tokens + block.drop_path(block.mlp(block.norm2(tokens)))
        return trunk.norm(tokens)[:, -64:], tokens.shape[1]

    def forward(self, templates, query_image, primary_features, main_geometry,
                query_geometry, primary_modality, window=None):
        assert primary_modality in (0, 1)
        auxiliary = 1 - primary_modality
        evidence, tokens = self.encode_query(
            templates[:, :, auxiliary * 3:(auxiliary + 1) * 3],
            query_image[:, auxiliary * 3:(auxiliary + 1) * 3])
        side = query_image.shape[-1] // 16
        main = sample_main_features(primary_features, main_geometry, query_geometry, side)
        local = evidence.transpose(1, 2).reshape(query_image.shape[0], 768, side, side)
        pair = [main, local] if primary_modality == 0 else [local, main]
        score, sizes, offsets = self.model.box_head.get_score_map(self.model.decode(torch.cat(pair, dim=1)))
        if window is not None:
            assert window.shape[-2:] == (side, side)
            score = score * window
        box = box_from_maps(score, sizes, offsets)
        return {'box': box, 'pixel_box': pixel_xywh(box, query_geometry),
                'score': score, 'size': sizes, 'offset': offsets,
                'auxiliary_tokens': tokens, 'query_grid_size': side,
                'query_geometry': query_geometry}

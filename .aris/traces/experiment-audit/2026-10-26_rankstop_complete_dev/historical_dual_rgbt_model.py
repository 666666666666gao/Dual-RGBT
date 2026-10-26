from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from lib.config.cadtrack.config import cfg, update_config_from_file
from lib.models.cadtrack.cadtrack import build_cadtrack


ACTION_NAMES = ('STOP', 'PRIMARY', 'SECONDARY', 'PREVIEW', 'EXPAND')


class Preview(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.GELU())

    def forward(self, image):
        return self.net(F.interpolate(image, (64, 64), mode='bilinear', align_corners=False))


class DualRGBT(nn.Module):
    """One shared pretrained trunk, with selection before auxiliary patch encoding.

    Input templates: [B,2,6,128,128]; search: [B,6,256,256].
    Actions are explicit so balanced executor training does not depend on an
    untrained policy. Online inference uses predict(), batch size one.
    """

    def __init__(self, checkpoint='/home/gaob/CADTrack_ep0014.pth.tar'):
        super().__init__()
        root = Path(__file__).resolve().parents[1]
        update_config_from_file(str(root / 'experiments/cadtrack/cadtrack.yaml'))
        baseline = build_cadtrack(cfg, training=False)
        state = torch.load(checkpoint, map_location='cpu')['net']
        baseline.load_state_dict(state, strict=True)
        self.backbone = baseline.backbone
        self.box_head = baseline.box_head
        self.decode = baseline.decode_fuse_search
        self.preview = Preview()
        self.quality = nn.Sequential(nn.Linear(74, 64), nn.GELU(), nn.Linear(64, 1))
        self.gain = nn.Sequential(nn.Linear(1690, 128), nn.GELU(), nn.Linear(128, 1))
        self.register_buffer('token_cost', torch.tensor([0., 193. / 385.,
                                                       193. / 385., 193. / 385., 1.]))

    def preview_state(self, templates, search, history):
        maps, descriptors, pools, qualities = [], [], [], []
        for modality in range(2):
            image = search[:, modality * 3:(modality + 1) * 3]
            template = templates[:, 0, modality * 3:(modality + 1) * 3]
            feature = self.preview(image)
            # Target occupies the centre of the factor-two template crop.
            desc = self.preview(template)[:, :, 2:6, 2:6].mean((2, 3))
            pool = feature.mean((2, 3))
            identity = image.new_zeros(image.shape[0], 2)
            identity[:, modality] = 1
            qualities.append(self.quality(torch.cat([pool, desc, history, identity], 1)))
            maps.append(F.cosine_similarity(feature, desc[:, :, None, None], dim=1))
            pools.append(pool)
            descriptors.append(desc)
        return {'maps': maps, 'pools': pools, 'descriptors': descriptors,
                'quality': torch.cat(qualities, 1), 'history': history}

    def encode(self, templates, image, indices=None):
        trunk = self.backbone
        template_tokens = [trunk.patch_embed(templates[:, i]) + trunk.pos_embed_z
                           for i in range(templates.shape[1])]
        if indices is None:
            search_tokens = trunk.patch_embed(image) + trunk.pos_embed_x
        else:
            # Unfold reads raw pixels only. No auxiliary full feature map is made.
            pixels = F.unfold(image, kernel_size=16, stride=16).transpose(1, 2)
            pixels = pixels.gather(1, indices[:, :, None].expand(-1, -1, pixels.shape[2]))
            projection = trunk.patch_embed.proj
            search_tokens = F.linear(pixels, projection.weight.flatten(1), projection.bias)
            search_tokens = search_tokens + trunk.pos_embed_x.expand(image.shape[0], -1, -1).gather(
                1, indices[:, :, None].expand(-1, -1, trunk.embed_dim))
        tokens = torch.cat([trunk.cls_token.expand(image.shape[0], -1, -1)] +
                           template_tokens + [search_tokens], 1)
        tokens = trunk.pos_drop(tokens)
        # The original Block.forward(x,xi) runs both streams. Apply the shared
        # submodules once to the selected stream, retaining the pretrained math.
        for block in trunk.blocks:
            tokens = tokens + block.drop_path(block.attn(block.norm1(tokens)))
            tokens = tokens + block.drop_path(block.mlp(block.norm2(tokens)))
        return trunk.norm(tokens)[:, -search_tokens.shape[1]:], tokens.shape[1]

    def primary(self, templates, search, modality, window=None):
        features, tokens = self.encode(templates[:, :, modality * 3:(modality + 1) * 3],
                                       search[:, modality * 3:(modality + 1) * 3])
        zeros = torch.zeros_like(features)
        pair = [features, zeros] if modality == 0 else [zeros, features]
        dense = torch.cat(pair, 2).transpose(1, 2).reshape(search.shape[0], 1536, 16, 16)
        score, box, size, offset = self.box_head(self.decode(dense))
        if window is not None:
            score = score * window
            box = self.box_head.cal_bbox(score, size, offset)
        return {'features': features, 'score': score, 'box': box,
                'size': size, 'offset': offset, 'primary_tokens': tokens}

    def candidates(self, primary, preview, modality):
        score = primary['score'].flatten(1)
        first = score.argmax(1)
        cells = torch.arange(256, device=score.device)[None]
        nearby = ((cells % 16 - first[:, None] % 16).abs() <= 2) & (
                  (cells // 16 - first[:, None] // 16).abs() <= 2)
        second = score.masked_fill(nearby, -1).argmax(1)
        coarse = preview['maps'][1 - modality].flatten(1).argmax(1)
        third = (coarse // 8 * 2 + 1) * 16 + coarse % 8 * 2 + 1
        indices = torch.stack([first, second, third, first], 1)
        gathered = indices[:, None].expand(-1, 2, -1)
        sizes = primary['size'].flatten(2).gather(2, gathered).transpose(1, 2)
        offsets = primary['offset'].flatten(2).gather(2, gathered).transpose(1, 2)
        centres = torch.stack([indices % 16, indices // 16], 2).to(offsets.dtype)
        boxes = torch.cat([(centres + offsets) / 16, sizes], 2)
        return indices, boxes

    def region_indices(self, candidate, action):
        batch = candidate.shape[0]
        if action == 4:
            return torch.arange(256, device=candidate.device)[None].expand(batch, -1)
        left = (candidate % 16 - 4).clamp(0, 8)
        top = (candidate // 16 - 4).clamp(0, 8)
        local = torch.arange(8, device=candidate.device)
        return ((top[:, None, None] + local[None, :, None]) * 16 +
                left[:, None, None] + local[None, None, :]).reshape(batch, 64)

    def correct(self, templates, search, primary, modality, action, candidate, box, window=None):
        indices = self.region_indices(candidate, action)
        auxiliary = 1 - modality
        evidence, tokens = self.encode(templates[:, :, auxiliary * 3:(auxiliary + 1) * 3],
                                       search[:, auxiliary * 3:(auxiliary + 1) * 3], indices)
        # Query coordinates choose what to encode; they do not force a new box.
        # Keep the auxiliary tokens' spatial identity for the pretrained head.
        sparse = torch.zeros_like(primary['features']).scatter(
            1, indices[:, :, None].expand(-1, -1, 768), evidence)
        pair = [primary['features'], sparse] if modality == 0 else [sparse, primary['features']]
        dense = torch.cat(pair, 2).transpose(1, 2).reshape(search.shape[0], 1536, 16, 16)
        score, refined, size, offset = self.box_head(self.decode(dense))
        if window is not None:
            score = score * window
            refined = self.box_head.cal_bbox(score, size, offset)
        return {'box': refined, 'score': score, 'size': size, 'offset': offset,
                'auxiliary_tokens': tokens, 'region_indices': indices}

    def gain_predictions(self, primary, preview, modality, indices, boxes):
        score = primary['score'].flatten(1)
        probs = score / score.sum(1, keepdim=True)
        stats = torch.stack([score.max(1)[0], score.mean(1),
                             -(probs * probs.log()).sum(1) / 5.545177], 1)
        identity = score.new_zeros(score.shape[0], 2)
        identity[:, modality] = 1
        context = torch.cat(preview['pools'] + preview['descriptors'] +
                            [preview['history'], primary['features'].mean(1), stats,
                             primary['box'], identity], 1)
        features = primary['features'].gather(1, indices[:, :, None].expand(-1, -1, 768))
        inputs = torch.cat([context[:, None].expand(-1, 4, -1), features, boxes,
                            self.token_cost[1:][None, :, None].expand(score.shape[0], -1, -1),
                            F.one_hot(torch.arange(4, device=score.device), 4).to(score.dtype)[None].expand(
                                score.shape[0], -1, -1)], 2)
        return torch.cat([score.new_zeros(score.shape[0], 1), self.gain(inputs).squeeze(2)], 1)

    def forward_action(self, templates, search, modality, action, history=None):
        if history is None:
            history = search.new_zeros(search.shape[0], 8)
        preview = self.preview_state(templates, search, history)
        main = self.primary(templates, search, modality)
        indices, boxes = self.candidates(main, preview, modality)
        result = {'box': main['box'], 'primary': main, 'preview': preview,
                  'action': action, 'modality': modality, 'auxiliary_tokens': 0}
        if action:
            result.update(self.correct(templates, search, main, modality, action,
                                       indices[:, action - 1], boxes[:, action - 1]))
        result['gains'] = self.gain_predictions(main, preview, modality, indices, boxes)
        return result

    @torch.no_grad()
    def predict(self, templates, search, history, cost_weight, window=None):
        assert search.shape[0] == 1, 'Online policy operates on one tracker frame.'
        preview = self.preview_state(templates, search, history)
        modality = int(preview['quality'].argmax(1).item())
        main = self.primary(templates, search, modality, window)
        indices, boxes = self.candidates(main, preview, modality)
        gains = self.gain_predictions(main, preview, modality, indices, boxes)
        action = int((gains - cost_weight * self.token_cost).argmax(1).item())
        result = {'box': main['box'], 'primary': main, 'preview': preview, 'gains': gains,
                  'modality': modality, 'action': action, 'auxiliary_tokens': 0}
        if action:
            result.update(self.correct(templates, search, main, modality, action,
                                       indices[:, action - 1], boxes[:, action - 1], window))
        return result

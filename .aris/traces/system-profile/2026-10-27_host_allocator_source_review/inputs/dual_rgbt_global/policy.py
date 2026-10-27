"""One shared tiny CNN proposes globally before any auxiliary ViT encoding."""
import numpy as np
import torch
from torch.nn import functional as F

from dual_rgbt.model import DualRGBT
from dual_rgbt_evidence.policy import EvidencePolicy


ACTION_NAMES = ('STOP', 'LOCAL_MAIN', 'LOCAL_ALTERNATE', 'GLOBAL_LOCAL', 'GLOBAL_WIDE')


def global_geometry(image):
    # Top-left square contains the complete current image; crop_region pads its remainder.
    return np.asarray([0., 0., float(max(image.shape[:2]))], dtype=np.float32)


class GlobalEvidencePolicy(EvidencePolicy):
    def preview_state(self, templates, main_preview, global_preview, history):
        state = self.executor.preview_state(templates, main_preview, history)
        features, maps = [], []
        for modality in range(2):
            # Preview.forward always resizes to 64. Global input must use its shared net directly.
            feature = self.executor.preview.net(global_preview[:, modality * 3:(modality + 1) * 3])
            assert feature.shape[-2:] == (32, 32)
            features.append(feature)
            maps.append(F.cosine_similarity(feature, state['descriptors'][modality][:, :, None, None], dim=1))
        state.update(broad_features=features, broad_maps=maps)
        return state

    def query_geometries(self, primary, preview, modality, main_geometry, global_geometry, image_sizes):
        score = primary['score'].flatten(1)
        first = score.argmax(1)
        cells = torch.arange(256, device=score.device)[None]
        nearby = ((cells % 16 - first[:, None] % 16).abs() <= 2) & (
                  (cells // 16 - first[:, None] // 16).abs() <= 2)
        second = score.masked_fill(nearby, -1).argmax(1)
        offsets = primary['offset'].flatten(2).gather(2, second[:, None, None].expand(-1, 2, -1)).squeeze(2)
        second_centre = (torch.stack([second % 16, second // 16], dim=1).to(offsets.dtype) + offsets) / 16
        global_cells = torch.arange(1024, device=score.device)
        normalized = (torch.stack([global_cells % 32, global_cells // 32], dim=1).to(score.dtype) + .5) / 32
        coordinates = normalized[None] * global_geometry[:, None, 2:3] + global_geometry[:, None, :2]
        # The square has actual padding for rectangular images: never propose a padding cell.
        valid = ((coordinates >= 0).all(2) &
                 (coordinates[:, :, 0] < image_sizes[:, None, 1]) &
                 (coordinates[:, :, 1] < image_sizes[:, None, 0]))
        global_score = preview['broad_maps'][1 - modality].flatten(1).masked_fill(~valid, -torch.inf)
        external = global_score.argmax(1)
        external_centre = coordinates.gather(1, external[:, None, None].expand(-1, -1, 2)).squeeze(1)
        centres = [primary['box'][:, :2] * main_geometry[:, 2:3] + main_geometry[:, :2],
                   second_centre * main_geometry[:, 2:3] + main_geometry[:, :2], external_centre]
        local_side = main_geometry[:, 2:3] / 2
        local = [torch.cat([centre - local_side / 2, local_side], dim=1) for centre in centres]
        wide_side = main_geometry[:, 2:3]
        wide = torch.cat([external_centre - wide_side / 2, wide_side], dim=1)
        return torch.stack(local + [wide], dim=1)


def build_policy(executor_checkpoint):
    executor = DualRGBT('/home/gaob/CADTrack_ep0014.pth.tar')
    executor.load_state_dict(torch.load(executor_checkpoint, map_location='cpu')['net'], strict=True)
    return GlobalEvidencePolicy(executor)

"""Complete causal tracker for the independent-region evidence policy."""
import time

import numpy as np
import torch

from dual_rgbt.data import crop_pair
from dual_rgbt_evidence.readout import pixel_xywh
from dual_rgbt_evidence.regions import broad_geometry, crop_region, execute_boxes, search_geometry
from lib.test.utils.hann import hann2d


class EvidenceTracker:
    def __init__(self, model, cost_weight):
        self.model = model.eval()
        self.cost_weight = cost_weight

    @torch.no_grad()
    def initialize(self, image, box):
        torch.cuda.synchronize()
        start = time.perf_counter()
        self.state = np.asarray(box, dtype=np.float32)
        template = crop_pair(image, self.state, 2., 128)[0].cuda()
        self.templates = torch.stack([template, template])[None]
        self.history = template.new_zeros(1, 8)
        self.window = hann2d(torch.tensor([16, 16]), centered=True).cuda()
        self.frame = self.since_query = 0
        torch.cuda.synchronize()
        return time.perf_counter() - start

    @torch.no_grad()
    def track(self, image):
        torch.cuda.synchronize()
        start = time.perf_counter()
        previous = self.state.copy()
        means = image.mean((0, 1))
        geometry = search_geometry(previous)
        broad = broad_geometry(geometry)
        main_preview = crop_region(image, geometry, 64, means)[None].cuda()
        broad_preview = crop_region(image, broad, 64, means)[None].cuda()
        preview = self.model.preview_state(self.templates, main_preview, broad_preview, self.history)
        modality = int(preview['quality'].argmax(1).item())
        main_image = crop_region(image, geometry, 256, means)[None].cuda()
        primary = self.model.executor.primary(self.templates, main_image, modality, self.window)
        main_geo = main_image.new_tensor(geometry)[None]
        broad_geo = main_image.new_tensor(broad)[None]
        primary_pixel_box = pixel_xywh(primary['box'], main_geo)
        primary_confidence = primary['score'].max().item()
        queries = self.model.query_geometries(primary, preview, modality, main_geo, broad_geo)
        gains = self.model.gain_predictions(primary, preview, modality, main_geo, broad_geo, queries)
        action = int((gains - self.cost_weight * self.model.executor.token_cost).argmax(1).item())
        output = primary
        raw_box = primary_pixel_box
        query_pixel_box = primary_pixel_box.new_zeros(primary_pixel_box.shape)
        query_confidence = 0.0
        auxiliary_tokens = 0
        if action:
            query_geo = queries[:, action - 1]
            input_side = 256 if action == 4 else 128
            pixels = crop_region(image, query_geo[0].cpu().numpy(), input_side, means)[None].cuda()
            output = self.model.query(self.templates, pixels, primary, main_geo, query_geo, modality)
            raw_box = output['pixel_box']
            query_pixel_box = raw_box
            query_confidence = output['score'].max().item()
            auxiliary_tokens = output['auxiliary_tokens']
        sizes = main_image.new_tensor([[image.shape[0], image.shape[1]]])
        self.state = execute_boxes(raw_box, sizes)[0].cpu().numpy()
        confidence = output['score'].max().item()
        self.frame += 1
        template_updated = int(self.frame % 5 == 0 and confidence >= .65)
        if template_updated:
            updated = crop_pair(image, self.state, 2., 128)[0].cuda()
            self.templates[:, 1, modality * 3:(modality + 1) * 3] = updated[modality * 3:(modality + 1) * 3]
            if action:
                other = 1 - modality
                self.templates[:, 1, other * 3:(other + 1) * 3] = updated[other * 3:(other + 1) * 3]
        self.since_query = 0 if action else self.since_query + 1
        movement = ((self.state[:2] + self.state[2:] / 2) -
                    (previous[:2] + previous[2:] / 2)) / previous[2:]
        scale = np.log(self.state[2:] / previous[2:])
        values = [float(modality == 0), float(modality == 1), confidence] + movement.tolist() + scale.tolist() + [self.since_query / 100.]
        self.history = self.history.new_tensor(values)[None]
        torch.cuda.synchronize()
        primary_box = primary_pixel_box[0].detach().cpu().tolist()
        query_box = query_pixel_box[0].detach().cpu().tolist()
        return self.state.copy(), dict(modality=modality, action=action, confidence=confidence,
                                       seconds=time.perf_counter() - start,
                                       primary_tokens=primary['primary_tokens'], auxiliary_tokens=auxiliary_tokens,
                                       primary_confidence=primary_confidence, query_confidence=query_confidence,
                                       query_selected=int(bool(action)), template_updated=template_updated,
                                       primary_box_x=primary_box[0], primary_box_y=primary_box[1],
                                       primary_box_w=primary_box[2], primary_box_h=primary_box[3],
                                       query_box_x=query_box[0], query_box_y=query_box[1],
                                       query_box_w=query_box[2], query_box_h=query_box[3],
                                       final_box_x=float(self.state[0]), final_box_y=float(self.state[1]),
                                       final_box_w=float(self.state[2]), final_box_h=float(self.state[3]))

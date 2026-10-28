"""Causal tracking with one benefit-selected raw auxiliary region."""
import time

import numpy as np
import torch

from dual_rgbt.data import crop_pair
from dual_rgbt_evidence.readout import pixel_xywh
from dual_rgbt_evidence.regions import crop_region, execute_boxes, search_geometry
from dual_rgbt_global.policy import global_geometry
from dual_rgbt_global.tracker import EvidenceTracker


class CandidateEvidenceTracker(EvidenceTracker):
    def __init__(self, model, cost_weight, max_candidates):
        super().__init__(model, cost_weight)
        self.max_candidates = max_candidates

    @torch.no_grad()
    def track(self, image):
        torch.cuda.synchronize()
        start = time.perf_counter()
        previous = self.state.copy()
        means = image.mean((0, 1))
        geometry, broad = search_geometry(previous), global_geometry(image)
        main_preview = crop_region(image, geometry, 64, means)[None].cuda()
        broad_preview = crop_region(image, broad, 256, means)[None].cuda()
        preview = self.model.preview_state(self.templates, main_preview, broad_preview, self.history)
        modality = int(preview['quality'].argmax(1).item())
        main_image = crop_region(image, geometry, 256, means)[None].cuda()
        primary = self.model.executor.primary(self.templates, main_image, modality, self.window)
        main_geo, broad_geo = main_image.new_tensor(geometry)[None], main_image.new_tensor(broad)[None]
        sizes = main_image.new_tensor([image.shape[:2]])
        primary_pixel_box = pixel_xywh(primary['box'], main_geo)
        primary_confidence = primary['score'].max().item()
        pool = self.model.candidate_queries(primary, preview, modality, main_geo, broad_geo,
                                            sizes, self.max_candidates)
        gains = self.model.candidate_gains(primary, preview, modality, main_geo, broad_geo,
                                           pool['geometries'], pool['kinds'])
        selected = int((gains - self.cost_weight * self.model.candidate_costs(pool['kinds'])).argmax(1).item())
        output, raw_box = primary, primary_pixel_box
        query_pixel_box = primary_pixel_box.new_zeros(primary_pixel_box.shape)
        query_confidence = 0.0
        auxiliary_tokens, action, rank = 0, 0, -1
        if selected:
            request = selected - 1
            action = int(pool['kinds'][0, request].item()) + 1
            rank = int(pool['ranks'][0, request].item())
            query_geo = pool['geometries'][:, request]
            side = 256 if action == 4 else 128
            pixels = crop_region(image, query_geo[0].cpu().numpy(), side, means)[None].cuda()
            output = self.model.query(self.templates, pixels, primary, main_geo, query_geo, modality)
            raw_box, auxiliary_tokens = output['pixel_box'], output['auxiliary_tokens']
            query_pixel_box = raw_box
            query_confidence = output['score'].max().item()
        self.state = execute_boxes(raw_box, sizes)[0].cpu().numpy()
        confidence = output['score'].max().item()
        self.frame += 1
        template_updated = int(self.frame % 5 == 0 and confidence >= .65)
        if template_updated:
            updated = crop_pair(image, self.state, 2., 128)[0].cuda()
            self.templates[:, 1, modality * 3:(modality + 1) * 3] = updated[modality * 3:(modality + 1) * 3]
            if selected:
                other = 1 - modality
                self.templates[:, 1, other * 3:(other + 1) * 3] = updated[other * 3:(other + 1) * 3]
        self.since_query = 0 if selected else self.since_query + 1
        movement = ((self.state[:2] + self.state[2:] / 2) -
                    (previous[:2] + previous[2:] / 2)) / previous[2:]
        scale = np.log(self.state[2:] / previous[2:])
        values = [float(modality == 0), float(modality == 1), confidence] + movement.tolist() + scale.tolist() + [self.since_query / 100.]
        self.history = self.history.new_tensor(values)[None]
        torch.cuda.synchronize()
        primary_box = primary_pixel_box[0].detach().cpu().tolist()
        query_box = query_pixel_box[0].detach().cpu().tolist()
        return self.state.copy(), dict(modality=modality, action=action, request_index=selected - 1,
            candidate_rank=rank, candidate_count=int(pool['candidate_indices'].numel()), confidence=confidence,
            seconds=time.perf_counter() - start, primary_tokens=primary['primary_tokens'],
            auxiliary_tokens=auxiliary_tokens, primary_confidence=primary_confidence,
            query_confidence=query_confidence, query_selected=int(bool(selected)),
            template_updated=template_updated, primary_box_x=primary_box[0], primary_box_y=primary_box[1],
            primary_box_w=primary_box[2], primary_box_h=primary_box[3], query_box_x=query_box[0],
            query_box_y=query_box[1], query_box_w=query_box[2], query_box_h=query_box[3],
            final_box_x=float(self.state[0]), final_box_y=float(self.state[1]),
            final_box_w=float(self.state[2]), final_box_h=float(self.state[3]))

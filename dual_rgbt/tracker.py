"""Causal online state, prepared for stage C/D; not yet benchmarked."""
import time
import numpy as np
import torch

from dual_rgbt.data import crop_pair
from lib.utils.box_ops import clip_box


class Tracker:
    def __init__(self, model, cost_weight):
        self.model = model.eval()
        self.cost_weight = cost_weight

    @torch.no_grad()
    def initialize(self, image, initial_box):
        torch.cuda.synchronize()
        start = time.perf_counter()
        self.state = np.asarray(initial_box, dtype=np.float32)
        template = crop_pair(image, self.state, 2., 128)[0].cuda()
        self.templates = torch.stack([template, template])[None]
        self.history = template.new_zeros(1, 8)
        self.frame = 0
        self.since_query = 0
        torch.cuda.synchronize()
        return time.perf_counter() - start

    @torch.no_grad()
    def track(self, image):
        torch.cuda.synchronize()
        start = time.perf_counter()
        previous = self.state.copy()
        search, _, geometry = crop_pair(image, previous, 4., 256)
        output = self.model.predict(self.templates, search[None].cuda(), self.history, self.cost_weight)
        box = output['box'][0].cpu().numpy()
        left, top, side = geometry
        xywh = np.concatenate([box[:2] * side + [left, top] - box[2:] * side / 2,
                               box[2:] * side])
        self.state = np.asarray(clip_box(xywh.tolist(), image.shape[0], image.shape[1], margin=10),
                                dtype=np.float32)
        self.frame += 1
        confidence = (output['quality_logit'].sigmoid().item() if output['action'] else
                      output['primary']['score'].max().item())
        if self.frame % 5 == 0 and confidence >= .65:
            updated = crop_pair(image, self.state, 2., 128)[0].cuda()
            primary = output['modality']
            self.templates[:, 1, primary * 3:(primary + 1) * 3] = updated[primary * 3:(primary + 1) * 3]
            if output['action']:
                auxiliary = 1 - primary
                self.templates[:, 1, auxiliary * 3:(auxiliary + 1) * 3] = updated[
                    auxiliary * 3:(auxiliary + 1) * 3]
        self.since_query = 0 if output['action'] else self.since_query + 1
        movement = ((self.state[:2] + self.state[2:] / 2) -
                    (previous[:2] + previous[2:] / 2)) / previous[2:]
        size_change = np.log(self.state[2:] / previous[2:])
        identity = [float(output['modality'] == 0), float(output['modality'] == 1)]
        values = identity + [confidence] + movement.tolist() + size_change.tolist() + [self.since_query / 100.]
        self.history = self.history.new_tensor(values)[None]
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        return self.state.copy(), {'modality': output['modality'], 'action': output['action'],
                                  'confidence': confidence, 'seconds': elapsed,
                                  'primary_tokens': output['primary']['primary_tokens'],
                                  'auxiliary_tokens': output['auxiliary_tokens']}

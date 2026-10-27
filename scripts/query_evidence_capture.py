"""Retain already computed primary predictions for offline query-benefit diagnosis."""
import numpy as np

from dual_rgbt_evidence.readout import pixel_xywh
from dual_rgbt_evidence.regions import execute_boxes, search_geometry


FIELDS = ['frame', 'primary_x', 'primary_y', 'primary_w', 'primary_h',
          'primary_confidence', 'selected_gain', 'selected_token_cost', 'selected_utility']


class QueryEvidenceCapture:
    """Evaluation-only observation; no GT input, extra encoder, or decision changes."""

    def __init__(self, model, cost_weight):
        self.model, self.cost_weight = model, cost_weight
        self.original_primary = model.executor.primary
        self.original_gains = model.candidate_gains
        model.executor.primary = self.primary
        model.candidate_gains = self.gains
        self.primary_output = self.gain_output = None

    def begin(self, image, previous_box):
        self.geometry = search_geometry(previous_box)
        self.image_size = image.shape[:2]
        self.primary_output = self.gain_output = None

    def primary(self, *args, **kwargs):
        assert self.primary_output is None
        output = self.original_primary(*args, **kwargs)
        self.primary_output = {'box': output['box'], 'score': output['score']}
        return output

    def gains(self, *args, **kwargs):
        assert self.gain_output is None
        output = self.original_gains(*args, **kwargs)
        self.gain_output = output
        return output

    def finish(self, frame, route, final_box):
        box = self.primary_output['box']
        geometry = box.new_tensor(self.geometry)[None]
        size = box.new_tensor([self.image_size])
        executed = execute_boxes(pixel_xywh(box, geometry), size)[0].cpu().numpy()
        request = route['request_index'] + 1
        gain = self.gain_output[0, request]
        cost = self.model.executor.token_cost[route['action']]
        if route['action'] == 0:
            assert request == 0 and np.array_equal(executed, final_box)
        row = dict(frame=frame, **dict(zip(FIELDS[1:5], map(float, executed))),
                   primary_confidence=float(self.primary_output['score'].max().item()),
                   selected_gain=float(gain.item()), selected_token_cost=float(cost.item()),
                   selected_utility=float((gain - self.cost_weight * cost).item()))
        self.primary_output = self.gain_output = None
        return row

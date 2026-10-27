"""Route-specific query acceptance for precision-first evaluation experiments.

This module is deliberately detached from the active tracker and training
plans. It only makes a deterministic decision from an already computed query
confidence. Thresholds must be fitted on the LasHeR development split per
route and modality; they are not portable defaults.
"""
from dataclasses import dataclass


ROUTES = ('rgb_primary', 'tir_primary', 'rgb_local128', 'tir_local128',
          'rgb_wide', 'tir_wide')


def route_name(primary_modality, action):
    """Return the route whose query score is being calibrated.

    ``primary_modality`` is 0 for RGB and 1 for TIR. Actions 1, 2, and 3
    are 128px local requests in the legacy and current policies; action 4 is
    the 256px wide request. Action 0 is STOP and is never accepted as a query.
    """
    assert primary_modality in (0, 1)
    assert action in (0, 1, 2, 3, 4)
    if action == 0:
        return 'rgb_primary' if primary_modality == 0 else 'tir_primary'
    names = {1: 'local128', 2: 'local128', 3: 'local128', 4: 'wide'}
    return ('rgb_' if primary_modality == 0 else 'tir_') + names[action]


@dataclass(frozen=True)
class QueryAcceptance:
    """Fixed thresholds selected on a development split."""

    thresholds: dict

    def __post_init__(self):
        assert set(self.thresholds) == set(ROUTES)
        assert all(0.0 <= float(value) <= 1.0 for value in self.thresholds.values())

    def accepts(self, primary_modality, action, query_confidence):
        """Whether the computed query may replace the primary prediction."""
        assert 0.0 <= float(query_confidence) <= 1.0
        if action == 0:
            return False
        return float(query_confidence) >= float(self.thresholds[route_name(primary_modality, action)])

    def accepts_template(self, primary_modality, action, query_confidence, accepted):
        """Keep state acceptance and template writing as separate decisions.

        The first implementation only permits a template write after query
        acceptance. A separate calibrated template threshold can be added
        later without changing the query decision interface.
        """
        assert isinstance(accepted, bool)
        return bool(accepted and self.accepts(primary_modality, action, query_confidence))

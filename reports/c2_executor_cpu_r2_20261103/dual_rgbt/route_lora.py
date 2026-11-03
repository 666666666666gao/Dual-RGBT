"""Low-rank updates selected by the executor's primary/local/full route."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class RouteLoRA(nn.Module):
    def __init__(self, linear, rank=8, routes=3, scale=1.):
        super().__init__()
        assert isinstance(linear, nn.Linear)
        assert rank > 0 and routes > 0
        self.linear = linear
        self.rank = rank
        self.routes = routes
        self.scale = scale
        self.a = nn.Parameter(linear.weight.new_empty(
            routes, rank, linear.in_features))
        self.b = nn.Parameter(linear.weight.new_zeros(
            routes, linear.out_features, rank))
        for matrix in self.a:
            nn.init.kaiming_uniform_(matrix, a=math.sqrt(5))

    def forward(self, x, route):
        assert 0 <= route < self.routes
        update = F.linear(F.linear(x, self.a[route]), self.b[route])
        return self.linear(x) + update * self.scale

    @torch.no_grad()
    def merge(self, route):
        """Return an independent frozen linear for this inference route."""
        assert 0 <= route < self.routes
        merged = nn.Linear(self.linear.in_features, self.linear.out_features,
                           bias=self.linear.bias is not None)
        merged = merged.to(device=self.linear.weight.device,
                           dtype=self.linear.weight.dtype)
        merged.weight.copy_(self.linear.weight +
                            (self.b[route] @ self.a[route]) * self.scale)
        if self.linear.bias is not None:
            merged.bias.copy_(self.linear.bias)
        merged.requires_grad_(False)
        merged.eval()
        return merged

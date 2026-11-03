"""Spatial missing-stream embeddings conditioned on its template descriptor."""
import torch
from torch import nn


class MissingModality(nn.Module):
    def __init__(self, mode="learned_film", tokens=256, channels=768,
                 descriptor_channels=32):
        super().__init__()
        assert mode in ("zero", "learned", "learned_film")
        self.mode = mode
        self.tokens = tokens
        self.channels = channels
        if mode != "zero":
            self.embedding = nn.Parameter(torch.zeros(2, tokens, channels))
        if mode == "learned_film":
            self.film = nn.ModuleList([
                nn.Linear(descriptor_channels, channels * 2) for _ in range(2)])
            for layer in self.film:
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, modality, descriptor):
        assert modality in (0, 1)
        batch = descriptor.shape[0]
        if self.mode == "zero":
            return descriptor.new_zeros(batch, self.tokens, self.channels)
        embedding = self.embedding[modality].unsqueeze(0).expand(batch, -1, -1)
        if self.mode == "learned":
            return embedding
        gamma, beta = self.film[modality](descriptor).chunk(2, dim=-1)
        return embedding * (1 + gamma[:, None, :]) + beta[:, None, :]

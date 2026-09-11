"""Cross-modal connector implementations used by controlled ablations."""
from __future__ import annotations

import torch
from torch import nn


class MLP2Connector(nn.Module):
    """Token-wise Linear-GELU-Dropout-Linear-LayerNorm connector."""

    def __init__(self, input_dim: int, output_dim: int, hidden_multiplier: int = 2, dropout: float = 0.1):
        super().__init__()
        hidden = max(output_dim, hidden_multiplier * input_dim)
        self.net = nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, output_dim), nn.LayerNorm(output_dim))

    def forward(self, tokens: torch.Tensor, view_mask: torch.Tensor | None = None) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"Expected [B,N,D] visual tokens, received {tuple(tokens.shape)}")
        output = self.net(tokens)
        if view_mask is not None:
            if view_mask.shape[:2] != output.shape[:2]:
                raise ValueError("view_mask must have shape [B,N] matching visual tokens")
            output = output.masked_fill(~view_mask[..., None].bool(), 0.0)
        return output


class LinearConnector(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__(); self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, tokens: torch.Tensor, view_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.projection(tokens)


class ResidualMLPConnector(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int | None = None, dropout: float = 0.1):
        super().__init__(); hidden_dim = hidden_dim or max(input_dim, output_dim)
        self.input = nn.Linear(input_dim, output_dim)
        self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(output_dim), nn.Linear(output_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, output_dim)) for _ in range(3)])

    def forward(self, tokens: torch.Tensor, view_mask: torch.Tensor | None = None) -> torch.Tensor:
        output = self.input(tokens)
        for block in self.blocks: output = output + block(output)
        return output


class LearnedQueryResampler(nn.Module):
    """Small learned-query resampler; not labelled Q-Former."""
    def __init__(self, input_dim: int, output_dim: int, num_queries: int = 32, heads: int = 8):
        super().__init__(); self.queries = nn.Parameter(torch.randn(num_queries, output_dim) / output_dim**0.5); self.input = nn.Linear(input_dim, output_dim); self.attention = nn.MultiheadAttention(output_dim, heads, batch_first=True); self.norm = nn.LayerNorm(output_dim)

    def forward(self, tokens: torch.Tensor, view_mask: torch.Tensor | None = None) -> torch.Tensor:
        values = self.input(tokens); queries = self.queries.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        if view_mask is not None: values = values.masked_fill(~view_mask[..., None].bool(), 0.0)
        output, _ = self.attention(queries, values, values, key_padding_mask=None if view_mask is None else ~view_mask.bool())
        return self.norm(output)


def build(name: str, input_dim: int, output_dim: int, **kwargs) -> nn.Module:
    choices = {"linear": LinearConnector, "mlp2": MLP2Connector, "residual_mlp": ResidualMLPConnector, "learned_query_resampler": LearnedQueryResampler}
    if name not in choices: raise KeyError(f"Unknown connector {name!r}; choose {sorted(choices)}")
    return choices[name](input_dim, output_dim, **kwargs)


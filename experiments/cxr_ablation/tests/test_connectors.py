import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.connectors import MLP2Connector, LearnedQueryResampler


def test_mlp2_preserves_token_count():
    output = MLP2Connector(8, 12)(torch.randn(2, 5, 8))
    assert output.shape == (2, 5, 12)


def test_query_resampler_has_fixed_query_count():
    output = LearnedQueryResampler(8, 12, num_queries=4, heads=4)(torch.randn(2, 7, 8))
    assert output.shape == (2, 4, 12)


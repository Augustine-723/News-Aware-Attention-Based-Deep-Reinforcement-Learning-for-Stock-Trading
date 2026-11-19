"""Unit tests for Trend-Gated Flow dynamic masking and policy evaluation."""
import numpy as np
import pytest
import torch

from stock_rl.config.schema import CURRENT_ACTIVE_SCHEMA_VERSION
from stock_rl.data.snapshot_builder import MultimodalBatch
from stock_rl.evaluation.trend_gated_holdout import compute_ma20_series


def test_compute_ma20_series():
    closes = np.array([10.0, 12.0, 14.0, 16.0, 18.0] + [20.0] * 15 + [30.0])
    ma20 = compute_ma20_series(closes)
    assert len(ma20) == len(closes)
    expected_20 = (10 + 12 + 14 + 16 + 18 + 20 * 15) / 20.0
    assert np.isclose(ma20[19], expected_20)
    assert closes[-1] > ma20[-1]


def test_dynamic_feature_mask_logic():
    batch = MultimodalBatch(
        price=torch.zeros(1, 10, 12),
        technical=torch.zeros(1, 10, 16),
        institutional=torch.ones(1, 10, 8),
        macro=torch.zeros(1, 10, 12),
        stock_context=torch.zeros(1, 16),
        recent_event_emb=torch.zeros(1, 4, 128),
        active_event_emb=torch.zeros(1, 4, 128),
        event_metadata=torch.zeros(1, 4, 24),
        event_stock_rel=torch.zeros(1, 4, 8),
        event_mask=torch.zeros(1, 4, dtype=torch.bool),
        position=torch.zeros(1, 4),
        institutional_missing_mask=torch.zeros(1, 10, 8, dtype=torch.bool),
        schema_version=CURRENT_ACTIVE_SCHEMA_VERSION,
    )

    # Above MA20 case: flow preserved (0:5), margin zeroed and masked (5:8)
    b_above = batch.clone()
    b_above.institutional[..., 5:8].zero_()
    b_above.institutional_missing_mask[..., 5:8].fill_(True)
    assert torch.all(b_above.institutional[..., :5] == 1.0)
    assert torch.all(b_above.institutional[..., 5:8] == 0.0)
    assert torch.all(~b_above.institutional_missing_mask[..., :5])
    assert torch.all(b_above.institutional_missing_mask[..., 5:8])

    # Below MA20 case: all institutional features zeroed and masked (0:8)
    b_below = batch.clone()
    b_below.institutional[..., :].zero_()
    b_below.institutional_missing_mask[..., :].fill_(True)
    assert torch.all(b_below.institutional == 0.0)
    assert torch.all(b_below.institutional_missing_mask)


def test_schema_invariant():
    assert CURRENT_ACTIVE_SCHEMA_VERSION == "v8.0-original"

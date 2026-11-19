import pytest
import torch
import pandas as pd
import numpy as np

from stock_rl.config.schema import get_event_age_index
from stock_rl.data.synthetic_data import generate_synthetic_batch
from stock_rl.evaluation.mops_unexamined_eval import evaluate_unexamined_arrivals


def test_evaluate_unexamined_arrivals_zero_flips_on_dummy_policy():
    class DummyModel(torch.nn.Module):
        def forward(self, obs):
            # Model score dominated by market features
            score = obs.price[:, 0, 0] * 0.1 + obs.position[:, 0] * 0.5
            return torch.stack((torch.zeros_like(score), score), dim=1), None, None

    model = DummyModel().eval()
    dates = pd.date_range("2024-01-02 20:00", periods=5, tz="Asia/Taipei")
    prices = np.full((5, 4), 100.0)
    
    batches = []
    age_idx = get_event_age_index("v8.0-original")
    for i in range(5):
        b = generate_synthetic_batch(1)
        b.event_mask.zero_()
        if i == 1:
            b.event_mask[0, 0] = True
            b.event_metadata[0, 0, age_idx] = 0.5  # < 1.0 day: newly arrived
        batches.append(b)

    counts_stock = {
        "2024-01-02": 0,
        "2024-01-03": 1,
        "2024-01-04": 0,
        "2024-01-05": 0,
    }

    prev_date = dates[0] - pd.Timedelta(days=1)
    prev_batch = batches[0]
    sample = (dates, batches, prices, None)

    out = evaluate_unexamined_arrivals(model, sample, prev_date, prev_batch, counts_stock)
    decisions = out["decisions"]
    assert len(decisions) == 4
    assert decisions[0]["new_announcements"] == 0
    assert decisions[1]["new_announcements"] == 1
    assert decisions[1]["real_action"] == decisions[1]["hidden_action"]
    assert decisions[1]["real_action"] == decisions[1]["restored_action"]
    assert out["strategies"]["real"]["executed_position_changes"] == out["strategies"]["hidden"]["executed_position_changes"]


def test_unexamined_arrivals_ledger_discrepancy_raises_value_error():
    class DummyModel(torch.nn.Module):
        def forward(self, obs):
            score = torch.ones((obs.price.shape[0],), dtype=torch.float32)
            return torch.stack((torch.zeros_like(score), score), dim=1), None, None

    model = DummyModel().eval()
    dates = pd.date_range("2024-01-02 20:00", periods=3, tz="Asia/Taipei")
    prices = np.full((3, 4), 100.0)
    batches = [generate_synthetic_batch(1) for _ in range(3)]
    for b in batches:
        b.event_mask.zero_()

    # Intentionally provide mismatch in ledger counts
    counts_stock = {"2024-01-02": 5}
    sample = (dates, batches, prices, None)

    with pytest.raises(ValueError, match="Ledger and visible new slots disagree"):
        evaluate_unexamined_arrivals(model, sample, dates[0] - pd.Timedelta(days=1), batches[0], counts_stock)

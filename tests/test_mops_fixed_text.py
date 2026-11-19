import numpy as np
import pandas as pd
import torch

from stock_rl.data.synthetic_data import generate_synthetic_batch
from stock_rl.evaluation.mops_fixed_text import paired_replay


def test_only_factual_actions_advance_the_shared_position():
    class Probe(torch.nn.Module):
        def forward(self, obs):
            score = obs.recent_event_emb[:, 0, 0] + obs.position[:, 0] * 0.25
            return torch.stack((torch.zeros_like(score), score), dim=1), None, None

    dates = pd.date_range("2024-03-04 20:00", periods=3, tz="Asia/Taipei")
    prices = np.full((3, 4), 100.0)
    batch = generate_synthetic_batch(1)
    batch.recent_event_emb.fill_(1)
    other = batch.clone()
    other.recent_event_emb.fill_(-1)
    other.position.fill_(99)
    rows, _ = paired_replay(Probe(), (dates, [batch] * 3, prices, None),
                            (dates, [other] * 3, prices, None))
    assert [r["real_gap"] for r in rows] == [1, 1.25]
    assert [r["constant_gap"] for r in rows] == [-1, -0.75]
    assert [r["constant_action"] for r in rows] == [0, 0]

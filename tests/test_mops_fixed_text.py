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


def test_timing_probe_preserves_slots_for_age_and_advances_only_factual_action():
    from stock_rl.config.schema import get_event_age_index
    from stock_rl.evaluation.mops_fixed_text import strategy_replays, timing_replay

    batch = generate_synthetic_batch(1)
    age_index = get_event_age_index(batch.schema_version)
    batch.event_mask.zero_()
    batch.event_mask[:, 0] = True
    batch.event_metadata[:, 0, age_index] = 0.5

    class Probe(torch.nn.Module):
        def forward(self, obs):
            visible = obs.event_mask.float()
            score = (visible * obs.event_metadata[..., age_index]).sum(1)
            score += visible.sum(1) * 0.25 - 0.5 + obs.position[:, 0] * 0.125
            return torch.stack((torch.zeros_like(score), score), dim=1), None, None

    dates = pd.date_range("2024-03-04 20:00", periods=3, tz="Asia/Taipei")
    sample = (dates, [batch] * 3, np.full((3, 4), 100.0), None)
    rows, actual = timing_replay(Probe(), sample)
    assert [r["real_gap"] for r in rows] == [0.25, 0.375]
    assert [r["fresh_gap"] for r in rows] == [-0.25, -0.125]
    assert [r["empty_gap"] for r in rows] == [-0.5, -0.375]
    paths = strategy_replays(Probe(), sample, actual)
    assert paths["real"]["executed_position_changes"] == 1
    assert paths["fresh"]["executed_position_changes"] == paths["empty"]["executed_position_changes"] == 0
    assert paths["real"]["net_return"] < paths["empty"]["net_return"] == 0

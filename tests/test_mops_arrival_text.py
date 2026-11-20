import pytest
import torch

from stock_rl.config.schema import get_event_age_index
from stock_rl.data.synthetic_data import generate_synthetic_batch
from stock_rl.evaluation.mops_arrival_text import replace_new_text


def test_replace_only_occupied_slots_younger_than_interval():
    obs = generate_synthetic_batch(1)
    constant = obs.clone()
    obs.event_mask.zero_()
    obs.event_mask[:, [0, 1, 32]] = True
    obs.event_metadata[..., get_event_age_index(obs.schema_version)].zero_()
    obs.event_metadata[:, 1, get_event_age_index(obs.schema_version)] = 3
    constant.event_mask.copy_(obs.event_mask)
    constant.recent_event_emb.fill_(4)
    constant.active_event_emb.fill_(5)
    other = replace_new_text(obs, constant, 3, 2)
    assert torch.equal(other.recent_event_emb[:, 0], constant.recent_event_emb[:, 0])
    assert torch.equal(other.active_event_emb[:, 0], constant.active_event_emb[:, 0])
    assert torch.equal(other.recent_event_emb[:, 1:], obs.recent_event_emb[:, 1:])
    assert torch.equal(other.active_event_emb[:, 1:], obs.active_event_emb[:, 1:])
    assert torch.equal(other.event_metadata, obs.event_metadata)
    assert torch.equal(other.event_mask, obs.event_mask)
    with pytest.raises(ValueError, match="ledger"):
        replace_new_text(obs, constant, 3, 1)

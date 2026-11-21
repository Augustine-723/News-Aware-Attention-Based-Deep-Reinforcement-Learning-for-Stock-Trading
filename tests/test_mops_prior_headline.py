import pytest
import torch

from stock_rl.config.schema import get_event_age_index
from stock_rl.data.synthetic_data import generate_synthetic_batch
from stock_rl.evaluation.mops_prior_headline import text_controls


def test_prior_donor_skips_identical_text_and_preserves_older_slots():
    obs = generate_synthetic_batch(1)
    obs.event_mask.zero_()
    obs.event_mask[:, :2] = True
    age = get_event_age_index(obs.schema_version)
    obs.event_metadata[:, :2, age] = torch.tensor([0., 4.])
    previous = obs.clone()
    previous.recent_event_emb[:, 1].fill_(3)
    swapped, blank, donors = text_controls(obs, previous, 1, 1)
    assert donors == [1]
    assert torch.equal(swapped.recent_event_emb[:, 0], previous.recent_event_emb[:, 1])
    assert not blank.recent_event_emb[:, 0].any()
    for view in (swapped, blank):
        assert torch.equal(obs.event_metadata, view.event_metadata)
        assert torch.equal(obs.recent_event_emb[:, 1:], view.recent_event_emb[:, 1:])
    previous.event_mask[:, 1] = False
    with pytest.raises(ValueError, match="distinct"):
        text_controls(obs, previous, 1, 1)

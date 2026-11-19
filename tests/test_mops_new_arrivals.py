import pytest
import torch

from stock_rl.config.schema import get_event_age_index
from stock_rl.data.synthetic_data import generate_synthetic_batch
from stock_rl.evaluation.mops_new_arrivals import hide_arrivals


def test_holiday_interval_is_open_at_previous_cutoff_and_closed_at_current_cutoff():
    obs = generate_synthetic_batch(1)
    age = get_event_age_index(obs.schema_version)
    obs.event_mask.zero_()
    obs.event_mask[:, :4] = True
    obs.event_metadata[0, :4, age] = torch.tensor([0, 5 - 1/86400, 5, 6])
    other, count = hide_arrivals(obs, 5)
    assert count == 2
    assert other.event_mask[0, :4].tolist() == [False, False, True, True]
    assert torch.equal(obs.event_metadata[:, 2:], other.event_metadata[:, 2:])
    assert obs.event_mask.sum() == 4
    assert torch.equal(obs.position, other.position)
    with pytest.raises(ValueError):
        hide_arrivals(obs, 0)


def test_displaced_history_ages_over_holiday_while_market_and_position_stay_current():
    from stock_rl.evaluation.mops_new_arrivals import previous_history

    current = generate_synthetic_batch(1, seed=1)
    previous = generate_synthetic_batch(1, seed=2)
    previous.event_mask.zero_()
    previous.event_mask[:, 0] = True
    previous.event_metadata.zero_()
    age = get_event_age_index(current.schema_version)
    previous.event_metadata[:, 0, age] = 2
    restored = previous_history(current, previous, 5)
    assert restored.event_metadata[0, 0, age] == 7
    assert restored.event_metadata[0, 1, age] == 0
    assert torch.equal(restored.event_mask, previous.event_mask)
    assert torch.equal(restored.price, current.price)
    assert torch.equal(restored.position, current.position)
    assert previous.event_metadata[0, 0, age] == 2

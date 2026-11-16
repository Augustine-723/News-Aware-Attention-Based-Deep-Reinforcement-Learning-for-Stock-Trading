"""
Regression Tests for V8.1 Final Review:
Issue 1 (P0): Event Metadata Index Mismatch Fix & Temporal Causal Memory Verification
- Unified Feature Index Lookup Interface (get_feature_index, table maps, EVENT_AGE_METADATA_INDEX).
- Event temporal ordering strictly governed by the real Event Age field (EV_META_13 / index 12).
- Verification that modifying Event Categories (EV_META_01..08, including former hardcoded index 4)
  does NOT alter causal memory temporal ordering or prior event mask.
- Verification that modifying historical available time (EV_META_13) correctly updates the prior event mask.
"""

import pytest
import torch
import numpy as np

from stock_rl.config.schema import (
    get_feature_index,
    EVENT_AGE_METADATA_INDEX,
    EVENT_METADATA_FEATURE_FIELDS,
    EVENT_METADATA_INDEX_BY_ID,
    EVENT_METADATA_INDEX_BY_NAME,
    PRICE_FEATURE_FIELDS,
    TECHNICAL_FEATURE_FIELDS,
    INSTITUTIONAL_FEATURE_FIELDS,
    MACRO_FEATURE_FIELDS,
    STOCK_CONTEXT_FEATURE_FIELDS,
    EVENT_STOCK_RELATION_FEATURE_FIELDS,
    POSITION_STATE_FEATURE_FIELDS,
    TOTAL_EVENT_SLOTS,
)
from stock_rl.models.full_model import StockRLFullModel
from stock_rl.data.synthetic_data import generate_synthetic_batch


def test_unified_feature_index_lookup_interface():
    """Verify unified feature index lookup interface by ID and Name across all tables."""
    # 1. Verify Event Age specifically
    assert EVENT_AGE_METADATA_INDEX == 12
    assert get_feature_index("EV_META_13") == 12
    assert get_feature_index("EV_META_13", "event_metadata") == 12
    assert get_feature_index("event_age_fractional_days") == 12
    assert get_feature_index("event_age_fractional_days", "event_metadata") == 12
    assert EVENT_METADATA_FEATURE_FIELDS[12].feature_id == "EV_META_13"
    assert EVENT_METADATA_FEATURE_FIELDS[12].name == "event_age_fractional_days"

    # 2. Verify EV_META_01 to EV_META_08 are the 8 category fields
    for i in range(8):
        f = EVENT_METADATA_FEATURE_FIELDS[i]
        assert f.feature_id == f"EV_META_{i+1:02d}"
        assert f.name.startswith("category_"), f"EV_META_{i+1:02d} must be category indicator, got {f.name}"
        assert get_feature_index(f.feature_id, "event_metadata") == i
        assert get_feature_index(f.name, "event_metadata") == i

    # Former index 4 is category_merger_acquisition
    assert EVENT_METADATA_FEATURE_FIELDS[4].feature_id == "EV_META_05"
    assert EVENT_METADATA_FEATURE_FIELDS[4].name == "category_merger_acquisition"

    # 3. Verify Position State
    assert get_feature_index("POS_04", "position_state") == 3
    assert get_feature_index("time_since_last_trade", "position_state") == 3

    # 4. Verify invalid key or table raises KeyError
    with pytest.raises(KeyError):
        get_feature_index("NON_EXISTENT_FEATURE_ID")

    with pytest.raises(KeyError):
        get_feature_index("EV_META_13", "invalid_table_name")


def test_event_temporal_ordering_strictly_by_true_event_age():
    """
    Verify that temporal causal memory is strictly governed by EV_META_13 (index 12),
    NOT by category or hardcoded index 4.
    """
    torch.manual_seed(100)
    model = StockRLFullModel()
    model.eval()

    batch = generate_synthetic_batch(batch_size=1, seed=42)

    with torch.no_grad():
        batch.event_mask[:, :] = False
        batch.event_mask[:, 0] = True
        batch.event_mask[:, 1] = True
        batch.event_mask[:, 2] = True

        # Initialize all ages to 0
        batch.event_metadata[:, :, EVENT_AGE_METADATA_INDEX] = 0.0

        # Slot 0: Oldest event (30 days ago)
        batch.event_metadata[:, 0, EVENT_AGE_METADATA_INDEX] = 30.0
        # Slot 1: Intermediate event (15 days ago)
        batch.event_metadata[:, 1, EVENT_AGE_METADATA_INDEX] = 15.0
        # Slot 2: Newest event (5 days ago)
        batch.event_metadata[:, 2, EVENT_AGE_METADATA_INDEX] = 5.0

        _, _, projected_events, _, causal_prior_memory = model.encode_multimodal_state(batch)

        # Slot 0 has NO prior events (30 > nothing). Its prior memory must be NRE token
        nre_token_vec = model.event_module.nre_token.squeeze(1)[0]
        assert torch.allclose(causal_prior_memory[0, 0], nre_token_vec, atol=1e-5), (
            "Oldest event slot 0 must fall back to NRE token"
        )

        # Slot 1 (age 15) has exactly one prior event: Slot 0 (age 30 > 15).
        # Its prior memory must strictly equal Slot 0's projected event representation
        assert torch.allclose(causal_prior_memory[0, 1], projected_events[0, 0], atol=1e-5), (
            "Slot 1's prior memory must strictly equal Slot 0's projected event"
        )

        # Slot 2 (age 5) has two prior events: Slot 0 (age 30) and Slot 1 (age 15).
        # Its prior memory must equal the mean of Slot 0 and Slot 1 projected representations
        expected_slot2_prior = (projected_events[0, 0] + projected_events[0, 1]) / 2.0
        assert torch.allclose(causal_prior_memory[0, 2], expected_slot2_prior, atol=1e-5), (
            "Slot 2's prior memory must equal average of Slot 0 and Slot 1 projected representations"
        )


def test_modifying_event_category_does_not_alter_temporal_sorting():
    """
    Requirement 5: Modifying Event Categories (EV_META_01..08, and specifically
    former index 4 EV_META_05) MUST NOT alter causal memory temporal ordering or prior mask.
    """
    torch.manual_seed(100)
    model = StockRLFullModel()
    model.eval()

    batch = generate_synthetic_batch(batch_size=1, seed=42)

    with torch.no_grad():
        batch.event_mask[:, :] = False
        batch.event_mask[:, 0] = True
        batch.event_mask[:, 1] = True

        # Keep real event age strictly fixed:
        batch.event_metadata[:, :, EVENT_AGE_METADATA_INDEX] = 0.0
        batch.event_metadata[:, 0, EVENT_AGE_METADATA_INDEX] = 20.0  # Slot 0 is older
        batch.event_metadata[:, 1, EVENT_AGE_METADATA_INDEX] = 10.0  # Slot 1 is newer

        # Baseline: initial category values
        batch.event_metadata[:, :, 0:8] = 0.0
        batch.event_metadata[:, 0, 4] = 1.0  # EV_META_05 (former hardcoded index 4)

        _, _, proj_orig, _, causal_prior_orig = model.encode_multimodal_state(batch)

        # Now heavily modify the event category fields (including index 4)
        batch_cat_modified = generate_synthetic_batch(batch_size=1, seed=42)
        batch_cat_modified.event_mask[:, :] = False
        batch_cat_modified.event_mask[:, 0] = True
        batch_cat_modified.event_mask[:, 1] = True

        # Ages remain identical
        batch_cat_modified.event_metadata[:, :, EVENT_AGE_METADATA_INDEX] = 0.0
        batch_cat_modified.event_metadata[:, 0, EVENT_AGE_METADATA_INDEX] = 20.0
        batch_cat_modified.event_metadata[:, 1, EVENT_AGE_METADATA_INDEX] = 10.0

        # Invert category features: set index 4 to 100.0 on slot 1, 0.0 on slot 0
        batch_cat_modified.event_metadata[:, 0, 4] = 0.0
        batch_cat_modified.event_metadata[:, 1, 4] = 100.0

        _, _, proj_mod, _, causal_prior_mod = model.encode_multimodal_state(batch_cat_modified)

        # Even though category at index 4 was inverted, the temporal relation MUST remain:
        # Slot 0 still has age 20 > 10, so Slot 0 is prior to Slot 1.
        # Slot 1's prior memory MUST still be Slot 0's projected representation!
        assert torch.allclose(causal_prior_mod[0, 1], proj_mod[0, 0], atol=1e-5), (
            "Modifying category features must not break the causal relationship that Slot 0 is prior to Slot 1"
        )
        # And Slot 0 still has NO prior events
        nre_token_vec = model.event_module.nre_token.squeeze(1)[0]
        assert torch.allclose(causal_prior_mod[0, 0], nre_token_vec, atol=1e-5), (
            "Slot 0 must still have no prior events regardless of category feature modifications"
        )


def test_modifying_historical_available_time_updates_prior_mask():
    """
    Requirement 6: Modifying event historical available time (EV_META_13)
    DOES correctly update the prior event mask and prior memory direction.
    """
    torch.manual_seed(100)
    model = StockRLFullModel()
    model.eval()

    batch = generate_synthetic_batch(batch_size=1, seed=42)

    with torch.no_grad():
        batch.event_mask[:, :] = False
        batch.event_mask[:, 0] = True
        batch.event_mask[:, 1] = True

        # Phase 1: Slot 0 is older (age 30), Slot 1 is newer (age 10)
        batch.event_metadata[:, :, EVENT_AGE_METADATA_INDEX] = 0.0
        batch.event_metadata[:, 0, EVENT_AGE_METADATA_INDEX] = 30.0
        batch.event_metadata[:, 1, EVENT_AGE_METADATA_INDEX] = 10.0

        _, _, proj_phase1, _, causal_phase1 = model.encode_multimodal_state(batch)

        # In Phase 1: Slot 1's prior is Slot 0
        assert torch.allclose(causal_phase1[0, 1], proj_phase1[0, 0], atol=1e-5)
        nre_token_vec = model.event_module.nre_token.squeeze(1)[0]
        assert torch.allclose(causal_phase1[0, 0], nre_token_vec, atol=1e-5)

        # Phase 2: Invert historical available time!
        # Make Slot 1 older (age 40) and Slot 0 newer (age 5)
        batch.event_metadata[:, 0, EVENT_AGE_METADATA_INDEX] = 5.0
        batch.event_metadata[:, 1, EVENT_AGE_METADATA_INDEX] = 40.0

        _, _, proj_phase2, _, causal_phase2 = model.encode_multimodal_state(batch)

        # In Phase 2: Slot 0's prior is now Slot 1!
        assert torch.allclose(causal_phase2[0, 0], proj_phase2[0, 1], atol=1e-5), (
            "Updating EV_META_13 available time must update prior mask so Slot 0 now sees Slot 1 as prior"
        )
        # And Slot 1 now has no prior events
        assert torch.allclose(causal_phase2[0, 1], nre_token_vec, atol=1e-5), (
            "Updating EV_META_13 available time must make Slot 1 fall back to NRE token"
        )

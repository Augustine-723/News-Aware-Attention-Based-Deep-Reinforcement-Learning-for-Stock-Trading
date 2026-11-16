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

    batch = generate_synthetic_batch(batch_size=1, seed=42, schema_version="v8.1-candidate")

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

    batch = generate_synthetic_batch(batch_size=1, seed=42, schema_version="v8.1-candidate")

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
        batch_cat_modified = generate_synthetic_batch(batch_size=1, seed=42, schema_version="v8.1-candidate")
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

    batch = generate_synthetic_batch(batch_size=1, seed=42, schema_version="v8.1-candidate")

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


def test_dual_feature_schema_versions_and_baseline_contract():
    """
    Requirement: Formal Dual Schema Architecture Setup
    - Original V8: Formal Baseline (v8.0-original) per ChatGPT V8 protocol
    - Current V8.1: Alternative Feature Set / Ablation (v8.1-candidate)
    - Distinct Feature Schema Version constants
    - Baseline V8.0 formal 12-dimensional quantitative price features contract (P01-P12)
    """
    from stock_rl.config.schema import (
        FeatureSchemaVersion,
        CURRENT_ACTIVE_SCHEMA_VERSION,
        BASELINE_V8_0_PRICE_FEATURE_FIELDS,
        PRICE_FEATURE_FIELDS,
    )

    # 1. Verify schema version identifiers
    assert FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value == "v8.0-original"
    assert FeatureSchemaVersion.ALTERNATIVE_V8_1_CANDIDATE.value == "v8.1-candidate"
    assert CURRENT_ACTIVE_SCHEMA_VERSION == FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value

    # 2. Verify BASELINE_V8_0 price fields: 12 fields locked per ChatGPT V8
    assert len(BASELINE_V8_0_PRICE_FEATURE_FIELDS) == 12
    baseline_names = [f.name for f in BASELINE_V8_0_PRICE_FEATURE_FIELDS]
    expected_baseline_names = [
        "return_1d",
        "return_5d",
        "return_20d",
        "overnight_gap",
        "open_close_return",
        "high_low_range",
        "close_location",
        "log_volume",
        "volume_ratio_20d",
        "turnover_ratio",
        "realized_volatility_20d",
        "log_trading_value",
    ]
    assert baseline_names == expected_baseline_names, f"Baseline names mismatch: {baseline_names}"

    # 3. Verify that all baseline price fields are tagged with version 'v8.0-original'
    for f in BASELINE_V8_0_PRICE_FEATURE_FIELDS:
        assert f.version == "v8.0-original", f"Field {f.feature_id} must have version 'v8.0-original', got {f.version}"
        assert f.math_definition, f"Field {f.feature_id} missing math_definition"
        assert f.unit, f"Field {f.feature_id} missing unit"
        assert f.source, f"Field {f.feature_id} missing source"

    # 4. Verify candidate ablation price fields: 12 fields
    assert len(PRICE_FEATURE_FIELDS) == 12
    cand_names = [f.name for f in PRICE_FEATURE_FIELDS]
    expected_cand_names = [
        "open_norm",
        "high_norm",
        "low_norm",
        "close_norm",
        "volume_norm",
        "turnover_norm",
        "vwap_norm",
        "daily_return",
        "intraday_range",
        "shadow_upper",
        "shadow_lower",
        "effective_spread_proxy",
    ]
    assert cand_names == expected_cand_names, f"Candidate names mismatch: {cand_names}"

    # 5. Semantic difference verification:
    # All 12 Price features in v8.0-original are distinct in name/definition from v8.1-candidate
    for i in range(12):
        assert baseline_names[i] != cand_names[i], (
            f"Index {i} ({baseline_names[i]}) must be semantically distinct from candidate ({cand_names[i]})"
        )


def test_authoritative_100_feature_schema_v8_2_contracts():
    """
    Requirement: V8.2 Authoritative 100-Feature Schema Finalization Contracts
    - Baseline v8.0-original full 100 dimensions across 8 modules
    - Dynamic event age index: 22 for v8.0-original (E23) vs 12 for v8.1-candidate (EV_META_13)
    - Stock context C01-C08 8-dim continuous industry embedding
    - Event metadata E01-E08 Actor, E09-E16 Type, E17-E22 Status embeddings
    - Price P07 close location bounded on [0, 1] (ratio_0_1)
    - Shorthand index lookup for P01-P12, T01-T16, I01-I08, M01-M12, C01-C16, E01-E24, R01-R08, S01-S04
    """
    from stock_rl.config.schema import (
        FeatureSchemaVersion,
        BASELINE_V8_0_ALL_FEATURE_FIELDS,
        BASELINE_V8_0_PRICE_FEATURE_FIELDS,
        BASELINE_V8_0_TECHNICAL_FEATURE_FIELDS,
        BASELINE_V8_0_INSTITUTIONAL_FEATURE_FIELDS,
        BASELINE_V8_0_MACRO_FEATURE_FIELDS,
        BASELINE_V8_0_STOCK_CONTEXT_FEATURE_FIELDS,
        BASELINE_V8_0_EVENT_METADATA_FEATURE_FIELDS,
        BASELINE_V8_0_EVENT_STOCK_RELATION_FEATURE_FIELDS,
        BASELINE_V8_0_POSITION_STATE_FEATURE_FIELDS,
        ORIGINAL_V8_EVENT_AGE_METADATA_INDEX,
        CANDIDATE_V8_1_EVENT_AGE_METADATA_INDEX,
        get_event_age_index,
        get_feature_index,
    )

    # 1. Total dimensions and module dimensions
    assert len(BASELINE_V8_0_ALL_FEATURE_FIELDS) == 100
    assert len(BASELINE_V8_0_PRICE_FEATURE_FIELDS) == 12
    assert len(BASELINE_V8_0_TECHNICAL_FEATURE_FIELDS) == 16
    assert len(BASELINE_V8_0_INSTITUTIONAL_FEATURE_FIELDS) == 8
    assert len(BASELINE_V8_0_MACRO_FEATURE_FIELDS) == 12
    assert len(BASELINE_V8_0_STOCK_CONTEXT_FEATURE_FIELDS) == 16
    assert len(BASELINE_V8_0_EVENT_METADATA_FEATURE_FIELDS) == 24
    assert len(BASELINE_V8_0_EVENT_STOCK_RELATION_FEATURE_FIELDS) == 8
    assert len(BASELINE_V8_0_POSITION_STATE_FEATURE_FIELDS) == 4

    # All baseline fields tagged with v8.0-original
    for f in BASELINE_V8_0_ALL_FEATURE_FIELDS:
        assert f.version == "v8.0-original"
        assert f.feature_id
        assert f.name
        assert f.math_definition
        assert f.unit
        assert f.source

    # 2. Dynamic Event Age index resolution
    assert ORIGINAL_V8_EVENT_AGE_METADATA_INDEX == 22
    assert CANDIDATE_V8_1_EVENT_AGE_METADATA_INDEX == 12
    assert get_event_age_index("v8.0-original") == 22
    assert get_event_age_index(FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL) == 22
    assert get_event_age_index("v8.1-candidate") == 12
    assert get_event_age_index(FeatureSchemaVersion.ALTERNATIVE_V8_1_CANDIDATE) == 12
    assert get_event_age_index() == 22  # Active default baseline (v8.0-original)

    # Event age index via get_feature_index
    assert get_feature_index("E23") == 22
    assert get_feature_index("EV_META_13") == 12
    assert get_feature_index("event_age_fractional_days", schema_version="v8.0-original") == 22
    assert get_feature_index("event_age_fractional_days", schema_version="v8.1-candidate") == 12

    # 3. Stock Context: C01-C08 8-dimensional industry embedding
    for i in range(8):
        f = BASELINE_V8_0_STOCK_CONTEXT_FEATURE_FIELDS[i]
        assert f.name == f"industry_embedding_{i+1:02d}"
        assert f.unit == "embedding"

    # C09-C16 scalar context features
    expected_c_scalars = [
        "log_market_capitalization",
        "log_avg_trading_value_60d",
        "volatility_percentile",
        "liquidity_percentile",
        "beta_120d",
        "market_cap_percentile_within_industry",
        "listing_age",
        "recent_trading_suspension_indicator",
    ]
    actual_c_scalars = [f.name for f in BASELINE_V8_0_STOCK_CONTEXT_FEATURE_FIELDS[8:16]]
    assert actual_c_scalars == expected_c_scalars, f"Stock Context mismatch: {actual_c_scalars}"

    # 4. Macro M01-M12 verification
    expected_macro_names = [
        "taiex_return_1d",
        "taiex_return_5d",
        "taiex_volatility_20d",
        "sector_return_1d",
        "sector_relative_strength_20d",
        "usdtwd_return",
        "sp500_return",
        "nasdaq_return",
        "sox_return",
        "vix_level",
        "us_10y_yield",
        "us_10y_yield_change",
    ]
    actual_macro_names = [f.name for f in BASELINE_V8_0_MACRO_FEATURE_FIELDS]
    assert actual_macro_names == expected_macro_names, f"Macro mismatch: {actual_macro_names}"

    # Verify cross-market features M07-M12 have T-1 <= 20:00 availability
    for f in BASELINE_V8_0_MACRO_FEATURE_FIELDS[6:]:
        assert "<= 20:00" in f.available_timing

    # 5. Event-Stock Relation R01-R08 verification
    expected_rel_names = [
        "direct_company_mention",
        "industry_match",
        "upstream_exposure",
        "downstream_exposure",
        "customer_relationship",
        "supplier_relationship",
        "geographic_exposure",
        "macro_factor_exposure",
    ]
    actual_rel_names = [f.name for f in BASELINE_V8_0_EVENT_STOCK_RELATION_FEATURE_FIELDS]
    assert actual_rel_names == expected_rel_names, f"Event Relation mismatch: {actual_rel_names}"

    # 6. Event Metadata: Authoritative 4/8/4 + 8 Layout:
    # E01-E04: Actor / Institution Role Embedding (4D)
    # E05-E12: Event Type Embedding (8D)
    # E13-E16: Event Status Embedding (4D)
    # E17: Evidence Traceability
    # E18: Source Reliability Estimate
    # E19: Source Independence
    # E20: Novelty Input Estimate
    # E21: Market Surprise (NO post-event reaction)
    # E22: Quantified Magnitude
    # E23: Event Age (Index 22)
    # E24: Timestamp Reliability (Index 23)

    # Actor embeddings E01-E04 (4D)
    for i in range(4):
        f = BASELINE_V8_0_EVENT_METADATA_FEATURE_FIELDS[i]
        assert f.name == f"actor_role_embedding_{i+1:02d}"
        assert f.unit == "embedding"

    # Type embeddings E05-E12 (8D)
    for i in range(8):
        f = BASELINE_V8_0_EVENT_METADATA_FEATURE_FIELDS[4 + i]
        assert f.name == f"event_type_embedding_{i+1:02d}"
        assert f.unit == "embedding"

    # Status embeddings E13-E16 (4D)
    for i in range(4):
        f = BASELINE_V8_0_EVENT_METADATA_FEATURE_FIELDS[12 + i]
        assert f.name == f"event_status_embedding_{i+1:02d}"
        assert f.unit == "embedding"

    # Scalar features E17-E22
    assert BASELINE_V8_0_EVENT_METADATA_FEATURE_FIELDS[16].name == "evidence_traceability"
    assert BASELINE_V8_0_EVENT_METADATA_FEATURE_FIELDS[17].name == "source_reliability_estimate"
    assert BASELINE_V8_0_EVENT_METADATA_FEATURE_FIELDS[18].name == "source_independence"
    assert BASELINE_V8_0_EVENT_METADATA_FEATURE_FIELDS[19].name == "novelty_input_estimate"
    # E21 Market Surprise must NOT use post-event reaction
    f_e21 = BASELINE_V8_0_EVENT_METADATA_FEATURE_FIELDS[20]
    assert f_e21.name == "market_surprise"
    assert "NO post-event" in f_e21.math_definition
    assert BASELINE_V8_0_EVENT_METADATA_FEATURE_FIELDS[21].name == "quantified_magnitude"

    # E23 Event Age at index 22
    f_e23 = BASELINE_V8_0_EVENT_METADATA_FEATURE_FIELDS[22]
    assert f_e23.feature_id == "EV_META_23"
    assert f_e23.name in ("event_age", "event_age_fractional_days")
    assert "E23" in f_e23.math_definition
    assert f_e23.unit == "days"

    # E24 Timestamp Reliability at index 23
    f_e24 = BASELINE_V8_0_EVENT_METADATA_FEATURE_FIELDS[23]
    assert f_e24.feature_id == "EV_META_24"
    assert f_e24.name == "timestamp_reliability"
    assert f_e24.unit == "score"

    # 7. Technical T01-T16 strict sequence and T14 Bollinger Bandwidth verification
    expected_tech_names = [
        "rsi_14",
        "rsi_7",
        "kdj_k",
        "kdj_d",
        "kdj_j",
        "macd_dif",
        "macd_signal",
        "macd_hist",
        "close_ema20_ratio",
        "close_ema60_ratio",
        "ema20_ema60_ratio",
        "atr14_close_ratio",
        "bollinger_percent_b",
        "bollinger_bandwidth",
        "normalized_obv_change_5d",
        "adx_14",
    ]
    actual_tech_names = [f.name for f in BASELINE_V8_0_TECHNICAL_FEATURE_FIELDS]
    assert actual_tech_names == expected_tech_names
    assert BASELINE_V8_0_TECHNICAL_FEATURE_FIELDS[13].name == "bollinger_bandwidth"
    assert "cci" not in [name.lower() for name in actual_tech_names], "CCI must not be in Original V8 technical"

    # 8. Institutional I01-I08 strict sequence and 20D Average Volume denominator verification
    expected_inst_names = [
        "foreign_net_buy_over_vol20d",
        "trust_net_buy_over_vol20d",
        "dealer_net_buy_over_vol20d",
        "foreign_net_buy_5d",
        "total_institutional_net_buy_5d",
        "margin_balance_change",
        "short_balance_change",
        "short_margin_balance_ratio",
    ]
    actual_inst_names = [f.name for f in BASELINE_V8_0_INSTITUTIONAL_FEATURE_FIELDS]
    assert actual_inst_names == expected_inst_names
    # Verify denominator for I01-I03 is MA20 Volume
    for idx in range(3):
        f = BASELINE_V8_0_INSTITUTIONAL_FEATURE_FIELDS[idx]
        assert "MA20(Volume)" in f.math_definition, f"{f.feature_id} must use MA20(Volume) denominator"

    # 9. Price P07 Close Location range [0, 1]
    p07 = BASELINE_V8_0_PRICE_FEATURE_FIELDS[6]
    assert p07.feature_id == "PRC_07"
    assert p07.name == "close_location"
    assert p07.unit == "ratio_0_1"
    assert "(P_close,t - P_low,t) / (P_high,t - P_low,t" in p07.math_definition

    # 10. Position State S04: time_since_last_successful_trade
    s04 = BASELINE_V8_0_POSITION_STATE_FEATURE_FIELDS[3]
    assert s04.name == "time_since_last_successful_trade"

    # 11. Shorthand resolution verification across all groups
    shorthand_tests = [
        ("P", 12),
        ("T", 16),
        ("I", 8),
        ("M", 12),
        ("C", 16),
        ("E", 24),
        ("R", 8),
        ("S", 4),
    ]
    for prefix, count in shorthand_tests:
        for idx in range(1, count + 1):
            code = f"{prefix}{idx:02d}"
            expected_idx = idx - 1
            assert get_feature_index(code) == expected_idx, f"Shorthand {code} must resolve to {expected_idx}"


def test_strict_v8_2_semantic_contracts_and_schema_isolation():
    """
    Requirement 6: Strict Semantic Contract Tests & Schema Isolation
    - Verify complete 100-dim definition for both Original and Candidate.
    - Verify Event Metadata 4/8/4+8 dimension layout.
    - Verify Technical T14 is Bollinger Bandwidth (no CCI in Original).
    - Verify Institutional I01-I03 denominator is 20D Average Volume.
    - Verify isolation: Event Age Index is 22 for Original (E23) and 12 for Candidate (EV_META_13).
    - Ensure dynamic lookup never cross-contaminates between schema versions.
    """
    from stock_rl.config.schema import (
        FeatureSchemaVersion,
        BASELINE_V8_0_ALL_FEATURE_FIELDS,
        ALL_FEATURE_FIELDS,
        BASELINE_V8_0_TECHNICAL_FEATURE_FIELDS,
        BASELINE_V8_0_INSTITUTIONAL_FEATURE_FIELDS,
        BASELINE_V8_0_EVENT_METADATA_FEATURE_FIELDS,
        EVENT_METADATA_FEATURE_FIELDS,
        ORIGINAL_V8_EVENT_AGE_METADATA_INDEX,
        CANDIDATE_V8_1_EVENT_AGE_METADATA_INDEX,
        get_event_age_index,
        get_feature_index,
    )

    # 1. Complete 100-dimension definitions
    assert len(BASELINE_V8_0_ALL_FEATURE_FIELDS) == 100
    assert len(ALL_FEATURE_FIELDS) == 100

    # 2. Event Metadata 4/8/4 + 8 partition test
    actor_fields = BASELINE_V8_0_EVENT_METADATA_FEATURE_FIELDS[0:4]
    type_fields = BASELINE_V8_0_EVENT_METADATA_FEATURE_FIELDS[4:12]
    status_fields = BASELINE_V8_0_EVENT_METADATA_FEATURE_FIELDS[12:16]
    scalar_fields = BASELINE_V8_0_EVENT_METADATA_FEATURE_FIELDS[16:24]
    assert len(actor_fields) == 4
    assert len(type_fields) == 8
    assert len(status_fields) == 4
    assert len(scalar_fields) == 8

    # 3. Technical T14 is Bollinger Bandwidth
    assert BASELINE_V8_0_TECHNICAL_FEATURE_FIELDS[13].name == "bollinger_bandwidth"
    assert BASELINE_V8_0_TECHNICAL_FEATURE_FIELDS[13].feature_id == "TECH_14"

    # 4. Institutional I01-I03 denominator is 20D Average Volume
    for f in BASELINE_V8_0_INSTITUTIONAL_FEATURE_FIELDS[:3]:
        assert "MA20(Volume)" in f.math_definition

    # 5. Event Age isolation between schemas
    assert ORIGINAL_V8_EVENT_AGE_METADATA_INDEX == 22
    assert CANDIDATE_V8_1_EVENT_AGE_METADATA_INDEX == 12

    orig_age_idx = get_event_age_index("v8.0-original")
    cand_age_idx = get_event_age_index("v8.1-candidate")
    assert orig_age_idx == 22
    assert cand_age_idx == 12
    assert orig_age_idx != cand_age_idx

    # Dynamic lookup isolation:
    # Querying Original schema
    assert get_feature_index("event_age", schema_version="v8.0-original") == 22
    assert get_feature_index("E23", schema_version="v8.0-original") == 22

    # Querying Candidate schema
    assert get_feature_index("EV_META_13", schema_version="v8.1-candidate") == 12
    assert get_feature_index("event_age_fractional_days", schema_version="v8.1-candidate") == 12
    assert EVENT_METADATA_FEATURE_FIELDS[12].feature_id == "EV_META_13"
    assert BASELINE_V8_0_EVENT_METADATA_FEATURE_FIELDS[22].feature_id == "EV_META_23"


def test_independent_authoritative_100_column_contract():
    """
    Requirement 4: Semantic Contract Test against an independent authoritative ground-truth list.
    Verifies all 100 columns in BASELINE_V8_0_ALL_FEATURE_FIELDS match the independent list exactly
    by column index, ID, and canonical name, ensuring zero schema drift across all 8 modules.
    """
    from stock_rl.config.schema import (
        BASELINE_V8_0_ALL_FEATURE_FIELDS,
        ALL_FEATURE_FIELDS,
        get_feature_index,
    )

    INDEPENDENT_100_SPEC = [
        # Price (12)
        ("P01", "return_1d"),
        ("P02", "return_5d"),
        ("P03", "return_20d"),
        ("P04", "overnight_gap"),
        ("P05", "open_close_return"),
        ("P06", "high_low_range"),
        ("P07", "close_location"),
        ("P08", "log_volume"),
        ("P09", "volume_ratio_20d"),
        ("P10", "turnover_ratio"),
        ("P11", "realized_volatility_20d"),
        ("P12", "log_trading_value"),
        # Technical (16)
        ("T01", "rsi_14"),
        ("T02", "rsi_7"),
        ("T03", "kdj_k"),
        ("T04", "kdj_d"),
        ("T05", "kdj_j"),
        ("T06", "macd_dif"),
        ("T07", "macd_signal"),
        ("T08", "macd_hist"),
        ("T09", "close_ema20_ratio"),
        ("T10", "close_ema60_ratio"),
        ("T11", "ema20_ema60_ratio"),
        ("T12", "atr14_close_ratio"),
        ("T13", "bollinger_percent_b"),
        ("T14", "bollinger_bandwidth"),
        ("T15", "normalized_obv_change_5d"),
        ("T16", "adx_14"),
        # Institutional (8)
        ("I01", "foreign_net_buy_over_vol20d"),
        ("I02", "trust_net_buy_over_vol20d"),
        ("I03", "dealer_net_buy_over_vol20d"),
        ("I04", "foreign_net_buy_5d"),
        ("I05", "total_institutional_net_buy_5d"),
        ("I06", "margin_balance_change"),
        ("I07", "short_balance_change"),
        ("I08", "short_margin_balance_ratio"),
        # Macro (12)
        ("M01", "taiex_return_1d"),
        ("M02", "taiex_return_5d"),
        ("M03", "taiex_volatility_20d"),
        ("M04", "sector_return_1d"),
        ("M05", "sector_relative_strength_20d"),
        ("M06", "usdtwd_return"),
        ("M07", "sp500_return"),
        ("M08", "nasdaq_return"),
        ("M09", "sox_return"),
        ("M10", "vix_level"),
        ("M11", "us_10y_yield"),
        ("M12", "us_10y_yield_change"),
        # Stock Context (16)
        ("C01", "industry_embedding_01"),
        ("C02", "industry_embedding_02"),
        ("C03", "industry_embedding_03"),
        ("C04", "industry_embedding_04"),
        ("C05", "industry_embedding_05"),
        ("C06", "industry_embedding_06"),
        ("C07", "industry_embedding_07"),
        ("C08", "industry_embedding_08"),
        ("C09", "log_market_capitalization"),
        ("C10", "log_avg_trading_value_60d"),
        ("C11", "volatility_percentile"),
        ("C12", "liquidity_percentile"),
        ("C13", "beta_120d"),
        ("C14", "market_cap_percentile_within_industry"),
        ("C15", "listing_age"),
        ("C16", "recent_trading_suspension_indicator"),
        # Event Metadata (24)
        ("E01", "actor_role_embedding_01"),
        ("E02", "actor_role_embedding_02"),
        ("E03", "actor_role_embedding_03"),
        ("E04", "actor_role_embedding_04"),
        ("E05", "event_type_embedding_01"),
        ("E06", "event_type_embedding_02"),
        ("E07", "event_type_embedding_03"),
        ("E08", "event_type_embedding_04"),
        ("E09", "event_type_embedding_05"),
        ("E10", "event_type_embedding_06"),
        ("E11", "event_type_embedding_07"),
        ("E12", "event_type_embedding_08"),
        ("E13", "event_status_embedding_01"),
        ("E14", "event_status_embedding_02"),
        ("E15", "event_status_embedding_03"),
        ("E16", "event_status_embedding_04"),
        ("E17", "evidence_traceability"),
        ("E18", "source_reliability_estimate"),
        ("E19", "source_independence"),
        ("E20", "novelty_input_estimate"),
        ("E21", "market_surprise"),
        ("E22", "quantified_magnitude"),
        ("E23", "event_age"),
        ("E24", "timestamp_reliability"),
        # Event-Stock Relation (8)
        ("R01", "direct_company_mention"),
        ("R02", "industry_match"),
        ("R03", "upstream_exposure"),
        ("R04", "downstream_exposure"),
        ("R05", "customer_relationship"),
        ("R06", "supplier_relationship"),
        ("R07", "geographic_exposure"),
        ("R08", "macro_factor_exposure"),
        # Position State (4)
        ("S01", "current_target_position"),
        ("S02", "unrealized_pnl_pct"),
        ("S03", "holding_duration_days"),
        ("S04", "time_since_last_successful_trade"),
    ]

    assert len(INDEPENDENT_100_SPEC) == 100
    assert len(BASELINE_V8_0_ALL_FEATURE_FIELDS) == 100
    assert len(ALL_FEATURE_FIELDS) == 100

    # 1. Strict item-by-item verification against independent ground truth
    for i, (expected_code, expected_name) in enumerate(INDEPENDENT_100_SPEC):
        actual_field = BASELINE_V8_0_ALL_FEATURE_FIELDS[i]
        assert actual_field.name == expected_name, (
            f"Index {i} ({expected_code}) mismatch: expected {expected_name}, got {actual_field.name}"
        )
        assert actual_field.version == "v8.0-original"
        # Shorthand resolution verification
        resolved_idx = get_feature_index(expected_code)
        # S04 is index 3 in its table, P01 is 0, etc.
        expected_tensor_idx = int(expected_code[1:]) - 1
        assert resolved_idx == expected_tensor_idx

    # 2. Verify Candidate Schema does NOT share corrupted/misaligned definitions
    # Candidate P01 is open_norm (not return_1d)
    assert ALL_FEATURE_FIELDS[0].name == "open_norm"
    # Candidate T14 is cci_14 (not bollinger_bandwidth)
    assert ALL_FEATURE_FIELDS[12 + 13].name == "cci_14"
    # Candidate Event Age is index 12 (EV_META_13)
    assert ALL_FEATURE_FIELDS[12 + 16 + 8 + 12 + 16 + 12].feature_id == "EV_META_13"






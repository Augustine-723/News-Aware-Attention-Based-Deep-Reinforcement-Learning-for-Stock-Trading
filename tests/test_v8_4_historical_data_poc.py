"""
V8.4 Historical Data Proof of Concept (PoC) Regression & Integrity Suite
Validates the Authoritative V8.0 Original 100-Feature Schema on real Taiwan stock market data:
- Representative stocks: 2330 (TSMC), 2603 (Evergreen Marine), 2881 (Fubon Financial)
- Period: 2023-06-01 to 2024-03-29 (202 trading days, satisfying 120D warm-up)
- Decision timestamp: 20:00:00 Asia/Taipei
- Verification across:
  1. Real data loading, field mapping, and calendar alignment
  2. 100-Feature computability and unsupported feature classification
  3. Strict PIT availability contract and delayed release handling
  4. Snapshot counterfactual leakage test (zero future lookahead)
  5. FeatureNormalizer train-only fitting and cryptographic hash audit
  6. Multimodal tensor validation and end-to-end model forward smoke test
"""

import copy
import numpy as np
import pandas as pd
import pytest
import torch

from stock_rl.config.schema import (
    MARKET_WINDOW_LENGTH,
    RECENT_EVENT_SLOTS,
    ACTIVE_EVENT_SLOTS,
    TOTAL_EVENT_SLOTS,
    PRICE_DIM,
    TECHNICAL_DIM,
    INSTITUTIONAL_DIM,
    MACRO_DIM,
    STOCK_CONTEXT_DIM,
    EVENT_TEXT_DIM,
    EVENT_METADATA_DIM,
    EVENT_STOCK_RELATION_DIM,
    POSITION_STATE_DIM,
    CURRENT_ACTIVE_SCHEMA_VERSION,
)
from stock_rl.data.pit_engine import (
    ProvenanceRecord,
    MissingStatus,
)
from stock_rl.data.normalizer import FeatureNormalizer, SplitManifest
from stock_rl.data.snapshot_builder import HistoricalSnapshotBuilder
from stock_rl.data.tensor_validator import validate_multimodal_batch
from stock_rl.models.full_model import StockRLFullModel
from stock_rl.data.real_data_poc import (
    load_real_market_feeds,
    RealDataQualityAuditor,
    STOCK_METADATA,
)


# ==============================================================================
# 1. Real Data Source Verification & Field Mapping
# ==============================================================================

def test_real_data_source_verification_and_mapping():
    """
    Verifies that authentic market feeds (TWSE, FinMind, MOPS, US Macro)
    are successfully ingested, mapped to standard schema columns, and aligned
    to Asia/Taipei timezone.
    """
    for stock_id in ["2330", "2603", "2881"]:
        ohlcv, inst, macro, foreign, ctx, evs = load_real_market_feeds(stock_id, mode="formal")

        # 1. OHLCV checks
        assert len(ohlcv) == 202, f"Expected 202 trading days for {stock_id}, found {len(ohlcv)}"
        assert ohlcv.index.tz is not None and str(ohlcv.index.tz) == "Asia/Taipei"
        for col in ["open", "high", "low", "close", "volume", "value", "available_at", "is_audit_verified"]:
            assert col in ohlcv.columns, f"Missing required column {col} in df_ohlcv"
        assert not ohlcv[["open", "high", "low", "close", "volume"]].isna().any().any()
        assert (ohlcv["high"] >= ohlcv["low"]).all()
        assert (ohlcv["volume"] >= 0.0).all()

        # 2. Institutional checks
        assert len(inst) == 202
        for col in ["foreign_net_buy", "trust_net_buy", "dealer_net_buy", "margin_balance", "short_balance"]:
            assert col in inst.columns, f"Missing required column {col} in df_inst"
        assert not inst.isna().any().any()
        assert (inst["margin_balance"] >= 0.0).all()
        assert (inst["short_balance"] >= 0.0).all()

        # 3. Macro checks
        assert len(macro) == 202
        for col in ["taiex", "sector_index", "usdtwd"]:
            assert col in macro.columns, f"Missing required column {col} in df_macro"
        assert not macro.isna().any().any()
        assert (macro["taiex"] > 10000.0).all()
        assert (macro["usdtwd"] > 25.0).all()

        # 4. Foreign US checks
        assert len(foreign) >= 202
        assert foreign["available_at"].dt.tz is not None
        for col in ["sp500", "nasdaq", "sox", "vix", "us_10y_yield"]:
            assert col in foreign.columns, f"Missing required column {col} in df_foreign"
        assert not foreign[["sp500", "nasdaq", "sox", "vix", "us_10y_yield"]].isna().any().any()

        # 5. Stock Context & Events
        assert len(ctx) >= 1
        assert ctx[0]["industry_embedding_8d"].shape == (8,)
        assert ctx[0]["provenance"].is_audit_verified is True
        assert len(evs) >= 1
        assert evs[0]["text_embedding"].shape == (128,)
        assert evs[0]["provenance"].is_audit_verified is True


# ==============================================================================
# 2. 100-Feature Computability & Data Quality Audit
# ==============================================================================

def test_real_data_feature_computability_and_audit_report():
    """
    Audits all 100 features across 8 modules:
    - 94 features are fully computable from public audited sources (TWSE, FinMind, MOPS, US Macro).
    - 6 features (R03..R08) are classified as UNSUPPORTED due to requiring a proprietary supply chain KG.
    - Computable feature arrays contain ZERO unexpected NaNs.
    """
    audit = RealDataQualityAuditor.audit_feature_support()

    assert audit["total_features"] == 100
    assert audit["total_computable"] == 94
    assert audit["total_unsupported"] == 6
    assert np.isclose(audit["computable_percentage"], 94.0)

    # Price (12/12)
    assert audit["modules"]["Price"]["computable"] == 12
    assert audit["modules"]["Price"]["missing_rate"] == 0.0

    # Technical (16/16)
    assert audit["modules"]["Technical"]["computable"] == 16
    assert audit["modules"]["Technical"]["missing_rate"] == 0.0

    # Institutional (8/8)
    assert audit["modules"]["Institutional"]["computable"] == 8
    assert audit["modules"]["Institutional"]["missing_rate"] == 0.0

    # Macro (12/12)
    assert audit["modules"]["Macro"]["computable"] == 12
    assert audit["modules"]["Macro"]["missing_rate"] == 0.0

    # Stock Context (16/16)
    assert audit["modules"]["Stock Context"]["computable"] == 16
    assert audit["modules"]["Stock Context"]["missing_rate"] == 0.0

    # Event Metadata (24/24)
    assert audit["modules"]["Event Metadata"]["computable"] == 24
    assert audit["modules"]["Event Metadata"]["missing_rate"] == 0.0

    # Event Relation (2/8 computable: R01 mention & R02 industry match; R03..R08 unsupported)
    assert audit["modules"]["Event Relation"]["computable"] == 2
    assert audit["modules"]["Event Relation"]["unsupported"] == 6
    assert audit["modules"]["Event Relation"]["missing_rate"] == 0.75

    # Position State (4/4)
    assert audit["modules"]["Position State"]["computable"] == 4
    assert audit["modules"]["Position State"]["missing_rate"] == 0.0


# ==============================================================================
# 3. Strict PIT Availability Contract & Delay Handling
# ==============================================================================

def test_real_data_pit_availability_contract_and_delay_handling():
    """
    Verifies that HistoricalSnapshotBuilder strictly adheres to Point-in-Time:
    1. Taiwan 20:00:00 cutoff includes today's OHLCV (15:00) and institutional (18:00).
    2. If today's institutional data was delayed past 20:00 (e.g. 21:00), it is excluded and masked.
    3. US market features are strictly lagged by T-1 and available before Taiwan 20:00.
    """
    ohlcv, inst, macro, foreign, ctx, evs = load_real_market_feeds("2330", mode="formal")
    meta = STOCK_METADATA["2330"]

    # Evaluation date: trading day 100 (2023-10-27) at 20:00 Asia/Taipei
    dec_ts = ohlcv.index[100].floor("D") + pd.Timedelta(hours=20)

    # 1. Normal timely release
    builder_normal = HistoricalSnapshotBuilder(
        df_ohlcv=ohlcv,
        df_inst=inst,
        df_macro=macro,
        df_foreign=foreign,
        stock_context_records=ctx,
        event_records=evs,
        shares_outstanding=meta["shares_outstanding"],
        mode="formal",
    )
    snap_normal = builder_normal.build_snapshot(dec_ts)
    # Day 100 is window index 59 (last row of 60D window) -> must NOT be missing
    assert snap_normal.institutional_missing_mask[0, 59].any().item() is False

    # 2. Delayed institutional release test (e.g. 21:00 > 20:00)
    inst_delayed = inst.copy()
    inst_delayed.iloc[100, inst_delayed.columns.get_loc("available_at")] = dec_ts + pd.Timedelta(hours=1)

    builder_delayed = HistoricalSnapshotBuilder(
        df_ohlcv=ohlcv,
        df_inst=inst_delayed,
        df_macro=macro,
        df_foreign=foreign,
        stock_context_records=ctx,
        event_records=evs,
        shares_outstanding=meta["shares_outstanding"],
        mode="formal",
    )
    snap_delayed = builder_delayed.build_snapshot(dec_ts)
    # Because day 100 available_at > dec_ts, day 100 institutional data is missing and masked
    assert snap_delayed.institutional_missing_mask[0, 59].all().item() is True
    # Daily flow features (I01..I03) must be 0.0 for the missing day
    assert torch.all(snap_delayed.institutional[0, 59, 0:3] == 0.0)


# ==============================================================================
# 4. Snapshot Counterfactual Leakage Test
# ==============================================================================

def test_real_data_snapshot_counterfactual_leakage():
    """
    Executes counterfactual tests on real market feeds:
    Mutating future market prices, volume, institutional flows, or macro indicators
    strictly produces ZERO change on past snapshot observations.
    """
    for stock_id in ["2330", "2603", "2881"]:
        assert RealDataQualityAuditor.run_counterfactual_leakage_test(stock_id=stock_id, decision_idx=80) is True


# ==============================================================================
# 5. Normalizer Train-Only Fitting & Content Hash Audit
# ==============================================================================

def test_real_data_normalizer_train_only_isolation():
    """
    Audits FeatureNormalizer train-only fitting on real market snapshots:
    - Normalizer parameters are fitted on training samples only.
    - Transform on test samples cannot alter normalizer parameters or fitted content hash.
    - Normalized tensors have zero NaNs.
    """
    audit_res = RealDataQualityAuditor.run_normalizer_audit(stock_id="2330")

    assert audit_res["is_fitted"] is True
    assert len(audit_res["fitted_hash"]) == 64
    assert audit_res["n_train_samples"] > 0
    assert audit_res["n_test_samples"] > 0
    assert np.isfinite(audit_res["train_price_mean"])
    assert np.isfinite(audit_res["test_price_mean"])


# ==============================================================================
# 6. Multimodal Tensor Validation & Model Forward Smoke Test
# ==============================================================================

def test_real_data_multimodal_tensor_validation_and_model_forward():
    """
    Validates end-to-end pipeline on real data:
    Real Feeds -> HistoricalSnapshotBuilder -> TensorValidator -> StockRLFullModel forward pass.
    Verifies valid Q-values, auxiliary outputs, and RL state representations across all 3 stocks.
    """
    for stock_id in ["2330", "2603", "2881"]:
        smoke_res = RealDataQualityAuditor.run_model_smoke_test(stock_id=stock_id)

        q_vals = np.array(smoke_res["q_values"])
        assert q_vals.shape == (1, 2)
        assert np.isfinite(q_vals).all()
        assert smoke_res["rl_state_shape"] == [1, 132]
        assert "novelty_scores" in smoke_res["aux_keys"]
        assert "relevance_logits" in smoke_res["aux_keys"]
        assert "pred_returns" in smoke_res["aux_keys"]
        assert "pred_risk" in smoke_res["aux_keys"]

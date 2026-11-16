"""
Regression tests for V8.1 Code Review 3:
1. Novelty Causal Memory:
   - Counterfactual invariance: Modifying future events produces ZERO change on earlier events' novelty prediction.
   - Fallback strictness: When an event has no prior visible events, fallback uses NRE / No-Prior-Memory token,
     not active_projected_memory.
2. Feature Dictionary full alignment:
   - Stable feature_ids (PRC_01..12, TECH_01..16, INST_01..08, MACRO_01..12, CTX_01..16, EV_META_01..24, EV_REL_01..08, POS_01..04).
   - Strict dimension check (12, 16, 8, 12, 16, 24, 8, 4) total 100 features.
   - All fields populated with math_definition, unit, source, available_at, missing_policy, version.
3. Position State:
   - 4th dimension restored to time_since_last_trade.
   - Successful trade (buy or sell) resets duration to 0.
   - Rejected trades (limit up/down) or hold states increment duration.
"""

import pytest
import torch
import numpy as np

from stock_rl.config.schema import (
    PRICE_FEATURE_FIELDS,
    TECHNICAL_FEATURE_FIELDS,
    INSTITUTIONAL_FEATURE_FIELDS,
    MACRO_FEATURE_FIELDS,
    STOCK_CONTEXT_FEATURE_FIELDS,
    EVENT_METADATA_FEATURE_FIELDS,
    EVENT_STOCK_RELATION_FEATURE_FIELDS,
    POSITION_STATE_FEATURE_FIELDS,
    ALL_FEATURE_FIELDS,
    PRICE_FEATURE_IDS,
    TECHNICAL_FEATURE_IDS,
    INSTITUTIONAL_FEATURE_IDS,
    MACRO_FEATURE_IDS,
    STOCK_CONTEXT_FEATURE_IDS,
    EVENT_METADATA_FEATURE_IDS,
    EVENT_STOCK_RELATION_FEATURE_IDS,
    POSITION_STATE_FEATURE_IDS,
    TOTAL_EVENT_SLOTS,
    EVENT_AGE_METADATA_INDEX,
)
from stock_rl.config.settings import TradingConfig
from stock_rl.data.synthetic_data import MultimodalBatch, generate_synthetic_batch
from stock_rl.models.full_model import StockRLFullModel
from stock_rl.env.stock_trading_env import PointInTimeStockEnv, ACTION_LONG, ACTION_CASH
from stock_rl.env.execution import ExecutionResult


def test_feature_dictionary_v8_alignment():
    """Verify all 8 feature groups strictly conform to V8 specifications."""
    expected_counts = {
        "Price": (PRICE_FEATURE_FIELDS, PRICE_FEATURE_IDS, 12, "PRC"),
        "Technical": (TECHNICAL_FEATURE_FIELDS, TECHNICAL_FEATURE_IDS, 16, "TECH"),
        "Institutional": (INSTITUTIONAL_FEATURE_FIELDS, INSTITUTIONAL_FEATURE_IDS, 8, "INST"),
        "Macro": (MACRO_FEATURE_FIELDS, MACRO_FEATURE_IDS, 12, "MACRO"),
        "Stock Context": (STOCK_CONTEXT_FEATURE_FIELDS, STOCK_CONTEXT_FEATURE_IDS, 16, "CTX"),
        "Event Metadata": (EVENT_METADATA_FEATURE_FIELDS, EVENT_METADATA_FEATURE_IDS, 24, "EV_META"),
        "Event Relation": (EVENT_STOCK_RELATION_FEATURE_FIELDS, EVENT_STOCK_RELATION_FEATURE_IDS, 8, "EV_REL"),
        "Position State": (POSITION_STATE_FEATURE_FIELDS, POSITION_STATE_FEATURE_IDS, 4, "POS"),
    }

    total_count = 0
    for name, (fields, ids, count, prefix) in expected_counts.items():
        assert len(fields) == count, f"{name} field count mismatch: expected {count}, got {len(fields)}"
        assert len(ids) == count, f"{name} id count mismatch: expected {count}, got {len(ids)}"
        total_count += count

        for idx, field in enumerate(fields):
            expected_id = f"{prefix}_{idx+1:02d}"
            assert field.feature_id == expected_id, f"Field ID mismatch: expected {expected_id}, got {field.feature_id}"
            assert field.math_definition, f"Field {field.feature_id} missing math_definition"
            assert field.unit, f"Field {field.feature_id} missing unit"
            assert field.source, f"Field {field.feature_id} missing source"
            assert field.available_timing, f"Field {field.feature_id} missing available_timing"
            assert field.missing_policy, f"Field {field.feature_id} missing missing_policy"
            assert field.version == "V8.0", f"Field {field.feature_id} version mismatch"

    assert len(ALL_FEATURE_FIELDS) == total_count == 100, f"Total feature fields must be 100, got {len(ALL_FEATURE_FIELDS)}"
    # Check POS_04 is specifically time_since_last_trade
    assert POSITION_STATE_FEATURE_FIELDS[3].feature_id == "POS_04"
    assert POSITION_STATE_FEATURE_FIELDS[3].name == "time_since_last_trade"


def test_novelty_counterfactual_future_invariance():
    """
    Test counterfactual invariance:
    Modifying future events MUST NOT alter novelty prediction for earlier events.
    """
    torch.manual_seed(42)
    model = StockRLFullModel()
    model.eval()

    batch = generate_synthetic_batch(batch_size=1)

    # Construct explicit temporal order in event_metadata index 4 (available_days_ago)
    # Slot 0: earliest event (e.g. 20 days ago, available_days_ago = 20)
    # Slot 1: intermediate event (e.g. 10 days ago, available_days_ago = 10)
    # Slot 2: newest/future event relative to slot 0 and 1 (e.g. 1 day ago, available_days_ago = 1)
    # Note: In schema, event_metadata[:, :, 4] is days_since_publication or age.
    # A larger age means it happened EARLIER in history.
    # Therefore:
    # Slot 0: age = 20 (earliest event in history)
    # Slot 1: age = 10 (later event than slot 0, but earlier than slot 2)
    # Slot 2: age = 1  (latest event in history)
    with torch.no_grad():
        batch.event_mask[:, :] = False
        batch.event_mask[:, 0] = True
        batch.event_mask[:, 1] = True
        batch.event_mask[:, 2] = True

        batch.event_metadata[:, :, EVENT_AGE_METADATA_INDEX] = 0.0
        batch.event_metadata[:, 0, EVENT_AGE_METADATA_INDEX] = 20.0  # earliest
        batch.event_metadata[:, 1, EVENT_AGE_METADATA_INDEX] = 10.0  # intermediate
        batch.event_metadata[:, 2, EVENT_AGE_METADATA_INDEX] = 1.0   # latest

        # Forward pass 1
        _, preds1, _ = model(batch)
        novelty1 = preds1["novelty_scores"].detach().clone()  # [1, 40, 1]

        # Counterfactual perturbation: Modify the FUTURE event (slot 2) text embedding completely
        batch_counterfactual = MultimodalBatch(
            price=batch.price.clone(),
            technical=batch.technical.clone(),
            institutional=batch.institutional.clone(),
            macro=batch.macro.clone(),
            stock_context=batch.stock_context.clone(),
            recent_event_emb=batch.recent_event_emb.clone(),
            active_event_emb=batch.active_event_emb.clone(),
            event_metadata=batch.event_metadata.clone(),
            event_stock_rel=batch.event_stock_rel.clone(),
            position=batch.position.clone(),
            event_mask=batch.event_mask.clone(),
        )
        # Perturb slot 2 (future event)
        batch_counterfactual.recent_event_emb[:, 2, :] += 50.0

        _, preds2, _ = model(batch_counterfactual)
        novelty2 = preds2["novelty_scores"].detach().clone()

        # Slot 0 has NO prior events (age 20 > nothing). Its prior memory is NRE token.
        # It must be identical between out1 and out2.
        assert torch.allclose(novelty1[:, 0, :], novelty2[:, 0, :], atol=1e-6), (
            f"Earliest event (slot 0) novelty changed when future event was modified! "
            f"Diff: {(novelty1[:, 0, :] - novelty2[:, 0, :]).abs().max().item()}"
        )

        # Slot 1 has prior event slot 0 (age 20 > 10). It does NOT see slot 2 (age 1 < 10).
        # Modifying slot 2 MUST NOT affect slot 1 novelty!
        assert torch.allclose(novelty1[:, 1, :], novelty2[:, 1, :], atol=1e-6), (
            f"Intermediate event (slot 1) novelty changed when future event (slot 2) was modified! "
            f"Diff: {(novelty1[:, 1, :] - novelty2[:, 1, :]).abs().max().item()}"
        )

        # But slot 2's own novelty or representation SHOULD change because its own features changed
        slot2_diff = (novelty1[:, 2, :] - novelty2[:, 2, :]).abs().max().item()
        assert slot2_diff > 1e-4, f"Slot 2 novelty should have changed when its features changed, diff was {slot2_diff}"


def test_novelty_no_prior_memory_fallback():
    """
    Test that an event with no prior visible events strictly falls back to NRE token
    rather than active_projected_memory.
    """
    torch.manual_seed(42)
    model = StockRLFullModel()
    model.eval()

    batch = generate_synthetic_batch(batch_size=1)
    with torch.no_grad():
        # Mask out all events except slot 0
        batch.event_mask[:, :] = False
        batch.event_mask[:, 0] = True
        batch.event_metadata[:, 0, EVENT_AGE_METADATA_INDEX] = 5.0

        _, _, projected_events, _, causal_prior_memory = model.encode_multimodal_state(batch)

        expected_fallback = model.event_module.nre_token.squeeze(1)[0]  # [128]
        actual_prior_0 = causal_prior_memory[0, 0]  # [128]

        assert torch.allclose(actual_prior_0, expected_fallback, atol=1e-6), (
            "When no prior events exist, causal prior memory must strictly equal NRE token"
        )


def test_position_state_time_since_last_trade():
    """
    Verify position state 4th dimension: time_since_last_trade
    - Initialized to 1.0 (60 / 60)
    - Successful execution resets to 0.0
    - Rejected trades increment days_since_last_trade
    - Cash hold and Long hold increment days_since_last_trade
    """
    config = TradingConfig(
        daily_price_limit_pct=0.10,
    )
    env = PointInTimeStockEnv(config=config, initial_capital=1_000_000.0)

    obs, _ = env.reset()
    # Step 0: Initial state
    assert env.days_since_last_trade == 60
    assert pytest.approx(obs.position[0, 3].item(), 1e-4) == 1.0

    # Step 1: Place LONG action, executes successfully (normal prices)
    obs, reward, terminated, truncated, info = env.step(ACTION_LONG)
    assert env.current_position == ACTION_LONG
    assert env.days_since_last_trade == 0
    assert pytest.approx(obs.position[0, 3].item(), 1e-4) == 0.0

    # Step 2: Hold LONG (no trade executed)
    obs, reward, terminated, truncated, info = env.step(ACTION_LONG)
    assert env.current_position == ACTION_LONG
    assert env.days_since_last_trade == 1
    assert pytest.approx(obs.position[0, 3].item(), 1e-4) == min(1 / 60.0, 1.0)

    # Step 3: Simulate rejected SELL trade (e.g. limit-down lock)
    orig_execute = env.simulator.execute_target_position

    def mock_rejected_sell(*args, **kwargs):
        res = ExecutionResult(
            executed_action=ACTION_LONG,  # remains LONG due to rejection
            executed_price=0.0,
            shares_traded=0.0,
            commission_paid=0.0,
            tax_paid=0.0,
            slippage_cost=0.0,
            order_rejected=True,
        )
        return env.cash, env.shares, res

    env.simulator.execute_target_position = mock_rejected_sell
    obs, reward, terminated, truncated, info = env.step(ACTION_CASH)  # sell order rejected
    # Must increment, NOT reset
    assert env.current_position == ACTION_LONG
    assert env.days_since_last_trade == 2
    assert pytest.approx(obs.position[0, 3].item(), 1e-4) == min(2 / 60.0, 1.0)

    # Step 4: Restore normal execution and execute successful SELL
    env.simulator.execute_target_position = orig_execute
    obs, reward, terminated, truncated, info = env.step(ACTION_CASH)  # sell succeeds
    assert env.current_position == ACTION_CASH
    assert env.days_since_last_trade == 0
    assert pytest.approx(obs.position[0, 3].item(), 1e-4) == 0.0

    # Step 5: Hold CASH
    obs, reward, terminated, truncated, info = env.step(ACTION_CASH)
    assert env.current_position == ACTION_CASH
    assert env.days_since_last_trade == 1
    assert pytest.approx(obs.position[0, 3].item(), 1e-4) == min(1 / 60.0, 1.0)

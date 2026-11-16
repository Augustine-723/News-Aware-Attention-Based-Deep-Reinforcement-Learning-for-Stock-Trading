"""
High-Fidelity Synthetic Point-in-Time Data Generator V8.0
Generates verifiable multimodal market, macro, event, and label tensors strictly
matching Specification V8.0 schemas, with 4 independent task masks and 3-level novelty targets.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np
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
    NUM_HORIZONS,
    CURRENT_ACTIVE_SCHEMA_VERSION,
    FeatureSchemaVersion,
    get_event_age_index,
)


@dataclass
class MultimodalBatch:
    """Standard multimodal batch container conforming to V8.0 tensor contracts."""
    price: torch.Tensor             # [B, 60, 12]
    technical: torch.Tensor         # [B, 60, 16]
    institutional: torch.Tensor     # [B, 60, 8]
    macro: torch.Tensor             # [B, 60, 12]
    stock_context: torch.Tensor     # [B, 16]
    recent_event_emb: torch.Tensor  # [B, 32, 128]
    active_event_emb: torch.Tensor  # [B, 8, 128]
    event_metadata: torch.Tensor    # [B, 40, 24]
    event_stock_rel: torch.Tensor   # [B, 40, 8]
    event_mask: torch.Tensor        # [B, 40] (bool)
    position: torch.Tensor          # [B, 4]

    # Supervised Targets
    relevance_labels: Optional[torch.Tensor] = None  # [B, 40, 1]
    novelty_labels: Optional[torch.Tensor] = None    # [B, 40, 1] (Values in {0.0, 0.5, 1.0})
    return_targets: Optional[torch.Tensor] = None    # [B, 3] (1D, 3D, 5D executable open-to-open returns)
    risk_targets: Optional[torch.Tensor] = None      # [B, 3] (1D, 3D, 5D future return magnitude proxy)

    # 4 Independent Task Validity Masks (V8.0 Requirement)
    relevance_mask: Optional[torch.Tensor] = None    # [B, 40, 1] (bool)
    novelty_mask: Optional[torch.Tensor] = None      # [B, 40, 1] (bool)
    return_mask: Optional[torch.Tensor] = None       # [B, 3] (bool)
    risk_mask: Optional[torch.Tensor] = None         # [B, 3] (bool)

    schema_version: str = CURRENT_ACTIVE_SCHEMA_VERSION

    def to(self, device: torch.device) -> "MultimodalBatch":
        """Transfers all non-None tensors to specified device."""
        return MultimodalBatch(
            price=self.price.to(device),
            technical=self.technical.to(device),
            institutional=self.institutional.to(device),
            macro=self.macro.to(device),
            stock_context=self.stock_context.to(device),
            recent_event_emb=self.recent_event_emb.to(device),
            active_event_emb=self.active_event_emb.to(device),
            event_metadata=self.event_metadata.to(device),
            event_stock_rel=self.event_stock_rel.to(device),
            event_mask=self.event_mask.to(device),
            position=self.position.to(device),
            relevance_labels=self.relevance_labels.to(device) if self.relevance_labels is not None else None,
            novelty_labels=self.novelty_labels.to(device) if self.novelty_labels is not None else None,
            return_targets=self.return_targets.to(device) if self.return_targets is not None else None,
            risk_targets=self.risk_targets.to(device) if self.risk_targets is not None else None,
            relevance_mask=self.relevance_mask.to(device) if self.relevance_mask is not None else None,
            novelty_mask=self.novelty_mask.to(device) if self.novelty_mask is not None else None,
            return_mask=self.return_mask.to(device) if self.return_mask is not None else None,
            risk_mask=self.risk_mask.to(device) if self.risk_mask is not None else None,
            schema_version=self.schema_version,
        )


def generate_synthetic_batch(
    batch_size: int = 64,
    seed: int = 42,
    include_targets: bool = True,
    schema_version: Optional[str] = None,
) -> MultimodalBatch:
    """
    Generates a synthetic multimodal batch strictly complying with V8.0 dimensions and value distributions.
    Supports both 'v8.0-original' (Baseline) and 'v8.1-candidate' (Ablation).
    """
    target_schema = schema_version if schema_version is not None else CURRENT_ACTIVE_SCHEMA_VERSION
    rng = np.random.default_rng(seed)

    if target_schema in (FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value, "v8.0-original", "v8.0"):
        # Original V8 Baseline 100-Feature Layout:
        # Price 12D:
        # P01..P03: returns in [-0.10, 0.10]
        # P04: overnight gap in [-0.05, 0.05]
        # P05: open-close return in [-0.08, 0.08]
        # P06: high-low range in [0.001, 0.15]
        # P07: close_location in [0.0, 1.0]
        # P08: log_volume >= 0
        # P09: volume_ratio_20d >= 0
        # P10: turnover_ratio in [0.001, 0.20]
        # P11: realized_volatility_20d >= 0
        # P12: log_trading_value >= 0
        p_rets = rng.uniform(-0.08, 0.08, size=(batch_size, MARKET_WINDOW_LENGTH, 3))
        p_gap = rng.uniform(-0.03, 0.03, size=(batch_size, MARKET_WINDOW_LENGTH, 1))
        p_oc = rng.uniform(-0.05, 0.05, size=(batch_size, MARKET_WINDOW_LENGTH, 1))
        p_hl = rng.uniform(0.005, 0.12, size=(batch_size, MARKET_WINDOW_LENGTH, 1))
        p_cl = rng.uniform(0.0, 1.0, size=(batch_size, MARKET_WINDOW_LENGTH, 1))
        p_vol = rng.uniform(9.0, 16.0, size=(batch_size, MARKET_WINDOW_LENGTH, 1))
        p_vrat = rng.uniform(0.3, 3.0, size=(batch_size, MARKET_WINDOW_LENGTH, 1))
        p_turn = rng.uniform(0.005, 0.10, size=(batch_size, MARKET_WINDOW_LENGTH, 1))
        p_rvol = rng.uniform(0.10, 0.60, size=(batch_size, MARKET_WINDOW_LENGTH, 1))
        p_tval = rng.uniform(13.0, 21.0, size=(batch_size, MARKET_WINDOW_LENGTH, 1))
        raw_price = np.concatenate([p_rets, p_gap, p_oc, p_hl, p_cl, p_vol, p_vrat, p_turn, p_rvol, p_tval], axis=-1)
        price = torch.tensor(raw_price, dtype=torch.float32)

        # Technical 16D:
        # T01..T05: RSI14, RSI7, KDJ K, D, J in [-1.0, 1.0]
        # T06..T08: MACD dif, signal, hist
        # T09..T11: EMA ratios
        # T12: ATR14/Close >= 0
        # T13: Bollinger %B
        # T14: Bollinger Bandwidth >= 0
        # T15: Normalized OBV change 5D
        # T16: ADX14 in [0.0, 1.0]
        t_osc = rng.uniform(-0.8, 0.8, size=(batch_size, MARKET_WINDOW_LENGTH, 5))
        t_macd = rng.uniform(-0.03, 0.03, size=(batch_size, MARKET_WINDOW_LENGTH, 3))
        t_ema = rng.uniform(-0.15, 0.15, size=(batch_size, MARKET_WINDOW_LENGTH, 3))
        t_atr = rng.uniform(0.01, 0.06, size=(batch_size, MARKET_WINDOW_LENGTH, 1))
        t_bpct = rng.uniform(0.0, 1.0, size=(batch_size, MARKET_WINDOW_LENGTH, 1))
        t_bbw = rng.uniform(0.02, 0.35, size=(batch_size, MARKET_WINDOW_LENGTH, 1))
        t_obv = rng.uniform(-1.5, 1.5, size=(batch_size, MARKET_WINDOW_LENGTH, 1))
        t_adx = rng.uniform(0.10, 0.70, size=(batch_size, MARKET_WINDOW_LENGTH, 1))
        raw_tech = np.concatenate([t_osc, t_macd, t_ema, t_atr, t_bpct, t_bbw, t_obv, t_adx], axis=-1)
        technical = torch.tensor(raw_tech, dtype=torch.float32)

        # Institutional 8D:
        # I01..I03: Net Buy / MA20 Volume
        # I04..I05: 5D Net Buy
        # I06..I07: Margin / Short change
        # I08: Short / Margin ratio >= 0
        inst_flow = rng.uniform(-0.4, 0.4, size=(batch_size, MARKET_WINDOW_LENGTH, 5))
        inst_bal = rng.uniform(-0.05, 0.05, size=(batch_size, MARKET_WINDOW_LENGTH, 2))
        inst_ratio = rng.uniform(0.01, 0.30, size=(batch_size, MARKET_WINDOW_LENGTH, 1))
        raw_inst = np.concatenate([inst_flow, inst_bal, inst_ratio], axis=-1)
        institutional = torch.tensor(raw_inst, dtype=torch.float32)

        # Macro 12D:
        # M01..M06: TAIEX 1D/5D, Vol20, Sector 1D, RelStrength 20D, USDTWD
        # M07..M12: US T-1 cross-market (S&P, NASDAQ, SOX, VIX, US 10Y, 10Y change)
        m_tw = rng.uniform(-0.04, 0.04, size=(batch_size, MARKET_WINDOW_LENGTH, 6))
        m_tw[:, :, 2] = np.abs(m_tw[:, :, 2]) * 5.0 + 0.10  # Vol >= 0
        m_us = rng.uniform(-0.03, 0.03, size=(batch_size, MARKET_WINDOW_LENGTH, 6))
        m_us[:, :, 3] = rng.uniform(0.12, 0.45, size=(batch_size, MARKET_WINDOW_LENGTH))  # VIX level / 100
        m_us[:, :, 4] = rng.uniform(0.02, 0.05, size=(batch_size, MARKET_WINDOW_LENGTH))  # US 10Y yield / 100
        raw_macro = np.concatenate([m_tw, m_us], axis=-1)
        macro = torch.tensor(raw_macro, dtype=torch.float32)

        # Stock Context 16D:
        # C01..C08: 8D Industry embedding (dense unit vector)
        ind_raw = rng.standard_normal((batch_size, 8))
        ind_norm = ind_raw / (np.linalg.norm(ind_raw, axis=-1, keepdims=True) + 1e-7)
        # C09..C16: Log Cap, Log Val, Vol Pct, Liq Pct, Beta, Ind Cap Pct, Age, Suspension
        c_cap = rng.uniform(18.0, 26.0, size=(batch_size, 1))
        c_val = rng.uniform(14.0, 20.0, size=(batch_size, 1))
        c_vpct = rng.uniform(0.0, 1.0, size=(batch_size, 1))
        c_lpct = rng.uniform(0.0, 1.0, size=(batch_size, 1))
        c_beta = rng.uniform(0.6, 1.6, size=(batch_size, 1))
        c_icap = rng.uniform(0.0, 1.0, size=(batch_size, 1))
        c_age = rng.uniform(0.1, 1.0, size=(batch_size, 1))
        c_susp = rng.choice([0.0, 1.0], size=(batch_size, 1), p=[0.95, 0.05])
        raw_ctx = np.concatenate([ind_norm, c_cap, c_val, c_vpct, c_lpct, c_beta, c_icap, c_age, c_susp], axis=-1)
        stock_context = torch.tensor(raw_ctx, dtype=torch.float32)

        # Event Metadata 24D:
        # E01..E04: 4D Actor Role Embedding
        # E05..E12: 8D Event Type Embedding
        # E13..E16: 4D Event Status Embedding
        # E17..E20: Traceability, Reliability, Independence, Novelty Input in [0, 1]
        # E21: Market Surprise pre-event implied move in [-0.15, 0.15]
        # E22: Quantified magnitude in [0, 1]
        # E23: Event Age in [0.1, 60.0] (Index 22)
        # E24: Timestamp Reliability in [0, 1] (Index 23)
        ev_meta_arr = np.zeros((batch_size, TOTAL_EVENT_SLOTS, EVENT_METADATA_DIM), dtype=np.float32)
        # Actor 4D
        ev_meta_arr[:, :, 0:4] = rng.uniform(-1.0, 1.0, size=(batch_size, TOTAL_EVENT_SLOTS, 4))
        # Type 8D
        ev_meta_arr[:, :, 4:12] = rng.uniform(-1.0, 1.0, size=(batch_size, TOTAL_EVENT_SLOTS, 8))
        # Status 4D
        ev_meta_arr[:, :, 12:16] = rng.uniform(-1.0, 1.0, size=(batch_size, TOTAL_EVENT_SLOTS, 4))
        # Scalars E17..E20
        ev_meta_arr[:, :, 16:20] = rng.uniform(0.0, 1.0, size=(batch_size, TOTAL_EVENT_SLOTS, 4))
        # E21: Market Surprise (strictly pre-event)
        ev_meta_arr[:, :, 20] = rng.uniform(-0.15, 0.15, size=(batch_size, TOTAL_EVENT_SLOTS))
        # E22: Quantified magnitude
        ev_meta_arr[:, :, 21] = rng.uniform(0.0, 1.0, size=(batch_size, TOTAL_EVENT_SLOTS))
        # E23: Event Age (Index 22) strictly >= 0.1
        ev_meta_arr[:, :, 22] = rng.uniform(0.1, 60.0, size=(batch_size, TOTAL_EVENT_SLOTS))
        # E24: Timestamp Reliability (Index 23)
        ev_meta_arr[:, :, 23] = rng.uniform(0.5, 1.0, size=(batch_size, TOTAL_EVENT_SLOTS))
        event_metadata = torch.tensor(ev_meta_arr, dtype=torch.float32)

        # Event-Stock Relation 8D:
        # R01..R02 binary; R03..R08 continuous weights in [0, 1]
        r_bin = rng.choice([0.0, 1.0], size=(batch_size, TOTAL_EVENT_SLOTS, 2), p=[0.7, 0.3])
        r_cont = rng.uniform(0.0, 1.0, size=(batch_size, TOTAL_EVENT_SLOTS, 6))
        event_stock_rel = torch.tensor(np.concatenate([r_bin, r_cont], axis=-1), dtype=torch.float32)

    else:
        # Candidate V8.1 layout (normalized tensors)
        price = torch.tensor(rng.standard_normal((batch_size, MARKET_WINDOW_LENGTH, PRICE_DIM)), dtype=torch.float32)
        technical = torch.tensor(rng.standard_normal((batch_size, MARKET_WINDOW_LENGTH, TECHNICAL_DIM)), dtype=torch.float32)
        institutional = torch.tensor(rng.standard_normal((batch_size, MARKET_WINDOW_LENGTH, INSTITUTIONAL_DIM)), dtype=torch.float32)
        macro = torch.tensor(rng.standard_normal((batch_size, MARKET_WINDOW_LENGTH, MACRO_DIM)), dtype=torch.float32)
        stock_context = torch.tensor(rng.standard_normal((batch_size, STOCK_CONTEXT_DIM)), dtype=torch.float32)

        event_metadata = torch.tensor(rng.uniform(0.0, 1.0, (batch_size, TOTAL_EVENT_SLOTS, EVENT_METADATA_DIM)), dtype=torch.float32)
        # Event age in fractional days (EV_META_13 / index 12 in candidate)
        event_metadata[:, :, 12] = torch.tensor(rng.uniform(0.1, 60.0, (batch_size, TOTAL_EVENT_SLOTS)), dtype=torch.float32)
        event_stock_rel = torch.tensor(rng.uniform(0.0, 1.0, (batch_size, TOTAL_EVENT_SLOTS, EVENT_STOCK_RELATION_DIM)), dtype=torch.float32)

    # Event embeddings: unit-norm text vectors
    raw_recent = rng.standard_normal((batch_size, RECENT_EVENT_SLOTS, EVENT_TEXT_DIM))
    raw_active = rng.standard_normal((batch_size, ACTIVE_EVENT_SLOTS, EVENT_TEXT_DIM))
    recent_emb = torch.tensor(raw_recent / (np.linalg.norm(raw_recent, axis=-1, keepdims=True) + 1e-8), dtype=torch.float32)
    active_emb = torch.tensor(raw_active / (np.linalg.norm(raw_active, axis=-1, keepdims=True) + 1e-8), dtype=torch.float32)

    # Event mask: simulate 5 to 30 active events per sample
    event_mask = torch.zeros((batch_size, TOTAL_EVENT_SLOTS), dtype=torch.bool)
    for b in range(batch_size):
        num_valid = rng.integers(5, TOTAL_EVENT_SLOTS + 1)
        event_mask[b, :num_valid] = True

    # Position features [target_position, pnl_pct, holding_days_norm, time_since_last_trade_norm]
    pos_actions = rng.choice([0.0, 1.0], size=(batch_size, 1))
    pos_pnl = rng.uniform(-0.15, 0.20, size=(batch_size, 1))
    pos_days = rng.uniform(0.0, 1.0, size=(batch_size, 1))
    pos_last = rng.uniform(0.0, 1.0, size=(batch_size, 1))
    position = torch.tensor(np.hstack([pos_actions, pos_pnl, pos_days, pos_last]), dtype=torch.float32)

    relevance_labels = None
    novelty_labels = None
    return_targets = None
    risk_targets = None
    relevance_mask = None
    novelty_mask = None
    return_mask = None
    risk_mask = None

    if include_targets:
        # Task 1: Relevance (binary 0 or 1) with validity mask
        relevance_raw = (rng.uniform(0.0, 1.0, (batch_size, TOTAL_EVENT_SLOTS, 1)) > 0.4).astype(np.float32)
        relevance_labels = torch.tensor(relevance_raw, dtype=torch.float32)
        relevance_mask = event_mask.unsqueeze(-1) & torch.tensor(rng.uniform(0.0, 1.0, (batch_size, TOTAL_EVENT_SLOTS, 1)) > 0.05, dtype=torch.bool)

        # Task 2: Novelty with V8 ordinal values {0.0: redundant, 0.5: incremental, 1.0: completely novel}
        novelty_choices = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        novelty_raw = rng.choice(novelty_choices, size=(batch_size, TOTAL_EVENT_SLOTS, 1), p=[0.4, 0.35, 0.25])
        novelty_labels = torch.tensor(novelty_raw, dtype=torch.float32)
        novelty_mask = event_mask.unsqueeze(-1) & torch.tensor(rng.uniform(0.0, 1.0, (batch_size, TOTAL_EVENT_SLOTS, 1)) > 0.10, dtype=torch.bool)

        # Task 3: Multi-Horizon Returns calculated from next executable open
        ret_1d = rng.normal(0.0005, 0.015, size=(batch_size, 1))
        ret_3d = rng.normal(0.0015, 0.025, size=(batch_size, 1))
        ret_5d = rng.normal(0.0025, 0.035, size=(batch_size, 1))
        return_targets = torch.tensor(np.hstack([ret_1d, ret_3d, ret_5d]), dtype=torch.float32)
        return_mask = torch.tensor(rng.uniform(0.0, 1.0, (batch_size, NUM_HORIZONS)) > 0.05, dtype=torch.bool)

        # Task 4: Future Risk Targets (realized return magnitude proxy > 0)
        risk_1d = np.abs(ret_1d) + rng.exponential(0.005, size=(batch_size, 1))
        risk_3d = np.abs(ret_3d) + rng.exponential(0.010, size=(batch_size, 1))
        risk_5d = np.abs(ret_5d) + rng.exponential(0.015, size=(batch_size, 1))
        risk_targets = torch.tensor(np.hstack([risk_1d, risk_3d, risk_5d]), dtype=torch.float32)
        risk_mask = return_mask.clone()

    return MultimodalBatch(
        price=price,
        technical=technical,
        institutional=institutional,
        macro=macro,
        stock_context=stock_context,
        recent_event_emb=recent_emb,
        active_event_emb=active_emb,
        event_metadata=event_metadata,
        event_stock_rel=event_stock_rel,
        event_mask=event_mask,
        position=position,
        relevance_labels=relevance_labels,
        novelty_labels=novelty_labels,
        return_targets=return_targets,
        risk_targets=risk_targets,
        relevance_mask=relevance_mask,
        novelty_mask=novelty_mask,
        return_mask=return_mask,
        risk_mask=risk_mask,
        schema_version=target_schema,
    )

"""
Decision-Aligned Historical Snapshot Builder per Specification V8.0 / V8.3 (P0-1)
Constructs strictly As-Of feature snapshots for discrete decision timestamps:
- Requires explicit timezone-aware decision_timestamp (Asia/Taipei 20:00).
- Slices Market (OHLCV, Technicals, Institutional) using strictly available_at <= decision_timestamp.
- Forms strict 60-day market window ending on or before decision_timestamp.
- Slices Macro & Overseas with timezone-aware As-Of join.
- Slices Stock Context using point-in-time dynamic calculations (NO lookahead to dataset end).
- Slices Events available as of decision_timestamp, sorted by available_at, dynamic event age.
- Tracks feature missing status and constructs explicit missing masks (P1-4).
- Counterfactual invariant: modifying future data cannot change past snapshots.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
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
    MissingStatus,
    ProvenanceRecord,
    HistoricalVersionSelector,
    asof_join_cross_market,
)
from stock_rl.data.feature_engine import (
    compute_price_features,
    compute_technical_features,
    compute_institutional_features,
    compute_macro_features,
    assemble_stock_context_features,
    assemble_event_metadata,
    assemble_event_stock_relation,
    build_audited_stock_context,
    build_audited_event_metadata,
)
from stock_rl.data.synthetic_data import MultimodalBatch


class HistoricalSnapshotBuilder:
    """
    P0-1 Unified Historical Snapshot Builder.
    Constructs pointwise As-Of multimodal observations strictly bounded by decision_timestamp.
    """
    def __init__(
        self,
        df_ohlcv: pd.DataFrame,
        df_inst: Optional[pd.DataFrame] = None,
        df_macro: Optional[pd.DataFrame] = None,
        df_foreign: Optional[pd.DataFrame] = None,
        stock_context_records: Optional[List[Dict[str, Any]]] = None,
        event_records: Optional[List[Dict[str, Any]]] = None,
        shares_outstanding: float = 1e8,
        schema_version: str = CURRENT_ACTIVE_SCHEMA_VERSION,
        mode: str = "formal",
    ):
        self.df_ohlcv = df_ohlcv.copy()
        self.df_inst = df_inst.copy() if df_inst is not None else None
        self.df_macro = df_macro.copy() if df_macro is not None else None
        self.df_foreign = df_foreign.copy() if df_foreign is not None else None
        self.stock_context_records = list(stock_context_records) if stock_context_records is not None else []
        self.event_records = list(event_records) if event_records is not None else []
        self.shares_outstanding = float(shares_outstanding)
        self.schema_version = schema_version
        self.mode = mode

        # Validate that available_at exists in df_ohlcv or derive from timezone-aware index
        if "available_at" not in self.df_ohlcv.columns:
            if isinstance(self.df_ohlcv.index, pd.DatetimeIndex) and self.df_ohlcv.index.tz is not None:
                # Align trade close to 15:00:00 Taipei
                self.df_ohlcv["available_at"] = [
                    ts.floor("D") + pd.Timedelta(hours=15) for ts in self.df_ohlcv.index
                ]
            else:
                if self.mode == "formal":
                    raise ValueError(
                        "Strict PIT Error: df_ohlcv must have a timezone-aware 'available_at' column in formal mode."
                    )
                self.df_ohlcv["available_at"] = pd.date_range(
                    "2024-01-01 15:00:00", periods=len(self.df_ohlcv), freq="D", tz="Asia/Taipei"
                )

    def build_snapshot(
        self,
        decision_timestamp: pd.Timestamp,
        position_state: Optional[np.ndarray] = None,
    ) -> MultimodalBatch:
        """
        Builds a single-observation MultimodalBatch (B=1) strictly as of decision_timestamp.
        Guarantees that NO feature uses data with available_at > decision_timestamp.
        """
        if decision_timestamp.tzinfo is None:
            raise ValueError(
                f"Strict PIT Error: decision_timestamp must be timezone-aware (got {decision_timestamp})."
            )

        # 1. As-Of slice Market OHLCV
        ohlcv_avail = self.df_ohlcv[self.df_ohlcv["available_at"] <= decision_timestamp]
        if len(ohlcv_avail) < MARKET_WINDOW_LENGTH:
            raise ValueError(
                f"Insufficient historical data: need at least {MARKET_WINDOW_LENGTH} trading days "
                f"available as of {decision_timestamp}, but only found {len(ohlcv_avail)}."
            )

        # Compute price & technical features on the historical slice
        p_all = compute_price_features(
            ohlcv_avail, shares_outstanding=self.shares_outstanding, schema_version=self.schema_version
        )
        t_all = compute_technical_features(ohlcv_avail, schema_version=self.schema_version)

        # Take exact 60-day window ending as of decision_timestamp
        p_window = p_all[-MARKET_WINDOW_LENGTH:]
        t_window = t_all[-MARKET_WINDOW_LENGTH:]
        ohlcv_window = ohlcv_avail.iloc[-MARKET_WINDOW_LENGTH:]
        window_dates = ohlcv_window.index

        # 2. As-Of slice Institutional Data
        inst_missing_mask = np.zeros((MARKET_WINDOW_LENGTH, INSTITUTIONAL_DIM), dtype=bool)
        if self.df_inst is not None:
            inst_avail_col = "available_at" if "available_at" in self.df_inst.columns else None
            if inst_avail_col is not None:
                inst_avail = self.df_inst[self.df_inst[inst_avail_col] <= decision_timestamp]
            else:
                inst_avail = self.df_inst.loc[self.df_inst.index <= decision_timestamp]

            # Reindex to match the exact 60 trading days of the OHLCV window
            inst_aligned = inst_avail.reindex(window_dates)
            # Tag missing institutional observations
            if inst_aligned.isnull().any().any():
                inst_missing_mask[inst_aligned.isnull().values] = True
                inst_aligned = inst_aligned.fillna(0.0)

            i_window = compute_institutional_features(
                inst_aligned,
                ohlcv_window["volume"],
                shares_outstanding=self.shares_outstanding,
                schema_version=self.schema_version,
            )
        else:
            i_window = np.zeros((MARKET_WINDOW_LENGTH, INSTITUTIONAL_DIM), dtype=np.float32)
            inst_missing_mask[:] = True

        # 3. As-Of slice Macro & Foreign Data
        macro_missing_mask = np.zeros((MARKET_WINDOW_LENGTH, MACRO_DIM), dtype=bool)
        if self.df_macro is not None:
            macro_avail_col = "available_at" if "available_at" in self.df_macro.columns else None
            if macro_avail_col is not None:
                m_avail = self.df_macro[self.df_macro[macro_avail_col] <= decision_timestamp]
            else:
                m_avail = self.df_macro.loc[self.df_macro.index <= decision_timestamp]

            m_aligned = m_avail.reindex(window_dates)
            if m_aligned.isnull().any().any():
                macro_missing_mask[m_aligned.isnull().values] = True
                m_aligned = m_aligned.ffill().fillna(0.0)

            if self.df_foreign is not None:
                # Foreign market strictly As-Of filtered
                foreign_avail = self.df_foreign[self.df_foreign["available_at"] <= decision_timestamp]
            else:
                foreign_avail = None

            m_window = compute_macro_features(
                m_aligned,
                df_foreign=foreign_avail,
                schema_version=self.schema_version,
                mode=self.mode,
            )
        else:
            m_window = np.zeros((MARKET_WINDOW_LENGTH, MACRO_DIM), dtype=np.float32)
            macro_missing_mask[:] = True

        # 4. As-Of slice Stock Context (P0-1 requirement 4: point-in-time dynamic calculations)
        ctx_missing_mask = np.zeros(STOCK_CONTEXT_DIM, dtype=bool)
        last_asof_close = float(ohlcv_window["close"].iloc[-1])
        market_cap_asof = float(last_asof_close * self.shares_outstanding)
        trading_val_60d_ma = float(ohlcv_window["value"].mean())

        # Select latest stock context record strictly available_at <= decision_timestamp
        avail_ctx_records = [
            r for r in self.stock_context_records
            if r.get("provenance") is not None and r["provenance"].is_available_as_of(decision_timestamp, mode=self.mode)
        ]
        if avail_ctx_records:
            latest_ctx_rec = max(avail_ctx_records, key=lambda r: (r["provenance"].available_at, r["provenance"].revision_id))
            ctx_vec = build_audited_stock_context(
                industry_embedding_8d=latest_ctx_rec.get("industry_embedding_8d", np.zeros(8, dtype=np.float32)),
                market_cap=market_cap_asof,
                trading_value_60d_ma=trading_val_60d_ma,
                volatility_percentile=latest_ctx_rec.get("volatility_percentile", 0.5),
                liquidity_percentile=latest_ctx_rec.get("liquidity_percentile", 0.5),
                beta_120d=latest_ctx_rec.get("beta_120d", 1.0),
                industry_market_cap_percentile=latest_ctx_rec.get("industry_market_cap_percentile", 0.5),
                days_since_ipo=latest_ctx_rec.get("days_since_ipo", 1000),
                had_trading_suspension_20d=latest_ctx_rec.get("had_trading_suspension_20d", False),
                provenance=latest_ctx_rec["provenance"],
                decision_time=decision_timestamp,
                schema_version=self.schema_version,
                mode=self.mode,
            )
        else:
            # Fallback neutral representation
            ctx_vec = np.zeros(STOCK_CONTEXT_DIM, dtype=np.float32)
            ctx_missing_mask[:] = True

        # 5. As-Of slice Events (P0-1 requirement 5)
        ev_meta_missing_mask = np.zeros((TOTAL_EVENT_SLOTS, EVENT_METADATA_DIM), dtype=bool)
        avail_events = [
            e for e in self.event_records
            if e.get("provenance") is not None and e["provenance"].is_available_as_of(decision_timestamp, mode=self.mode)
        ]
        # Sort by available_at descending (most recent first)
        avail_events.sort(key=lambda e: e["provenance"].available_at, reverse=True)

        ev_meta_list = []
        ev_rel_list = []
        ev_emb_recent = np.zeros((RECENT_EVENT_SLOTS, EVENT_TEXT_DIM), dtype=np.float32)
        ev_emb_active = np.zeros((ACTIVE_EVENT_SLOTS, EVENT_TEXT_DIM), dtype=np.float32)
        event_mask = np.zeros(TOTAL_EVENT_SLOTS, dtype=bool)

        num_events = min(len(avail_events), TOTAL_EVENT_SLOTS)
        for k in range(TOTAL_EVENT_SLOTS):
            if k < num_events:
                ev = avail_events[k]
                meta = build_audited_event_metadata(
                    actor_embedding_4d=ev.get("actor_embedding_4d", np.zeros(4, dtype=np.float32)),
                    type_embedding_8d=ev.get("type_embedding_8d", np.zeros(8, dtype=np.float32)),
                    status_embedding_4d=ev.get("status_embedding_4d", np.zeros(4, dtype=np.float32)),
                    evidence_traceability=ev.get("evidence_traceability", 1.0),
                    source_reliability=ev.get("source_reliability", 1.0),
                    source_independence=ev.get("source_independence", 1.0),
                    novelty_input_estimate=ev.get("novelty_input_estimate", 0.5),
                    market_surprise_preevent=ev.get("market_surprise_preevent", 0.0),
                    quantified_magnitude=ev.get("quantified_magnitude", 0.5),
                    timestamp_reliability=ev.get("timestamp_reliability", 1.0),
                    provenance=ev["provenance"],
                    decision_time=decision_timestamp,
                    schema_version=self.schema_version,
                    mode=self.mode,
                )
                rel_dict = ev.get("relation", {})
                rel = assemble_event_stock_relation(
                    direct_company_mention=rel_dict.get("direct_company_mention", False),
                    industry_match=rel_dict.get("industry_match", False),
                    upstream_exposure=rel_dict.get("upstream_exposure", 0.0),
                    downstream_exposure=rel_dict.get("downstream_exposure", 0.0),
                    customer_relationship=rel_dict.get("customer_relationship", 0.0),
                    supplier_relationship=rel_dict.get("supplier_relationship", 0.0),
                    geographic_exposure=rel_dict.get("geographic_exposure", 0.0),
                    macro_factor_exposure=rel_dict.get("macro_factor_exposure", 0.0),
                )
                ev_meta_list.append(meta)
                ev_rel_list.append(rel)
                event_mask[k] = True

                # Fill embeddings
                text_emb = ev.get("text_embedding", np.zeros(EVENT_TEXT_DIM, dtype=np.float32))
                norm = np.linalg.norm(text_emb)
                if norm > 1e-6:
                    text_emb = text_emb / norm
                else:
                    text_emb[0] = 1.0

                if k < RECENT_EVENT_SLOTS:
                    ev_emb_recent[k] = text_emb
                if k < ACTIVE_EVENT_SLOTS:
                    ev_emb_active[k] = text_emb
            else:
                # Masked padded slot
                ev_meta_list.append(np.zeros(EVENT_METADATA_DIM, dtype=np.float32))
                ev_rel_list.append(np.zeros(EVENT_STOCK_RELATION_DIM, dtype=np.float32))
                ev_meta_missing_mask[k, :] = True

        # Ensure unit norm for zero-padded embedding slots so layernorm is numerically stable
        for rk in range(RECENT_EVENT_SLOTS):
            if np.linalg.norm(ev_emb_recent[rk]) < 1e-6:
                ev_emb_recent[rk, 0] = 1.0
        for ak in range(ACTIVE_EVENT_SLOTS):
            if np.linalg.norm(ev_emb_active[ak]) < 1e-6:
                ev_emb_active[ak, 0] = 1.0

        # 6. Position State
        if position_state is not None:
            pos_vec = np.asarray(position_state, dtype=np.float32)
        else:
            pos_vec = np.zeros(POSITION_STATE_DIM, dtype=np.float32)

        # 7. Assemble Single Observation MultimodalBatch (B=1)
        batch = MultimodalBatch(
            price=torch.tensor(p_window, dtype=torch.float32).unsqueeze(0),
            technical=torch.tensor(t_window, dtype=torch.float32).unsqueeze(0),
            institutional=torch.tensor(i_window, dtype=torch.float32).unsqueeze(0),
            macro=torch.tensor(m_window, dtype=torch.float32).unsqueeze(0),
            stock_context=torch.tensor(ctx_vec, dtype=torch.float32).unsqueeze(0),
            recent_event_emb=torch.tensor(ev_emb_recent, dtype=torch.float32).unsqueeze(0),
            active_event_emb=torch.tensor(ev_emb_active, dtype=torch.float32).unsqueeze(0),
            event_metadata=torch.tensor(np.array(ev_meta_list), dtype=torch.float32).unsqueeze(0),
            event_stock_rel=torch.tensor(np.array(ev_rel_list), dtype=torch.float32).unsqueeze(0),
            event_mask=torch.tensor(event_mask, dtype=torch.bool).unsqueeze(0),
            position=torch.tensor(pos_vec, dtype=torch.float32).unsqueeze(0),
            relevance_labels=torch.ones((1, TOTAL_EVENT_SLOTS, 1), dtype=torch.float32),
            novelty_labels=torch.zeros((1, TOTAL_EVENT_SLOTS, 1), dtype=torch.float32),
            return_targets=torch.zeros((1, 3), dtype=torch.float32),
            risk_targets=torch.zeros((1, 3), dtype=torch.float32),
            relevance_mask=torch.tensor(event_mask, dtype=torch.bool).unsqueeze(0).unsqueeze(-1),
            novelty_mask=torch.tensor(event_mask, dtype=torch.bool).unsqueeze(0).unsqueeze(-1),
            return_mask=torch.ones((1, 3), dtype=torch.bool),
            risk_mask=torch.ones((1, 3), dtype=torch.bool),
            macro_missing_mask=torch.tensor(macro_missing_mask, dtype=torch.bool).unsqueeze(0),
            institutional_missing_mask=torch.tensor(inst_missing_mask, dtype=torch.bool).unsqueeze(0),
            stock_context_missing_mask=torch.tensor(ctx_missing_mask, dtype=torch.bool).unsqueeze(0),
            event_metadata_missing_mask=torch.tensor(ev_meta_missing_mask, dtype=torch.bool).unsqueeze(0),
            schema_version=self.schema_version,
        )
        return batch

    def build_batch(
        self,
        decision_timestamps: List[pd.Timestamp],
        position_states: Optional[List[np.ndarray]] = None,
    ) -> MultimodalBatch:
        """
        Builds a batched MultimodalBatch across multiple sequential decision timestamps.
        Each sample in the batch is constructed strictly via its independent As-Of snapshot.
        """
        single_batches = []
        for idx, dec_ts in enumerate(decision_timestamps):
            pos = position_states[idx] if position_states is not None else None
            snap = self.build_snapshot(dec_ts, position_state=pos)
            single_batches.append(snap)

        from stock_rl.rl.replay_buffer import MultimodalReplayBuffer
        return MultimodalReplayBuffer._collate_batches(single_batches, schema_version=self.schema_version)

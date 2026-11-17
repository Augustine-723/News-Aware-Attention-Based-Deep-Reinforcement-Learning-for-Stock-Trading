"""
Decision-Aligned Historical Snapshot Builder per Specification V8.0 / V8.3 (P0-1..P0-3, P1-4..P1-5)
Constructs strictly As-Of feature snapshots for discrete decision timestamps:
- P0-1: Formal OHLCV availability fails closed: strictly requires explicit timezone-aware available_at.
  DataFrame index automatic estimation is strictly prohibited in formal mode.
- P0-2: Institutional & Macro Historical As-Of Alignment:
  Mandates explicit timezone-aware available_at in formal mode.
  Separates observation_date, trading_date, and available_at.
  Prohibits unverified df.index <= decision_time or direct .reindex().
  Performs timezone-aware As-Of join against 60-day historical trading days.
- P0-3: Strict 1-to-1 Slot Alignment between Recent (0..31) and Active (32..39) events:
  Guarantees slot k correspondence across text embeddings, metadata, relations, and masks.
  Enforces strict separation (disjoint sets) between recent and active events.
- P1-4: Eliminates fake supervised ground truth. When ground truth is absent,
  all label validity masks are explicitly False. Real returns strictly computed from T+1 executable open.
- P1-5: Canonical Content Hashing across all features, missing masks, targets, masks, schema, shapes & dtypes.
"""

import warnings
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch

from stock_rl.config.schema import (
    MARKET_WINDOW_LENGTH,
    RECENT_EVENT_SLOTS,
    ACTIVE_EVENT_SLOTS,
    TOTAL_EVENT_SLOTS,
    NUM_HORIZONS,
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
from stock_rl.data.normalizer import SplitManifest


class HistoricalSnapshotBuilder:
    """
    P0-1..P0-3, P1-4..P1-5 Unified Historical Snapshot Builder.
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
        active_event_records: Optional[List[Dict[str, Any]]] = None,
        shares_outstanding: float = 1e8,
        schema_version: str = CURRENT_ACTIVE_SCHEMA_VERSION,
        mode: str = "formal",
        builder_version: str = "8.3.0",
        dataset_id: Optional[str] = None,
    ):
        self.df_ohlcv = df_ohlcv.copy()
        self.df_inst = df_inst.copy() if df_inst is not None else None
        self.df_macro = df_macro.copy() if df_macro is not None else None
        self.df_foreign = df_foreign.copy() if df_foreign is not None else None
        self.stock_context_records = list(stock_context_records) if stock_context_records is not None else []
        self.event_records = list(event_records) if event_records is not None else []
        self.active_event_records = list(active_event_records) if active_event_records is not None else []
        self.shares_outstanding = float(shares_outstanding)
        self.schema_version = schema_version
        self.mode = mode
        self.builder_version = builder_version
        self.dataset_id = dataset_id

        # ----------------------------------------------------------------------
        # P0-1: Formal OHLCV Availability Must Fail Closed
        # ----------------------------------------------------------------------
        if "available_at" not in self.df_ohlcv.columns:
            if self.mode == "formal":
                raise ValueError(
                    "Strict PIT Error: df_ohlcv missing required explicit 'available_at' column in formal mode. "
                    "Automatic estimation from DataFrame index is strictly prohibited."
                )
            warnings.warn(
                "Estimated Availability Fallback: deriving df_ohlcv 'available_at' from index in non-formal mode.",
                UserWarning,
            )
            if isinstance(self.df_ohlcv.index, pd.DatetimeIndex) and self.df_ohlcv.index.tz is not None:
                self.df_ohlcv["available_at"] = [
                    ts.floor("D") + pd.Timedelta(hours=15) for ts in self.df_ohlcv.index
                ]
            else:
                self.df_ohlcv["available_at"] = pd.date_range(
                    "2024-01-01 15:00:00", periods=len(self.df_ohlcv), freq="D", tz="Asia/Taipei"
                )

        if self.mode == "formal":
            if getattr(self.df_ohlcv["available_at"].dt, "tz", None) is None:
                raise ValueError(
                    "Strict PIT Error: 'available_at' in df_ohlcv must be timezone-aware in formal mode."
                )
            if "is_audit_verified" in self.df_ohlcv.columns:
                self.df_ohlcv = self.df_ohlcv[self.df_ohlcv["is_audit_verified"] == True]

        # ----------------------------------------------------------------------
        # P0-2: Institutional Data Validation
        # ----------------------------------------------------------------------
        if self.df_inst is not None:
            if "available_at" not in self.df_inst.columns:
                if self.mode == "formal":
                    raise ValueError(
                        "Strict PIT Error: df_inst missing required explicit 'available_at' column in formal mode. "
                        "Using DataFrame index as available_at is strictly prohibited."
                    )
                warnings.warn(
                    "Estimated Availability Fallback: deriving df_inst 'available_at' in non-formal mode.",
                    UserWarning,
                )
                if isinstance(self.df_inst.index, pd.DatetimeIndex) and self.df_inst.index.tz is not None:
                    self.df_inst["available_at"] = [
                        ts.floor("D") + pd.Timedelta(hours=17) for ts in self.df_inst.index
                    ]
                else:
                    self.df_inst["available_at"] = pd.date_range(
                        "2024-01-01 17:00:00", periods=len(self.df_inst), freq="D", tz="Asia/Taipei"
                    )

            if self.mode == "formal":
                if getattr(self.df_inst["available_at"].dt, "tz", None) is None:
                    raise ValueError(
                        "Strict PIT Error: 'available_at' in df_inst must be timezone-aware in formal mode."
                    )
                if "is_audit_verified" in self.df_inst.columns:
                    self.df_inst = self.df_inst[self.df_inst["is_audit_verified"] == True]

        # ----------------------------------------------------------------------
        # P0-2: Macro Data Validation
        # ----------------------------------------------------------------------
        if self.df_macro is not None:
            if "available_at" not in self.df_macro.columns:
                if self.mode == "formal":
                    raise ValueError(
                        "Strict PIT Error: df_macro missing required explicit 'available_at' column in formal mode. "
                        "Using DataFrame index as available_at is strictly prohibited."
                    )
                warnings.warn(
                    "Estimated Availability Fallback: deriving df_macro 'available_at' in non-formal mode.",
                    UserWarning,
                )
                if isinstance(self.df_macro.index, pd.DatetimeIndex) and self.df_macro.index.tz is not None:
                    self.df_macro["available_at"] = [
                        ts.floor("D") + pd.Timedelta(hours=18) for ts in self.df_macro.index
                    ]
                else:
                    self.df_macro["available_at"] = pd.date_range(
                        "2024-01-01 18:00:00", periods=len(self.df_macro), freq="D", tz="Asia/Taipei"
                    )

            if self.mode == "formal":
                if getattr(self.df_macro["available_at"].dt, "tz", None) is None:
                    raise ValueError(
                        "Strict PIT Error: 'available_at' in df_macro must be timezone-aware in formal mode."
                    )
                if "is_audit_verified" in self.df_macro.columns:
                    self.df_macro = self.df_macro[self.df_macro["is_audit_verified"] == True]

    def _asof_align_feed(
        self,
        df: Optional[pd.DataFrame],
        window_dates: pd.DatetimeIndex,
        decision_timestamp: pd.Timestamp,
        feature_dim: int,
        feed_name: str = "feed",
        ffill_missing: bool = False,
    ) -> Tuple[Optional[pd.DataFrame], np.ndarray]:
        """
        P0-2 As-Of aligns an auxiliary historical feed (Institutional, Macro) to the exact 60-day window dates.
        Strict PIT Rules:
        - Strictly filters available_at <= decision_timestamp.
        - Separates observation_date / trading_date from available_at.
        - Resolves historical revisions (picks latest available_at <= decision_timestamp).
        - Tracks missing observations in missing_mask.
        - Prohibits df.index <= decision_time or direct reindex on unverified index.
        """
        if df is None:
            return None, np.ones((len(window_dates), feature_dim), dtype=bool)

        # 1. Strictly filter by available_at <= decision_timestamp
        avail_mask = df["available_at"] <= decision_timestamp
        df_avail = df[avail_mask].copy()

        # 2. Extract trading / observation date explicitly
        if "trading_date" in df_avail.columns:
            t_dates = pd.to_datetime(df_avail["trading_date"]).dt.tz_localize(None).dt.floor("D")
        elif "observation_date" in df_avail.columns:
            t_dates = pd.to_datetime(df_avail["observation_date"]).dt.tz_localize(None).dt.floor("D")
        elif isinstance(df_avail.index, pd.DatetimeIndex):
            t_dates = df_avail.index.tz_localize(None).floor("D")
        else:
            t_dates = pd.to_datetime(df_avail.index).tz_localize(None).floor("D")
        df_avail["_trading_date_normalized"] = t_dates

        # 3. Resolve revisions: latest available_at per trading date
        sort_cols = ["available_at"]
        if "revision_id" in df_avail.columns:
            sort_cols.append("revision_id")
        df_avail = df_avail.sort_values(sort_cols).drop_duplicates(subset=["_trading_date_normalized"], keep="last")

        # 4. Map to exact window trading dates
        if isinstance(window_dates, pd.DatetimeIndex):
            w_dates = window_dates.tz_localize(None).floor("D")
        else:
            w_dates = pd.to_datetime(window_dates).tz_localize(None).floor("D")
        df_mapped = df_avail.set_index("_trading_date_normalized")
        aligned = df_mapped.reindex(w_dates)
        aligned.index = window_dates

        # 5. Extract feature columns and determine missing mask
        meta_cols = {
            "available_at", "_trading_date_normalized", "is_audit_verified",
            "observation_date", "trading_date", "revision_id", "provenance"
        }
        feature_cols = [c for c in aligned.columns if c not in meta_cols]
        aligned_features = aligned[feature_cols]

        is_null = aligned_features.isnull().values
        missing_mask = np.zeros((len(window_dates), feature_dim), dtype=bool)
        # If any raw input column for a trading date is missing, all derived features for that date are missing
        row_is_missing = is_null.any(axis=1)
        missing_mask[row_is_missing, :] = True

        # 6. Fill policy: forward-fill for continuous macro series, zero-fill for institutional
        if ffill_missing:
            filled_df = aligned_features.ffill().fillna(0.0)
        else:
            filled_df = aligned_features.fillna(0.0)

        return filled_df, missing_mask

    def compute_future_targets_asof(
        self,
        decision_timestamp: pd.Timestamp,
        purge_boundary_date: Optional[pd.Timestamp] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Formal Multi-Horizon Target Builder per V8.0 / P1-4:
        - Strictly computes forward returns from next-day executable open (T+1 Open).
        - 1D Return: (Open[T+2] - Open[T+1]) / Open[T+1]
        - 3D Return: (Open[T+4] - Open[T+1]) / Open[T+1]
        - 5D Return: (Open[T+6] - Open[T+1]) / Open[T+1]
        - Risk targets: |Return_h| (strictly non-negative return magnitude)
        - Boundary Purge: Any horizon whose end date exceeds purge_boundary_date
          is strictly masked out (return_mask=False, risk_mask=False).
        """
        ret_targets = np.zeros(NUM_HORIZONS, dtype=np.float32)
        ret_mask = np.zeros(NUM_HORIZONS, dtype=bool)
        risk_targets = np.zeros(NUM_HORIZONS, dtype=np.float32)
        risk_mask = np.zeros(NUM_HORIZONS, dtype=bool)

        d_decision = (
            decision_timestamp.tz_localize(None).floor("D")
            if decision_timestamp.tzinfo is not None
            else decision_timestamp.floor("D")
        )

        if isinstance(self.df_ohlcv.index, pd.DatetimeIndex):
            dates_norm = self.df_ohlcv.index.tz_localize(None).floor("D")
        else:
            dates_norm = pd.to_datetime(self.df_ohlcv.index).tz_localize(None).floor("D")

        future_mask = dates_norm > d_decision
        future_indices = np.where(future_mask)[0]

        # Horizons require:
        # T+1: future_indices[0] (execution open)
        # T+2: future_indices[1] (1D horizon exit)
        # T+4: future_indices[3] (3D horizon exit)
        # T+6: future_indices[5] (5D horizon exit)
        horizon_steps = [(0, 1), (0, 3), (0, 5)]  # (entry_idx, exit_idx) for 1D, 3D, 5D

        purge_d = None
        if purge_boundary_date is not None:
            if getattr(purge_boundary_date, "tzinfo", None) is not None:
                purge_d = purge_boundary_date.tz_localize(None).floor("D")
            else:
                purge_d = purge_boundary_date.floor("D")

        if len(future_indices) > 0:
            entry_idx = future_indices[0]
            entry_open = float(self.df_ohlcv["open"].iloc[entry_idx])
            if entry_open > 1e-7:
                for h, (step_in, step_out) in enumerate(horizon_steps):
                    if len(future_indices) > step_out:
                        exit_idx = future_indices[step_out]
                        exit_date = dates_norm[exit_idx]
                        if purge_d is not None and exit_date > purge_d:
                            # Purged boundary: crosses into validation/test split
                            continue
                        exit_open = float(self.df_ohlcv["open"].iloc[exit_idx])
                        ret_h = (exit_open - entry_open) / entry_open
                        ret_targets[h] = ret_h
                        ret_mask[h] = True
                        risk_targets[h] = abs(ret_h)
                        risk_mask[h] = True

        return ret_targets, ret_mask, risk_targets, risk_mask

    def build_snapshot(
        self,
        decision_timestamp: pd.Timestamp,
        position_state: Optional[np.ndarray] = None,
        include_ground_truth_targets: bool = False,
        purge_boundary_date: Optional[pd.Timestamp] = None,
    ) -> MultimodalBatch:
        """
        Builds a single-observation MultimodalBatch (B=1) strictly as of decision_timestamp.
        Guarantees that NO feature uses data with available_at > decision_timestamp.
        """
        if decision_timestamp.tzinfo is None:
            raise ValueError(
                f"Strict PIT Error: decision_timestamp must be timezone-aware (got {decision_timestamp})."
            )

        # ----------------------------------------------------------------------
        # 1. As-Of slice Market OHLCV
        # ----------------------------------------------------------------------
        ohlcv_avail = self.df_ohlcv[self.df_ohlcv["available_at"] <= decision_timestamp]
        if len(ohlcv_avail) < MARKET_WINDOW_LENGTH:
            raise ValueError(
                f"Insufficient historical data: need at least {MARKET_WINDOW_LENGTH} trading days "
                f"available as of {decision_timestamp}, but only found {len(ohlcv_avail)}."
            )

        p_all = compute_price_features(
            ohlcv_avail, shares_outstanding=self.shares_outstanding, schema_version=self.schema_version
        )
        t_all = compute_technical_features(ohlcv_avail, schema_version=self.schema_version)

        p_window = p_all[-MARKET_WINDOW_LENGTH:]
        t_window = t_all[-MARKET_WINDOW_LENGTH:]
        ohlcv_window = ohlcv_avail.iloc[-MARKET_WINDOW_LENGTH:]
        window_dates = ohlcv_window.index

        # ----------------------------------------------------------------------
        # 2. P0-2 As-Of slice Institutional Data
        # ----------------------------------------------------------------------
        inst_df, inst_missing_mask = self._asof_align_feed(
            self.df_inst,
            window_dates=window_dates,
            decision_timestamp=decision_timestamp,
            feature_dim=INSTITUTIONAL_DIM,
            feed_name="df_inst",
            ffill_missing=False,
        )
        if inst_df is not None:
            i_window = compute_institutional_features(
                inst_df,
                ohlcv_window["volume"],
                shares_outstanding=self.shares_outstanding,
                schema_version=self.schema_version,
            )
        else:
            i_window = np.zeros((MARKET_WINDOW_LENGTH, INSTITUTIONAL_DIM), dtype=np.float32)

        # ----------------------------------------------------------------------
        # 3. P0-2 As-Of slice Macro & Foreign Data
        # ----------------------------------------------------------------------
        macro_df, macro_missing_mask = self._asof_align_feed(
            self.df_macro,
            window_dates=window_dates,
            decision_timestamp=decision_timestamp,
            feature_dim=MACRO_DIM,
            feed_name="df_macro",
            ffill_missing=True,
        )
        if macro_df is not None:
            foreign_avail = None
            if self.df_foreign is not None:
                foreign_avail = self.df_foreign[self.df_foreign["available_at"] <= decision_timestamp]

            m_window = compute_macro_features(
                macro_df,
                df_foreign=foreign_avail,
                schema_version=self.schema_version,
                mode=self.mode,
            )
        else:
            m_window = np.zeros((MARKET_WINDOW_LENGTH, MACRO_DIM), dtype=np.float32)

        # ----------------------------------------------------------------------
        # 4. As-Of slice Stock Context
        # ----------------------------------------------------------------------
        ctx_missing_mask = np.zeros(STOCK_CONTEXT_DIM, dtype=bool)
        last_asof_close = float(ohlcv_window["close"].iloc[-1])
        market_cap_asof = float(last_asof_close * self.shares_outstanding)
        trading_val_60d_ma = float(ohlcv_window["value"].mean())

        avail_ctx_records = [
            r for r in self.stock_context_records
            if r.get("provenance") is not None and r["provenance"].is_available_as_of(decision_timestamp, mode=self.mode)
        ]
        if avail_ctx_records:
            latest_ctx_rec = max(
                avail_ctx_records,
                key=lambda r: (r["provenance"].available_at, getattr(r["provenance"], "revision_id", 0))
            )
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
            ctx_vec = np.zeros(STOCK_CONTEXT_DIM, dtype=np.float32)
            ctx_missing_mask[:] = True

        # ----------------------------------------------------------------------
        # 5. P0-3 As-Of slice Events: 1-to-1 Slot Alignment
        # ----------------------------------------------------------------------
        ev_meta_missing_mask = np.zeros((TOTAL_EVENT_SLOTS, EVENT_METADATA_DIM), dtype=bool)

        avail_events = [
            e for e in self.event_records
            if e.get("provenance") is not None and e["provenance"].is_available_as_of(decision_timestamp, mode=self.mode)
        ]
        avail_active = [
            e for e in self.active_event_records
            if e.get("provenance") is not None and e["provenance"].is_available_as_of(decision_timestamp, mode=self.mode)
        ]

        # Separate pools with strict disjoint sets guarantee
        active_pool = list(avail_active)
        recent_pool = []
        for e in avail_events:
            is_active = (
                e.get("is_active", False) is True
                or str(e.get("event_status", "")).lower() in ("active", "ongoing", "standing")
            )
            if is_active:
                active_pool.append(e)
            else:
                recent_pool.append(e)

        active_obj_ids = {id(e) for e in active_pool}
        active_str_ids = {e["event_id"] for e in active_pool if "event_id" in e}
        recent_pool = [
            e for e in recent_pool
            if id(e) not in active_obj_ids and e.get("event_id") not in active_str_ids
        ]

        recent_pool.sort(key=lambda e: e["provenance"].available_at, reverse=True)
        active_pool.sort(key=lambda e: e["provenance"].available_at, reverse=True)

        ev_meta_arr = np.zeros((TOTAL_EVENT_SLOTS, EVENT_METADATA_DIM), dtype=np.float32)
        ev_rel_arr = np.zeros((TOTAL_EVENT_SLOTS, EVENT_STOCK_RELATION_DIM), dtype=np.float32)
        ev_emb_recent = np.zeros((RECENT_EVENT_SLOTS, EVENT_TEXT_DIM), dtype=np.float32)
        ev_emb_active = np.zeros((ACTIVE_EVENT_SLOTS, EVENT_TEXT_DIM), dtype=np.float32)
        event_mask = np.zeros(TOTAL_EVENT_SLOTS, dtype=bool)

        # Supervised target arrays (P1-4: default all False)
        relevance_labels = np.zeros((TOTAL_EVENT_SLOTS, 1), dtype=np.float32)
        novelty_labels = np.zeros((TOTAL_EVENT_SLOTS, 1), dtype=np.float32)
        relevance_mask = np.zeros((TOTAL_EVENT_SLOTS, 1), dtype=bool)
        novelty_mask = np.zeros((TOTAL_EVENT_SLOTS, 1), dtype=bool)

        slot_event_ids: List[Optional[str]] = [None] * TOTAL_EVENT_SLOTS

        # Fill Recent Slots: 0..31
        num_recent = min(len(recent_pool), RECENT_EVENT_SLOTS)
        for k in range(RECENT_EVENT_SLOTS):
            if k < num_recent:
                ev = recent_pool[k]
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
                ev_meta_arr[k] = meta
                ev_rel_arr[k] = rel
                event_mask[k] = True

                text_emb = ev.get("text_embedding", np.zeros(EVENT_TEXT_DIM, dtype=np.float32))
                norm = np.linalg.norm(text_emb)
                if norm > 1e-6:
                    text_emb = text_emb / norm
                else:
                    text_emb = np.zeros(EVENT_TEXT_DIM, dtype=np.float32)
                    text_emb[0] = 1.0
                ev_emb_recent[k] = text_emb

                if "ground_truth_relevance" in ev:
                    relevance_labels[k, 0] = float(ev["ground_truth_relevance"])
                    relevance_mask[k, 0] = True
                if "ground_truth_novelty" in ev:
                    novelty_labels[k, 0] = float(ev["ground_truth_novelty"])
                    novelty_mask[k, 0] = True

                slot_event_ids[k] = ev.get("event_id", ev["provenance"].source_id)
            else:
                ev_meta_missing_mask[k, :] = True
                ev_emb_recent[k, 0] = 1.0

        # Fill Active Slots: 32..39
        num_active = min(len(active_pool), ACTIVE_EVENT_SLOTS)
        for m in range(ACTIVE_EVENT_SLOTS):
            global_k = RECENT_EVENT_SLOTS + m  # 32 + m
            if m < num_active:
                ev = active_pool[m]
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
                ev_meta_arr[global_k] = meta
                ev_rel_arr[global_k] = rel
                event_mask[global_k] = True

                text_emb = ev.get("text_embedding", np.zeros(EVENT_TEXT_DIM, dtype=np.float32))
                norm = np.linalg.norm(text_emb)
                if norm > 1e-6:
                    text_emb = text_emb / norm
                else:
                    text_emb = np.zeros(EVENT_TEXT_DIM, dtype=np.float32)
                    text_emb[0] = 1.0
                ev_emb_active[m] = text_emb

                if "ground_truth_relevance" in ev:
                    relevance_labels[global_k, 0] = float(ev["ground_truth_relevance"])
                    relevance_mask[global_k, 0] = True
                if "ground_truth_novelty" in ev:
                    novelty_labels[global_k, 0] = float(ev["ground_truth_novelty"])
                    novelty_mask[global_k, 0] = True

                slot_event_ids[global_k] = ev.get("event_id", ev["provenance"].source_id)
            else:
                ev_meta_missing_mask[global_k, :] = True
                ev_emb_active[m, 0] = 1.0

        # ----------------------------------------------------------------------
        # 6. Position State
        # ----------------------------------------------------------------------
        if position_state is not None:
            pos_vec = np.asarray(position_state, dtype=np.float32)
        else:
            pos_vec = np.zeros(POSITION_STATE_DIM, dtype=np.float32)

        # ----------------------------------------------------------------------
        # 7. P1-4 Future Multi-Horizon Return & Risk Targets
        # ----------------------------------------------------------------------
        return_targets = np.zeros(NUM_HORIZONS, dtype=np.float32)
        risk_targets = np.zeros(NUM_HORIZONS, dtype=np.float32)
        return_mask = np.zeros(NUM_HORIZONS, dtype=bool)
        risk_mask = np.zeros(NUM_HORIZONS, dtype=bool)

        if include_ground_truth_targets:
            r_targets, r_m, rk_targets, rk_m = self.compute_future_targets_asof(
                decision_timestamp=decision_timestamp,
                purge_boundary_date=purge_boundary_date,
            )
            return_targets[:] = r_targets
            return_mask[:] = r_m
            risk_targets[:] = rk_targets
            risk_mask[:] = rk_m

        # ----------------------------------------------------------------------
        # 8. Assemble Single Observation MultimodalBatch (B=1)
        # ----------------------------------------------------------------------
        batch = MultimodalBatch(
            price=torch.tensor(p_window, dtype=torch.float32).unsqueeze(0),
            technical=torch.tensor(t_window, dtype=torch.float32).unsqueeze(0),
            institutional=torch.tensor(i_window, dtype=torch.float32).unsqueeze(0),
            macro=torch.tensor(m_window, dtype=torch.float32).unsqueeze(0),
            stock_context=torch.tensor(ctx_vec, dtype=torch.float32).unsqueeze(0),
            recent_event_emb=torch.tensor(ev_emb_recent, dtype=torch.float32).unsqueeze(0),
            active_event_emb=torch.tensor(ev_emb_active, dtype=torch.float32).unsqueeze(0),
            event_metadata=torch.tensor(ev_meta_arr, dtype=torch.float32).unsqueeze(0),
            event_stock_rel=torch.tensor(ev_rel_arr, dtype=torch.float32).unsqueeze(0),
            event_mask=torch.tensor(event_mask, dtype=torch.bool).unsqueeze(0),
            position=torch.tensor(pos_vec, dtype=torch.float32).unsqueeze(0),
            relevance_labels=torch.tensor(relevance_labels, dtype=torch.float32).unsqueeze(0),
            novelty_labels=torch.tensor(novelty_labels, dtype=torch.float32).unsqueeze(0),
            return_targets=torch.tensor(return_targets, dtype=torch.float32).unsqueeze(0),
            risk_targets=torch.tensor(risk_targets, dtype=torch.float32).unsqueeze(0),
            relevance_mask=torch.tensor(relevance_mask, dtype=torch.bool).unsqueeze(0),
            novelty_mask=torch.tensor(novelty_mask, dtype=torch.bool).unsqueeze(0),
            return_mask=torch.tensor(return_mask, dtype=torch.bool).unsqueeze(0),
            risk_mask=torch.tensor(risk_mask, dtype=torch.bool).unsqueeze(0),
            macro_missing_mask=torch.tensor(macro_missing_mask, dtype=torch.bool).unsqueeze(0),
            institutional_missing_mask=torch.tensor(inst_missing_mask, dtype=torch.bool).unsqueeze(0),
            stock_context_missing_mask=torch.tensor(ctx_missing_mask, dtype=torch.bool).unsqueeze(0),
            event_metadata_missing_mask=torch.tensor(ev_meta_missing_mask, dtype=torch.bool).unsqueeze(0),
            schema_version=self.schema_version,
        )
        batch.slot_event_ids = [slot_event_ids]
        return batch

    def build_batch(
        self,
        decision_timestamps: List[pd.Timestamp],
        position_states: Optional[List[np.ndarray]] = None,
        include_ground_truth_targets: bool = False,
        purge_boundary_date: Optional[pd.Timestamp] = None,
    ) -> MultimodalBatch:
        """
        Builds a batched MultimodalBatch across multiple sequential decision timestamps.
        Each sample in the batch is constructed strictly via its independent As-Of snapshot.
        """
        single_batches = []
        for idx, dec_ts in enumerate(decision_timestamps):
            pos = position_states[idx] if position_states is not None else None
            snap = self.build_snapshot(
                dec_ts,
                position_state=pos,
                include_ground_truth_targets=include_ground_truth_targets,
                purge_boundary_date=purge_boundary_date,
            )
            single_batches.append(snap)

        from stock_rl.rl.replay_buffer import MultimodalReplayBuffer
        batch = MultimodalReplayBuffer._collate_batches(single_batches, schema_version=self.schema_version)
        batch.slot_event_ids = [s.slot_event_ids[0] for s in single_batches]
        return batch

    def compute_snapshot_hash(
        self,
        batch: MultimodalBatch,
        dataset_id: Optional[str] = None,
    ) -> str:
        """
        P1-5 Canonical Content Hashing for batches built by this builder.
        """
        return SplitManifest.compute_batch_content_hash(
            batch,
            builder_version=self.builder_version,
            dataset_id=dataset_id or self.dataset_id,
        )

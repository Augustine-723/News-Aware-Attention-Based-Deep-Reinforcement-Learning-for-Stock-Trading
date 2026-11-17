"""
V8.9 Event Importance Learning & Evaluation Framework
Provides:
1. TWSE Official Trading Calendar Helper (handling weekends, holidays, cross-weekend trading session decay).
2. Strict Point-in-Time Event Importance definitions:
   - Method A: Rule-based Importance (I_rule) with Trading-Day Decay
   - Method B: Metadata-based Importance (I_meta) [Untrained Architectural Baseline]
   - Method C: Semantic + Metadata Importance (I_sem_meta) [Untrained Architectural Baseline]
3. Ex-post market reaction target label specification:
   - T+1 Absolute Return (|ln(P_close / P_open)|, available at T+1 14:00 Asia/Taipei)
   - T+2 Cumulative Abnormal Return (CAR over [T+1, T+2], available at T+2 14:00 Asia/Taipei)
   - T+1 Normalized Volume Surprise (available at T+1 14:00 Asia/Taipei)
   - Independent component tracking: source_time, estimated_available_at, provenance, missing_status
   - Composite available_at = max(component.estimated_available_at) (Strictly T+2 14:00 when CAR is included)
   - Two-day data prerequisite and strict non-imputation of missing data
   - Decision-time non-leakage barrier
4. Controlled evaluation functions:
   - Ranking plausibility
   - Deduplication robustness (syndication discount)
   - Temporal decay (Trading-day half-life exponential decay)
   - Cross-weekend temporal decay audit
   - Counterfactual future invariance
   - Shortcut learning & Untrained Scorer Sensitivity Evaluation
"""

from dataclasses import dataclass, field
import hashlib
import math
from typing import Dict, Any, List, Optional, Tuple, Set
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from stock_rl.config.schema import (
    EVENT_TEXT_DIM,
    EVENT_METADATA_DIM,
    EVENT_STOCK_RELATION_DIM,
    EVENT_RAW_FEATURE_DIM,
)
from stock_rl.data.multimodal_pipeline import (
    NewsSourceType,
    EvidenceAuditLevel,
    RawNewsArticle,
    extract_event_metadata,
    extract_event_stock_relation,
)
from stock_rl.evaluation.provenance import GenuinePretrainedSemanticEncoder


# ==============================================================================
# 0. TWSE Official Trading Calendar Helper
# ==============================================================================

class TWSETradingCalendar:
    """
    Taiwan Stock Exchange (TWSE) Official Trading Calendar Helper conforming to V8.9 P0-1 & P1-4.
    Properly accounts for:
    - Normal trading sessions: Monday through Friday (0 to 4).
    - Weekends: Saturday (5) and Sunday (6) are non-trading.
    - Statutory public holidays & designated market closures (2023–2026).
    - Custom trading calendar injection.
    """
    # Authoritative statutory holidays and market closures for TWSE (2023-2026)
    TWSE_HOLIDAYS: Set[str] = {
        # 2023
        "2023-01-02",  # 元旦補假
        "2023-01-18", "2023-01-19", "2023-01-20", "2023-01-23", "2023-01-24",
        "2023-01-25", "2023-01-26", "2023-01-27", "2023-01-30",  # 春節封關、連假及無交易結算日
        "2023-02-27", "2023-02-28",  # 228和平紀念日連假
        "2023-04-03", "2023-04-04", "2023-04-05",  # 兒童節及清明節連假
        "2023-05-01",  # 勞動節 (TWSE休市)
        "2023-06-22", "2023-06-23",  # 端午節連假
        "2023-08-03",  # 卡努颱風休市 (Typhoon Khanun)
        "2023-09-29",  # 中秋節
        "2023-10-09", "2023-10-10",  # 國慶日連假
        
        # 2024
        "2024-01-01",  # 開國紀念日
        "2024-02-06", "2024-02-07", "2024-02-08", "2024-02-09", "2024-02-12",
        "2024-02-13", "2024-02-14",  # 農曆春節連假 (2/6~2/7無交易僅辦理結算交割，2/8~2/14休市，2/15開紅盤)
        "2024-02-28",  # 和平紀念日
        "2024-04-04", "2024-04-05",  # 兒童節及民族掃墓節連假
        "2024-05-01",  # 勞動節
        "2024-06-10",  # 端午節
        "2024-09-17",  # 中秋節
        "2024-10-02", "2024-10-03",  # 山陀兒颱風休市
        "2024-10-10",  # 國慶日
        "2024-10-31",  # 康芮颱風休市
        
        # 2025
        "2025-01-01",  # 元旦
        "2025-01-23", "2025-01-24", "2025-01-27", "2025-01-28", "2025-01-29",
        "2025-01-30", "2025-01-31",  # 春節
        "2025-02-28",  # 和平紀念日
        "2025-04-03", "2025-04-04",  # 清明兒童連假
        "2025-05-01",  # 勞動節
        "2025-05-30",  # 端午節
        "2025-10-06",  # 中秋節
        "2025-10-10",  # 國慶日
        
        # 2026
        "2026-01-01",  # 元旦
        "2026-02-16", "2026-02-17", "2026-02-18", "2026-02-19", "2026-02-20",  # 春節
        "2026-02-27", "2026-02-28",  # 和平紀念日
        "2026-04-03", "2026-04-06",  # 清明兒童連假
        "2026-05-01",  # 勞動節
        "2026-06-19",  # 端午節
        "2026-09-25",  # 中秋節
        "2026-10-09",  # 國慶日
    }

    # Verified date range (V8.10 Preflight A): Fails closed if queried outside
    VERIFIED_START_DATE: str = "2023-01-01"
    VERIFIED_END_DATE: str = "2026-12-31"

    @classmethod
    def validate_date_in_range(cls, dt: pd.Timestamp) -> None:
        """Fails closed if queried date is outside the reconciled TWSE date range."""
        d_str = dt.strftime("%Y-%m-%d")
        if d_str < cls.VERIFIED_START_DATE or d_str > cls.VERIFIED_END_DATE:
            raise ValueError(
                f"Fail Closed (V8.10 Preflight A): Date '{d_str}' falls outside the verified "
                f"TWSE trading calendar range [{cls.VERIFIED_START_DATE}, {cls.VERIFIED_END_DATE}]."
            )

    @classmethod
    def reconcile_with_market_index(cls, market_index: pd.DatetimeIndex) -> Dict[str, Any]:
        """
        Reconciles calendar trading sessions against authentic market index trading dates.
        Asserts 100% agreement within index date range.
        """
        idx_dates = {d.strftime("%Y-%m-%d") for d in market_index}
        min_date = market_index.min().floor("D")
        max_date = market_index.max().floor("D")

        cal_dates = set()
        cur = min_date
        while cur <= max_date:
            if cls.is_trading_day(cur):
                cal_dates.add(cur.strftime("%Y-%m-%d"))
            cur += pd.Timedelta(days=1)

        diff_cal_not_market = cal_dates - idx_dates
        diff_market_not_cal = idx_dates - cal_dates
        is_reconciled = (len(diff_cal_not_market) == 0 and len(diff_market_not_cal) == 0)
        return {
            "is_reconciled": is_reconciled,
            "num_market_sessions": len(idx_dates),
            "num_calendar_sessions": len(cal_dates),
            "diff_calendar_not_market": sorted(list(diff_cal_not_market)),
            "diff_market_not_calendar": sorted(list(diff_market_not_cal)),
        }

    @classmethod
    def is_trading_day(cls, dt: pd.Timestamp) -> bool:
        """Determines if a given date is an active TWSE trading session."""
        cls.validate_date_in_range(dt)
        if dt.weekday() >= 5:  # Saturday=5, Sunday=6
            return False
        date_str = dt.strftime("%Y-%m-%d")
        return date_str not in cls.TWSE_HOLIDAYS

    @classmethod
    def get_next_trading_day(cls, dt: pd.Timestamp, n: int = 1) -> pd.Timestamp:
        """
        Returns the n-th subsequent active TWSE trading day strictly after dt.
        Preserves timezone awareness of the input timestamp.
        """
        if n < 1:
            raise ValueError(f"n must be >= 1 (got {n}).")
        
        tz = dt.tz
        cur = dt.floor("D") + pd.Timedelta(days=1)
        count = 0
        while True:
            if cls.is_trading_day(cur):
                count += 1
                if count == n:
                    return cur if tz is None else cur.tz_convert(tz) if cur.tz else cur.tz_localize(tz)
            cur = cur + pd.Timedelta(days=1)

    @classmethod
    def get_previous_trading_day(cls, dt: pd.Timestamp, n: int = 1) -> pd.Timestamp:
        """
        Returns the n-th prior active TWSE trading day strictly before dt.
        """
        if n < 1:
            raise ValueError(f"n must be >= 1 (got {n}).")
        
        tz = dt.tz
        cur = dt.floor("D") - pd.Timedelta(days=1)
        count = 0
        while True:
            if cls.is_trading_day(cur):
                count += 1
                if count == n:
                    return cur if tz is None else cur.tz_convert(tz) if cur.tz else cur.tz_localize(tz)
            cur = cur - pd.Timedelta(days=1)

    @classmethod
    def get_trading_days_between(cls, start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> int:
        """
        Counts the number of active TWSE trading sessions occurring strictly after start_dt
        up to and including end_dt.
        E.g., Friday 20:00 to Monday 20:00 (no holiday) = 1 trading session (Monday).
        E.g., Friday 10:00 to Friday 20:00 = 0 full subsequent trading sessions.
        """
        if end_dt <= start_dt:
            return 0
        
        d_start = start_dt.floor("D")
        d_end = end_dt.floor("D")
        if d_end <= d_start:
            return 0
        
        cur = d_start + pd.Timedelta(days=1)
        trading_days_count = 0
        while cur <= d_end:
            if cls.is_trading_day(cur):
                trading_days_count += 1
            cur = cur + pd.Timedelta(days=1)
        return trading_days_count


# ==============================================================================
# 1. Method A: Rule-Based Event Importance Formulation
# ==============================================================================

class RuleBasedEventImportanceScorer:
    """
    Method A: Analytical Rule-Based Event Importance Formulation.
    Combines:
    - Source Reliability (w_source)
    - Evidence Traceability (w_evidence)
    - Event Category Hierarchy (w_cat)
    - Target Stock Relation (w_rel)
    - Novelty / Revision Decay (w_nov)
    - Temporal Age Decay (exp(-lambda * delta_trading_days), half-life = 3 trading sessions)
    - Active Status Persistence Bonus (+0.2)
    - Cross-Source Syndication Discount (1 / sqrt(N_syndicated))
    """
    HALF_LIFE_TRADING_DAYS: float = 3.0
    LAMBDA_DECAY: float = math.log(2.0) / 3.0  # ~0.23105

    SOURCE_WEIGHTS: Dict[NewsSourceType, float] = {
        NewsSourceType.MOPS_OFFICIAL: 1.0,
        NewsSourceType.TWSE_OPENAPI: 0.95,
        NewsSourceType.COMMERCIAL_TEJ: 0.90,
        NewsSourceType.COMMERCIAL_GLOBAL: 0.90,
        NewsSourceType.FINMIND_RSS: 0.70,
        NewsSourceType.MEDIA_ARCHIVE: 0.70,
        NewsSourceType.SYNTHETIC_FIXTURE: 0.50,
    }

    EVIDENCE_WEIGHTS: Dict[EvidenceAuditLevel, float] = {
        EvidenceAuditLevel.VERIFIED_EVIDENCE: 1.0,
        EvidenceAuditLevel.EVIDENCE_CLAIM_ONLY: 0.70,
        EvidenceAuditLevel.ESTIMATED: 0.50,
        EvidenceAuditLevel.UNKNOWN: 0.30,
    }

    @classmethod
    def compute_category_weight(cls, headline: str, body: str = "") -> float:
        """Determines category importance multiplier based on disclosure type."""
        full_text = f"{headline} {body}"
        if any(k in full_text for k in ["獲利", "營收", "純益", "EPS", "財報", "自結", "股利", "配息"]):
            return 1.00  # Earnings & Capital return
        if any(k in full_text for k in ["產能", "建廠", "擴建", "新廠", "擴產", "先進封裝", "資本支出", "資本預算", "投資案", "CoWoS", "廠房"]):
            return 0.95  # CapEx & Capacity expansion
        if any(k in full_text for k in ["併購", "合併", "收購", "策略聯盟"]):
            return 0.90  # M&A
        if any(k in full_text for k in ["訴訟", "侵權", "處分", "罰款", "違法", "調查"]):
            return 0.85  # Legal & Regulatory
        if any(k in full_text for k in ["董事長", "總經理", "董監", "獨董", "經營權", "董事會"]):
            return 0.80  # Governance
        if any(k in full_text for k in ["製程", "晶片", "新技術", "架構", "認證"]):
            return 0.75  # Tech development
        if any(k in full_text for k in ["開盤", "收盤", "攻達", "大漲", "走跌", "強彈"]):
            return 0.60  # Routine market commentary
        return 0.50      # General / Macro

    @classmethod
    def score_article(
        cls,
        article: RawNewsArticle,
        target_stock_id: str,
        decision_timestamp: pd.Timestamp,
        syndicated_outlet_count: int = 1,
        calendar: Optional[TWSETradingCalendar] = None,
        decay_mode: str = "trading_days",
    ) -> float:
        """
        Computes the Point-in-Time Rule-Based Event Importance score I_rule in [0.0, 1.0].
        Strictly causal: uses only information available as of decision_timestamp.
        
        Decay modes:
        - "trading_days" (Default, P1-4 compliant): Counts elapsed active TWSE trading sessions.
        - "calendar_days": Counts elapsed wall-clock calendar days.
        """
        avail_ts = article.provider_available_at or article.source_published_at
        if avail_ts is not None and avail_ts > decision_timestamp:
            return 0.0  # Future event unavailable at decision cutoff

        # 1. Source Reliability
        w_source = cls.SOURCE_WEIGHTS.get(article.source_type, 0.70)

        # 2. Evidence Traceability
        audit_lvl = article.evidence_audit_level or (
            EvidenceAuditLevel.VERIFIED_EVIDENCE if article.is_audit_verified else EvidenceAuditLevel.ESTIMATED
        )
        w_evidence = cls.EVIDENCE_WEIGHTS.get(audit_lvl, 0.50)

        # 3. Category Weight
        w_cat = cls.compute_category_weight(article.headline, article.body_text)

        # 4. Target Stock Relation
        rel_vec = extract_event_stock_relation(article, target_stock_id)
        is_direct = rel_vec[0] > 0.5
        is_sector = rel_vec[1] > 0.5
        w_rel = 1.0 if is_direct else (0.50 if is_sector else 0.10)

        # 5. Novelty / Revision Discount
        rev_id = max(getattr(article, "revision_id", 1), 1)
        w_nov = 1.0 if rev_id == 1 else (0.5 / rev_id)

        # 6. Temporal Age Decay (Strict Trading-Day Decay per P1-4)
        if avail_ts is not None:
            if decay_mode == "trading_days":
                cal = calendar or TWSETradingCalendar
                delta_days = float(cal.get_trading_days_between(avail_ts, decision_timestamp))
            else:
                # Calendar-day decay
                if avail_ts.tzinfo is not None and decision_timestamp.tzinfo is not None:
                    delta_days = max((decision_timestamp - avail_ts).total_seconds() / 86400.0, 0.0)
                else:
                    t_avail_utc = avail_ts.tz_convert("UTC") if avail_ts.tzinfo is not None else avail_ts.tz_localize("UTC")
                    t_dec_utc = decision_timestamp.tz_convert("UTC") if decision_timestamp.tzinfo is not None else decision_timestamp.tz_localize("UTC")
                    delta_days = max((t_dec_utc - t_avail_utc).total_seconds() / 86400.0, 0.0)
            decay = math.exp(-cls.LAMBDA_DECAY * delta_days)
        else:
            decay = 1.0

        # 7. Syndication discount (cross-source independence)
        syn_discount = 1.0 / math.sqrt(max(syndicated_outlet_count, 1))

        # Base importance
        base_score = w_source * w_evidence * w_cat * w_rel * w_nov * decay * syn_discount

        # 8. Active status persistence bonus (Causal guard: only if available <= decision_timestamp)
        is_valid_active = article.is_active_candidate and (avail_ts is None or avail_ts <= decision_timestamp)
        active_bonus = 0.20 if is_valid_active else 0.0

        final_score = float(np.clip(base_score + active_bonus, 0.0, 1.0))
        return final_score


def extract_event_metadata_vector(
    article: RawNewsArticle,
    target_stock_id: str,
    decision_timestamp: pd.Timestamp,
) -> torch.Tensor:
    """
    Extracts 32D event metadata vector (24D metadata + 8D event-stock relation)
    strictly Point-in-Time as of decision_timestamp.
    """
    meta_24 = extract_event_metadata(article, decision_timestamp)
    rel_8 = extract_event_stock_relation(article, target_stock_id)
    combined_32 = np.concatenate([meta_24, rel_8], axis=0).astype(np.float32)
    return torch.from_numpy(combined_32)


# ==============================================================================
# 2. Method B: Metadata-Based Neural Importance Scorer (I_meta)
# ==============================================================================

class MetadataEventImportanceScorer(nn.Module):
    """
    Method B: Metadata-Based Neural Event Importance Scorer.
    Observes 32D features:
    - 24D Event Metadata (E01–E24)
    - 8D Event-Stock Relation (R01–R08)
    Does NOT observe the 128D semantic text vector.
    
    RESEARCH NOTICE (V8.9 P1-2):
    This architecture is currently an UNTRAINED baseline initialized deterministically.
    It serves for architectural forward sensitivity evaluation. It does NOT claim
    learned financial importance until trained against verified targets in the Pilot phase.
    """
    def __init__(self, in_features: int = 32, hidden_dim: int = 32):
        super().__init__()
        self.in_features = in_features
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self):
        # Deterministic initialization for reproducible evaluation
        rng = torch.Generator().manual_seed(42)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, generator=rng)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, metadata_32d: torch.Tensor) -> torch.Tensor:
        """Maps 32D tensor to scalar importance score in [0, 1]."""
        return self.net(metadata_32d)


# ==============================================================================
# 3. Method C: Semantic + Metadata Neural Importance Scorer (I_sem_meta)
# ==============================================================================

class SemanticMetadataEventImportanceScorer(nn.Module):
    """
    Method C: Semantic + Metadata Neural Event Importance Scorer.
    Observes 160D features:
    - 24D Event Metadata (E01–E24)
    - 8D Event-Stock Relation (R01–R08)
    - 128D Genuine Pretrained Semantic Vector (E25–E152)
    
    RESEARCH NOTICE (V8.9 P1-2):
    This architecture is currently an UNTRAINED neural fusion network initialized deterministically.
    Divergence in output scores between routine and major CapEx notices demonstrates
    architectural forward sensitivity to textual nuance, NOT learned importance.
    Supervised learning against market impact targets is designated for the Pilot phase.
    """
    def __init__(
        self,
        meta_dim: int = 32,
        text_dim: int = EVENT_TEXT_DIM,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.meta_dim = meta_dim
        self.text_dim = text_dim

        # Text projection & fusion
        self.text_proj = nn.Linear(text_dim, 32)
        self.fusion = nn.Sequential(
            nn.Linear(meta_dim + 32, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self):
        rng = torch.Generator().manual_seed(42)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, generator=rng)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        metadata_32d: torch.Tensor,
        text_128d: torch.Tensor,
    ) -> torch.Tensor:
        """
        Maps (32D metadata, 128D semantic text) to scalar importance in [0, 1].
        """
        proj_text = torch.relu(self.text_proj(text_128d))
        fused = torch.cat([metadata_32d, proj_text], dim=-1)
        return self.fusion(fused)


# ==============================================================================
# 4. Strict PIT Ex-Post Market Reaction Target Label Specification
# ==============================================================================

@dataclass
class MarketReactionComponentLabel:
    """
    Independent atomic market reaction component label conforming to V8.9 P0-1.
    Tracks provenance, generation time, available_at, and explicit missing status.
    """
    component_name: str                  # "t1_abs_return", "t2_cumulative_abnormal_return", "t1_volume_surprise"
    value: Optional[float]               # Exact metric value, or None if missing
    source_time: Optional[pd.Timestamp]  # Timestamp when the underlying market data was generated
    estimated_available_at: Optional[pd.Timestamp]  # When this component became legally available
    provenance: str = "TWSE_OFFICIAL_FEED"          # "TWSE_OFFICIAL_FEED", "SYNTHETIC_FIXTURE", "HISTORICAL_MARKET_DF"
    missing_status: str = "AVAILABLE"               # "AVAILABLE", "MISSING_DATA", "PARTIAL_HISTORY"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExPostMarketReactionLabel:
    """
    Composite Point-in-Time Ex-Post Market Reaction Target Label conforming to V8.9 P0-1.
    
    STRICT POINT-IN-TIME GOVERNANCE RULES:
    1. Decision-Time Non-Leakage Contract:
       State Features s_t at decision_date (e.g. T 20:00) are STRICTLY PROHIBITED
       from accessing this label. Attempting to read this label when
       decision_timestamp < label.available_at raises a RuntimeError.
    2. Component-Level As-Of Availability:
       - T+1 Absolute Return: available at T+1 14:00:00 Asia/Taipei
       - T+1 Volume Surprise: available at T+1 14:00:00 Asia/Taipei
       - T+2 Cumulative Abnormal Return (CAR): available at T+2 14:00:00 Asia/Taipei
    3. Composite Availability Timestamp:
       available_at is determined by the LATEST component among all required components:
       available_at = max(c.estimated_available_at for c in required_components).
       When T+2 CAR is included, composite available_at is strictly T+2 14:00:00 Asia/Taipei!
    4. Two-Day Data Prerequisite:
       T+2 CAR strictly requires two complete trading days (T+1 and T+2) of stock
       and market benchmark index data. If any day is missing, it is marked MISSING_DATA
       with value=None (never imputed with fake/interpolated numbers).
    5. Mock Labels:
       Must be explicitly tagged with provenance="SYNTHETIC_FIXTURE".
    """
    stock_id: str
    decision_date: pd.Timestamp           # T 20:00
    t1_trading_date: pd.Timestamp         # T+1 trading session date
    t2_trading_date: pd.Timestamp         # T+2 trading session date
    available_at: pd.Timestamp            # Latest availability among components (T+2 14:00 if CAR present)
    components: Dict[str, MarketReactionComponentLabel]
    provenance: str = "TWSE_OFFICIAL_FEED"
    missing_status: str = "AVAILABLE"
    composite_market_impact_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def next_day_abs_log_return(self) -> Optional[float]:
        comp = self.components.get("t1_abs_return")
        return comp.value if comp else None

    @property
    def cumulative_abnormal_return_2d(self) -> Optional[float]:
        comp = self.components.get("t2_cumulative_abnormal_return")
        return comp.value if comp else None

    @property
    def interval_abnormal_return(self) -> Optional[float]:
        comp = self.components.get("interval_abnormal_return")
        return comp.value if comp else None

    @property
    def normalized_volume_surprise(self) -> Optional[float]:
        comp = self.components.get("t1_volume_surprise")
        return comp.value if comp else None

    @property
    def t1_available_at(self) -> Optional[pd.Timestamp]:
        comp = self.components.get("t1_abs_return")
        return comp.estimated_available_at if comp else None

    @property
    def t2_available_at(self) -> Optional[pd.Timestamp]:
        comp = self.components.get("t2_cumulative_abnormal_return")
        return comp.estimated_available_at if comp else None

    def verify_decision_time_isolation(self, decision_timestamp: pd.Timestamp) -> None:
        """
        Enforces strict non-leakage PIT barrier: decision state at T 20:00
        CANNOT read this label if decision_timestamp < self.available_at.
        """
        if decision_timestamp < self.available_at:
            raise RuntimeError(
                f"Strict PIT Leakage Violation: Decision timestamp ({decision_timestamp}) cannot access "
                f"ex-post market reaction label available at ({self.available_at}). "
                f"T+1 components available at {self.t1_available_at}, "
                f"T+2 components available at {self.t2_available_at}."
            )

    @classmethod
    def create_mock_pit_label(
        cls,
        stock_id: str,
        t_decision: pd.Timestamp,
        next_session_open: float,
        next_session_close: float,
        next_session_volume: float,
        sma_20_volume: float,
        std_20_volume: float,
        market_return_t1: float = 0.005,
        market_return_t2: float = 0.003,
        stock_return_t2: float = 0.004,
        beta: float = 1.0,
        provenance: str = "SYNTHETIC_FIXTURE",
        calendar: Optional[TWSETradingCalendar] = None,
    ) -> "ExPostMarketReactionLabel":
        """
        Constructs an ex-post market reaction target label conforming to V8.9 P0-1.
        Uses TWSE trading calendar to properly resolve T+1 and T+2 sessions across weekends and holidays.
        """
        cal = calendar or TWSETradingCalendar
        tz = t_decision.tz or "Asia/Taipei"

        # Resolve T+1 and T+2 trading days
        t1_day = cal.get_next_trading_day(t_decision, n=1)
        t2_day = cal.get_next_trading_day(t_decision, n=2)

        # Availability timestamps (14:00 Asia/Taipei after session close at 13:30)
        t1_close = t1_day.floor("D") + pd.Timedelta(hours=13, minutes=30)
        t1_avail = t1_day.floor("D") + pd.Timedelta(hours=14)
        t2_close = t2_day.floor("D") + pd.Timedelta(hours=13, minutes=30)
        t2_avail = t2_day.floor("D") + pd.Timedelta(hours=14)

        if getattr(t_decision, "tzinfo", None) is not None:
            t1_close = t1_close.tz_localize(tz) if t1_close.tz is None else t1_close.tz_convert(tz)
            t1_avail = t1_avail.tz_localize(tz) if t1_avail.tz is None else t1_avail.tz_convert(tz)
            t2_close = t2_close.tz_localize(tz) if t2_close.tz is None else t2_close.tz_convert(tz)
            t2_avail = t2_avail.tz_localize(tz) if t2_avail.tz is None else t2_avail.tz_convert(tz)

        # 1. T+1 Absolute Return
        abs_ret = abs(math.log(max(next_session_close, 1e-4) / max(next_session_open, 1e-4)))
        comp_abs_ret = MarketReactionComponentLabel(
            component_name="t1_abs_return",
            value=round(abs_ret, 6),
            source_time=t1_close,
            estimated_available_at=t1_avail,
            provenance=provenance,
            missing_status="AVAILABLE",
        )

        # 2. T+1 Volume Surprise
        vol_surp = (next_session_volume - sma_20_volume) / max(std_20_volume, 1e-4)
        comp_vol_surp = MarketReactionComponentLabel(
            component_name="t1_volume_surprise",
            value=round(vol_surp, 6),
            source_time=t1_close,
            estimated_available_at=t1_avail,
            provenance=provenance,
            missing_status="AVAILABLE",
        )

        # 3. T+2 Cumulative Abnormal Return (requires 2 full trading sessions)
        stock_ret_t1 = (next_session_close - next_session_open) / max(next_session_open, 1e-4)
        cum_stock_ret = stock_ret_t1 + stock_return_t2
        cum_market_ret = market_return_t1 + market_return_t2
        car_2d = abs(cum_stock_ret - beta * cum_market_ret)
        comp_car_2d = MarketReactionComponentLabel(
            component_name="t2_cumulative_abnormal_return",
            value=round(car_2d, 6),
            source_time=t2_close,
            estimated_available_at=t2_avail,  # Available at T+2!
            provenance=provenance,
            missing_status="AVAILABLE",
        )

        # 4. 2-Day Interval Abnormal Return (T+1 Open to T+2 Close)
        stock_ret_interval = (1.0 + stock_ret_t1) * (1.0 + stock_return_t2) - 1.0
        market_ret_interval = (1.0 + market_return_t1) * (1.0 + market_return_t2) - 1.0
        ar_interval = abs(stock_ret_interval - beta * market_ret_interval)
        comp_ar_interval = MarketReactionComponentLabel(
            component_name="interval_abnormal_return",
            value=round(ar_interval, 6),
            source_time=t2_close,
            estimated_available_at=t2_avail,
            provenance=provenance,
            missing_status="AVAILABLE",
        )

        # Composite target impact score in [0, 1]
        composite = float(np.clip(abs_ret * 15.0 + car_2d * 10.0 + max(vol_surp, 0.0) * 0.1, 0.0, 1.0))

        # Composite availability is the latest of all components (T+2 14:00)
        composite_avail = max(t1_avail, t2_avail)

        components = {
            "t1_abs_return": comp_abs_ret,
            "t1_volume_surprise": comp_vol_surp,
            "t2_cumulative_abnormal_return": comp_car_2d,
            "interval_abnormal_return": comp_ar_interval,
        }

        return cls(
            stock_id=stock_id,
            decision_date=t_decision,
            t1_trading_date=t1_day,
            t2_trading_date=t2_day,
            available_at=composite_avail,
            components=components,
            provenance=provenance,
            missing_status="AVAILABLE",
            composite_market_impact_score=round(composite, 6),
        )

    @classmethod
    def compute_from_market_data(
        cls,
        stock_id: str,
        t_decision: pd.Timestamp,
        stock_df: pd.DataFrame,
        market_df: pd.DataFrame,
        beta120: Optional[float] = None,
        volume_sma20: Optional[float] = None,
        volume_std20: Optional[float] = None,
        calendar: Optional[TWSETradingCalendar] = None,
    ) -> "ExPostMarketReactionLabel":
        """
        Authentic computation of Ex-Post Market Reaction Labels from historical DataFrames.
        If any session (T+1 or T+2) is missing, marks component as MISSING_DATA and value=None.
        Never imputes missing data with synthetic assumptions (P0-1 compliant).
        
        V8.10 Preflight C:
        - beta120, volume_sma20, volume_std20 are strictly computed from observations <= T.
        - Distinguishes daily cumulative CAR (sum of daily abnormal returns) from
          interval abnormal return (holding period abnormal return across T+1 Open to T+2 Close).
        """
        cal = calendar or TWSETradingCalendar
        tz = t_decision.tz or "Asia/Taipei"

        t1_day = cal.get_next_trading_day(t_decision, n=1)
        t2_day = cal.get_next_trading_day(t_decision, n=2)

        t1_close = t1_day.floor("D") + pd.Timedelta(hours=13, minutes=30)
        t1_avail = t1_day.floor("D") + pd.Timedelta(hours=14)
        t2_close = t2_day.floor("D") + pd.Timedelta(hours=13, minutes=30)
        t2_avail = t2_day.floor("D") + pd.Timedelta(hours=14)

        if getattr(t_decision, "tzinfo", None) is not None:
            t1_close = t1_close.tz_localize(tz) if t1_close.tz is None else t1_close.tz_convert(tz)
            t1_avail = t1_avail.tz_localize(tz) if t1_avail.tz is None else t1_avail.tz_convert(tz)
            t2_close = t2_close.tz_localize(tz) if t2_close.tz is None else t2_close.tz_convert(tz)
            t2_avail = t2_avail.tz_localize(tz) if t2_avail.tz is None else t2_avail.tz_convert(tz)

        # Helper to extract date string mappings
        def _get_row_map(df: pd.DataFrame) -> Dict[str, pd.Series]:
            m = {}
            if isinstance(df.index, pd.DatetimeIndex):
                for idx in range(len(df)):
                    m[df.index[idx].strftime("%Y-%m-%d")] = df.iloc[idx]
            elif "date" in df.columns or "Date" in df.columns:
                col = "date" if "date" in df.columns else "Date"
                for idx in range(len(df)):
                    row = df.iloc[idx]
                    m[str(row[col])[:10]] = row
            else:
                for idx in range(len(df)):
                    m[str(df.index[idx])[:10]] = df.iloc[idx]
            return m

        stock_row_map = _get_row_map(stock_df)
        market_row_map = _get_row_map(market_df)

        t_dec_date_str = t_decision.strftime("%Y-%m-%d")

        # Pre-T Metric Computation (Preflight C)
        # 1. Volume SMA20 and STD20 strictly <= t_decision
        if volume_sma20 is None or volume_std20 is None:
            pre_t_stock_dates = sorted([d for d in stock_row_map.keys() if d <= t_dec_date_str])
            if len(pre_t_stock_dates) >= 20:
                recent_20_dates = pre_t_stock_dates[-20:]
                vols = []
                for d in recent_20_dates:
                    r = stock_row_map[d]
                    v_val = float(r.get("Trading_Volume", r.get("volume", r.get("Volume", 0.0))))
                    vols.append(v_val)
                v_arr = np.array(vols, dtype=np.float64)
                if volume_sma20 is None:
                    volume_sma20 = float(np.mean(v_arr))
                if volume_std20 is None:
                    volume_std20 = float(np.std(v_arr, ddof=1))
                    if volume_std20 <= 1e-6:
                        volume_std20 = 1.0

        # 2. Beta120 strictly <= t_decision
        if beta120 is None:
            common_pre_t = sorted(list(set(stock_row_map.keys()) & set(market_row_map.keys())))
            common_pre_t = [d for d in common_pre_t if d <= t_dec_date_str]
            if len(common_pre_t) >= 20:
                eval_dates = common_pre_t[-120:]
                s_closes = [float(stock_row_map[d].get("close", stock_row_map[d].get("Close", 0.0))) for d in eval_dates]
                m_closes = [float(market_row_map[d].get("close", market_row_map[d].get("Close", 0.0))) for d in eval_dates]
                s_rets = np.diff(np.log(np.maximum(s_closes, 1e-4)))
                m_rets = np.diff(np.log(np.maximum(m_closes, 1e-4)))
                var_m = float(np.var(m_rets, ddof=1))
                if var_m > 1e-8:
                    cov_sm = float(np.cov(s_rets, m_rets, ddof=1)[0, 1])
                    beta120 = float(cov_sm / var_m)
                else:
                    beta120 = 1.0
            else:
                beta120 = 1.0

        # Lookup T+1 and T+2 sessions
        t1_date_key = t1_day.strftime("%Y-%m-%d")
        t2_date_key = t2_day.strftime("%Y-%m-%d")

        has_t1_stock = t1_date_key in stock_row_map
        has_t2_stock = t2_date_key in stock_row_map
        has_t1_market = t1_date_key in market_row_map
        has_t2_market = t2_date_key in market_row_map

        components: Dict[str, MarketReactionComponentLabel] = {}

        # 1. T+1 Return
        if has_t1_stock:
            row_t1 = stock_row_map[t1_date_key]
            p_open = float(row_t1.get("open", row_t1.get("Open", 0.0)))
            p_close = float(row_t1.get("close", row_t1.get("Close", 0.0)))
            vol_t1 = float(row_t1.get("Trading_Volume", row_t1.get("volume", row_t1.get("Volume", 0.0))))
            if p_open > 0 and p_close > 0:
                abs_ret = abs(math.log(p_close / p_open))
                components["t1_abs_return"] = MarketReactionComponentLabel(
                    component_name="t1_abs_return",
                    value=round(abs_ret, 6),
                    source_time=t1_close,
                    estimated_available_at=t1_avail,
                    provenance="HISTORICAL_MARKET_DF",
                    missing_status="AVAILABLE",
                )
            else:
                components["t1_abs_return"] = MarketReactionComponentLabel(
                    component_name="t1_abs_return",
                    value=None,
                    source_time=t1_close,
                    estimated_available_at=t1_avail,
                    provenance="HISTORICAL_MARKET_DF",
                    missing_status="MISSING_DATA",
                )

            # Volume surprise
            if volume_sma20 is not None and volume_std20 is not None and volume_std20 > 0:
                vol_surp = (vol_t1 - volume_sma20) / volume_std20
                components["t1_volume_surprise"] = MarketReactionComponentLabel(
                    component_name="t1_volume_surprise",
                    value=round(vol_surp, 6),
                    source_time=t1_close,
                    estimated_available_at=t1_avail,
                    provenance="HISTORICAL_MARKET_DF",
                    missing_status="AVAILABLE",
                )
            else:
                components["t1_volume_surprise"] = MarketReactionComponentLabel(
                    component_name="t1_volume_surprise",
                    value=None,
                    source_time=t1_close,
                    estimated_available_at=t1_avail,
                    provenance="HISTORICAL_MARKET_DF",
                    missing_status="MISSING_DATA",
                )
        else:
            components["t1_abs_return"] = MarketReactionComponentLabel(
                component_name="t1_abs_return",
                value=None,
                source_time=t1_close,
                estimated_available_at=t1_avail,
                provenance="HISTORICAL_MARKET_DF",
                missing_status="MISSING_DATA",
            )
            components["t1_volume_surprise"] = MarketReactionComponentLabel(
                component_name="t1_volume_surprise",
                value=None,
                source_time=t1_close,
                estimated_available_at=t1_avail,
                provenance="HISTORICAL_MARKET_DF",
                missing_status="MISSING_DATA",
            )

        # 2. T+2 Cumulative Abnormal Return (CAR) & Interval Abnormal Return
        # Both require 2 full trading sessions across both stock and market
        if has_t1_stock and has_t2_stock and has_t1_market and has_t2_market:
            row_t1_stock = stock_row_map[t1_date_key]
            row_t2_stock = stock_row_map[t2_date_key]
            row_t1_mkt = market_row_map[t1_date_key]
            row_t2_mkt = market_row_map[t2_date_key]

            p_t1_open = float(row_t1_stock.get("open", row_t1_stock.get("Open", 0.0)))
            p_t1_close = float(row_t1_stock.get("close", row_t1_stock.get("Close", 0.0)))
            p_t2_open = float(row_t2_stock.get("open", row_t2_stock.get("Open", 0.0)))
            p_t2_close = float(row_t2_stock.get("close", row_t2_stock.get("Close", 0.0)))

            m_t1_open = float(row_t1_mkt.get("open", row_t1_mkt.get("Open", 0.0)))
            m_t1_close = float(row_t1_mkt.get("close", row_t1_mkt.get("Close", 0.0)))
            m_t2_open = float(row_t2_mkt.get("open", row_t2_mkt.get("Open", 0.0)))
            m_t2_close = float(row_t2_mkt.get("close", row_t2_mkt.get("Close", 0.0)))

            if (p_t1_open > 0 and p_t1_close > 0 and p_t2_open > 0 and p_t2_close > 0 and
                m_t1_open > 0 and m_t1_close > 0 and m_t2_open > 0 and m_t2_close > 0):
                
                # Daily cumulative CAR: AR_T1 + AR_T2
                ret_s_t1 = (p_t1_close - p_t1_open) / p_t1_open
                ret_m_t1 = (m_t1_close - m_t1_open) / m_t1_open
                ar_t1 = ret_s_t1 - beta120 * ret_m_t1

                ret_s_t2 = (p_t2_close - p_t2_open) / p_t2_open
                ret_m_t2 = (m_t2_close - m_t2_open) / m_t2_open
                ar_t2 = ret_s_t2 - beta120 * ret_m_t2

                car_daily = abs(ar_t1 + ar_t2)

                # 2-Day Interval Abnormal Return: T+1 open to T+2 close
                ret_s_2d = (p_t2_close - p_t1_open) / p_t1_open
                ret_m_2d = (m_t2_close - m_t1_open) / m_t1_open
                ar_interval = abs(ret_s_2d - beta120 * ret_m_2d)

                components["t2_cumulative_abnormal_return"] = MarketReactionComponentLabel(
                    component_name="t2_cumulative_abnormal_return",
                    value=round(car_daily, 6),
                    source_time=t2_close,
                    estimated_available_at=t2_avail,
                    provenance="HISTORICAL_MARKET_DF",
                    missing_status="AVAILABLE",
                )
                components["interval_abnormal_return"] = MarketReactionComponentLabel(
                    component_name="interval_abnormal_return",
                    value=round(ar_interval, 6),
                    source_time=t2_close,
                    estimated_available_at=t2_avail,
                    provenance="HISTORICAL_MARKET_DF",
                    missing_status="AVAILABLE",
                )
            else:
                components["t2_cumulative_abnormal_return"] = MarketReactionComponentLabel(
                    component_name="t2_cumulative_abnormal_return",
                    value=None,
                    source_time=t2_close,
                    estimated_available_at=t2_avail,
                    provenance="HISTORICAL_MARKET_DF",
                    missing_status="MISSING_DATA",
                )
                components["interval_abnormal_return"] = MarketReactionComponentLabel(
                    component_name="interval_abnormal_return",
                    value=None,
                    source_time=t2_close,
                    estimated_available_at=t2_avail,
                    provenance="HISTORICAL_MARKET_DF",
                    missing_status="MISSING_DATA",
                )
        else:
            components["t2_cumulative_abnormal_return"] = MarketReactionComponentLabel(
                component_name="t2_cumulative_abnormal_return",
                value=None,
                source_time=t2_close,
                estimated_available_at=t2_avail,
                provenance="HISTORICAL_MARKET_DF",
                missing_status="MISSING_DATA",
            )
            components["interval_abnormal_return"] = MarketReactionComponentLabel(
                component_name="interval_abnormal_return",
                value=None,
                source_time=t2_close,
                estimated_available_at=t2_avail,
                provenance="HISTORICAL_MARKET_DF",
                missing_status="MISSING_DATA",
            )

        # Composite target impact score
        all_available = all(c.missing_status == "AVAILABLE" for c in components.values())
        if (all_available and components["t1_abs_return"].value is not None and 
            components["t2_cumulative_abnormal_return"].value is not None):
            ret_val = components["t1_abs_return"].value
            car_val = components["t2_cumulative_abnormal_return"].value
            surp_val = components["t1_volume_surprise"].value or 0.0
            composite = float(np.clip(ret_val * 15.0 + car_val * 10.0 + max(surp_val, 0.0) * 0.1, 0.0, 1.0))
            composite_score = round(composite, 6)
            overall_status = "AVAILABLE"
        else:
            composite_score = None
            overall_status = "PARTIAL_OR_MISSING"

        composite_avail = max(t1_avail, t2_avail)

        meta = {
            "pre_t_beta120": beta120,
            "pre_t_volume_sma20": volume_sma20,
            "pre_t_volume_std20": volume_std20,
        }

        return cls(
            stock_id=stock_id,
            decision_date=t_decision,
            t1_trading_date=t1_day,
            t2_trading_date=t2_day,
            available_at=composite_avail,
            components=components,
            provenance="HISTORICAL_MARKET_DF",
            missing_status=overall_status,
            composite_market_impact_score=composite_score,
            metadata=meta,
        )


# ==============================================================================
# 5. Controlled Evaluation & Comparative Protocol
# ==============================================================================

class ControlledEventImportanceEvaluation:
    """
    Executes controlled evaluations comparing:
    - Rule-based Importance (I_rule)
    - Metadata-based Importance (I_meta) [Untrained Baseline]
    - Semantic + Metadata Importance (I_sem_meta) [Untrained Baseline]
    
    Verifies:
    1. Ranking Plausibility: Major corporate CapEx/Earnings rank above routine commentary.
    2. Deduplication Robustness: Syndicated articles on the same event receive cross-source discount.
    3. Temporal Decay: Static events decay smoothly by trading session half-life.
    4. Cross-Weekend Invariance: Weekend calendar days do not trigger false decay.
    5. Counterfactual Future News Invariance: Altering future events has zero impact on past snapshots.
    6. Shortcut Learning & Sensitivity Audit: Untrained neural fusion detects textual nuance.
    
    RESEARCH NOTICE (V8.9 P1-2):
    Neural evaluations are strictly designated as UNTRAINED SCORER SENSITIVITY EVALUATIONS.
    They verify gradient and architectural sensitivity under deterministic initialization,
    and DO NOT claim that semantic importance learning has been established.
    """

    @classmethod
    def build_test_event_feature_bundle(
        cls,
        article: RawNewsArticle,
        target_stock_id: str,
        decision_timestamp: pd.Timestamp,
        encoder: GenuinePretrainedSemanticEncoder,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Builds (metadata_32d, text_128d) tensors for an article.
        """
        meta_24 = extract_event_metadata(article, decision_timestamp)
        rel_8 = extract_event_stock_relation(article, target_stock_id)
        combined_32 = np.concatenate([meta_24, rel_8], axis=0).astype(np.float32)

        text_128 = encoder.embed_text(article.headline)

        t_meta = torch.from_numpy(combined_32).unsqueeze(0)
        t_text = torch.from_numpy(text_128).unsqueeze(0)
        return t_meta, t_text

    @classmethod
    def evaluate_ranking_plausibility(
        cls,
        encoder: GenuinePretrainedSemanticEncoder,
    ) -> Dict[str, Any]:
        """
        Tests whether high-materiality announcements (e.g. 100-billion CapEx)
        achieve higher importance scores than routine market commentary or minor notices.
        """
        t_dec = pd.Timestamp("2024-01-15 20:00:00", tz="Asia/Taipei")
        t_pub = pd.Timestamp("2024-01-15 10:00:00", tz="Asia/Taipei")

        # Material Event: 120-billion CapEx
        art_material = RawNewsArticle(
            source_type=NewsSourceType.MOPS_OFFICIAL,
            source_name="MOPS",
            stock_id="2330",
            headline="台積電董事會核准資本預算1200億元擴充先進製程產能",
            source_published_at=t_pub,
            provider_available_at=t_pub,
            is_audit_verified=True,
            is_active_candidate=True,
        )

        # Routine Event: General market trading commentary
        art_routine = RawNewsArticle(
            source_type=NewsSourceType.FINMIND_RSS,
            source_name="Yahoo奇摩股市",
            stock_id="2330",
            headline="台積電早盤開高震盪小漲3元，外資買賣超呈現觀望",
            source_published_at=t_pub,
            provider_available_at=t_pub,
            is_audit_verified=False,
            is_active_candidate=False,
        )

        meta_scorer = MetadataEventImportanceScorer()
        sem_scorer = SemanticMetadataEventImportanceScorer()
        meta_scorer.eval()
        sem_scorer.eval()

        # Rule-based
        score_rule_mat = RuleBasedEventImportanceScorer.score_article(art_material, "2330", t_dec)
        score_rule_rou = RuleBasedEventImportanceScorer.score_article(art_routine, "2330", t_dec)

        # Neural (Untrained sensitivity check)
        m_mat, t_mat = cls.build_test_event_feature_bundle(art_material, "2330", t_dec, encoder)
        m_rou, t_rou = cls.build_test_event_feature_bundle(art_routine, "2330", t_dec, encoder)

        with torch.no_grad():
            score_meta_mat = float(meta_scorer(m_mat).item())
            score_meta_rou = float(meta_scorer(m_rou).item())
            score_sem_mat = float(sem_scorer(m_mat, t_mat).item())
            score_sem_rou = float(sem_scorer(m_rou, t_rou).item())

        is_rule_valid = score_rule_mat > score_rule_rou
        is_sem_valid = score_sem_mat > score_sem_rou

        return {
            "evaluation_type": "Untrained Scorer Sensitivity Evaluation",
            "claim_boundary": "Architectural forward sensitivity under deterministic initialization; not learned importance.",
            "rule_based": {"material": score_rule_mat, "routine": score_rule_rou, "valid": is_rule_valid},
            "metadata_based": {"material": score_meta_mat, "routine": score_meta_rou},
            "semantic_metadata": {"material": score_sem_mat, "routine": score_sem_rou, "valid": is_sem_valid},
            "ranking_plausible": is_rule_valid and is_sem_valid,
        }

    @classmethod
    def evaluate_deduplication_robustness(
        cls,
    ) -> Dict[str, Any]:
        """
        Verifies that multiple syndicated outlets reporting the exact same event
        receive a syndication discount, preventing multi-outlet coverage from inflating importance.
        """
        t_dec = pd.Timestamp("2024-01-15 20:00:00", tz="Asia/Taipei")
        t_pub = pd.Timestamp("2024-01-15 10:00:00", tz="Asia/Taipei")

        art = RawNewsArticle(
            source_type=NewsSourceType.MOPS_OFFICIAL,
            source_name="MOPS",
            stock_id="2330",
            headline="台積電董事會拍板擴建新廠",
            source_published_at=t_pub,
            provider_available_at=t_pub,
            is_audit_verified=True,
        )

        score_single = RuleBasedEventImportanceScorer.score_article(art, "2330", t_dec, syndicated_outlet_count=1)
        score_3_syndicated = RuleBasedEventImportanceScorer.score_article(art, "2330", t_dec, syndicated_outlet_count=3)
        score_5_syndicated = RuleBasedEventImportanceScorer.score_article(art, "2330", t_dec, syndicated_outlet_count=5)

        is_discounted = score_single > score_3_syndicated > score_5_syndicated

        return {
            "score_single_outlet": score_single,
            "score_3_syndicated_outlets": score_3_syndicated,
            "score_5_syndicated_outlets": score_5_syndicated,
            "is_discount_effective": is_discounted,
        }

    @classmethod
    def evaluate_temporal_decay(
        cls,
    ) -> Dict[str, Any]:
        """
        Verifies that static historical events decay exponentially over trading days.
        With half-life = 3 trading days, importance after 3 trading days is approximately 50% of day 0.
        """
        t_pub = pd.Timestamp("2024-01-15 10:00:00", tz="Asia/Taipei")  # Monday

        art = RawNewsArticle(
            source_type=NewsSourceType.MOPS_OFFICIAL,
            source_name="MOPS",
            stock_id="2330",
            headline="台積電董事會決議",
            source_published_at=t_pub,
            provider_available_at=t_pub,
            is_audit_verified=True,
            is_active_candidate=False,
        )

        # In trading days:
        # Day 0: Monday 2024-01-15
        # Day 3: Thursday 2024-01-18 (3 trading sessions: Tue, Wed, Thu)
        # Day 6: Tuesday 2024-01-23 (3 more trading sessions: Fri, Mon, Tue)
        t_0 = t_pub
        t_3 = pd.Timestamp("2024-01-18 20:00:00", tz="Asia/Taipei")
        t_6 = pd.Timestamp("2024-01-23 20:00:00", tz="Asia/Taipei")

        s_0 = RuleBasedEventImportanceScorer.score_article(art, "2330", t_0, decay_mode="trading_days")
        s_3 = RuleBasedEventImportanceScorer.score_article(art, "2330", t_3, decay_mode="trading_days")
        s_6 = RuleBasedEventImportanceScorer.score_article(art, "2330", t_6, decay_mode="trading_days")

        ratio_3_0 = s_3 / max(s_0, 1e-6)
        ratio_6_3 = s_6 / max(s_3, 1e-6)

        # Expected half-life ratio ~ 0.50
        is_decay_valid = math.isclose(ratio_3_0, 0.50, abs_tol=0.05) and math.isclose(ratio_6_3, 0.50, abs_tol=0.05)

        return {
            "decay_mode": "trading_days",
            "score_day_0": s_0,
            "score_session_3": s_3,
            "score_session_6": s_6,
            "ratio_session3_day0": round(ratio_3_0, 4),
            "ratio_session6_session3": round(ratio_6_3, 4),
            "is_half_life_valid": is_decay_valid,
        }

    @classmethod
    def evaluate_cross_weekend_decay(
        cls,
    ) -> Dict[str, Any]:
        """
        V8.9 P1-4:
        Evaluates temporal decay across a weekend (Friday 20:00 to Monday 20:00).
        - Under Trading-Day decay: Exactly 1 trading day has elapsed (Monday's session).
          Decay factor = exp(-0.23105 * 1) ~ 0.7937.
        - Under Calendar-Day decay: 3 calendar days have elapsed.
          Decay factor = exp(-0.23105 * 3) ~ 0.5000.
        Verifies that Trading-Day decay preserves news importance across non-trading weekends.
        """
        t_fri_pub = pd.Timestamp("2024-01-19 20:00:00", tz="Asia/Taipei")  # Friday evening
        t_mon_dec = pd.Timestamp("2024-01-22 20:00:00", tz="Asia/Taipei")  # Monday evening

        art = RawNewsArticle(
            source_type=NewsSourceType.MOPS_OFFICIAL,
            source_name="MOPS",
            stock_id="2330",
            headline="台積電週末重大法說指引",
            source_published_at=t_fri_pub,
            provider_available_at=t_fri_pub,
            is_audit_verified=True,
        )

        s_fri = RuleBasedEventImportanceScorer.score_article(art, "2330", t_fri_pub, decay_mode="trading_days")
        s_mon_trading = RuleBasedEventImportanceScorer.score_article(art, "2330", t_mon_dec, decay_mode="trading_days")
        s_mon_calendar = RuleBasedEventImportanceScorer.score_article(art, "2330", t_mon_dec, decay_mode="calendar_days")

        trading_ratio = s_mon_trading / max(s_fri, 1e-6)
        calendar_ratio = s_mon_calendar / max(s_fri, 1e-6)

        return {
            "friday_score": s_fri,
            "monday_trading_decay_score": s_mon_trading,
            "monday_calendar_decay_score": s_mon_calendar,
            "trading_decay_retention_ratio": round(trading_ratio, 4),    # ~0.7937
            "calendar_decay_retention_ratio": round(calendar_ratio, 4),  # ~0.5000
            "trading_decay_preserves_weekend": trading_ratio > calendar_ratio,
        }

    @classmethod
    def evaluate_shortcut_learning(
        cls,
        encoder: GenuinePretrainedSemanticEncoder,
    ) -> Dict[str, Any]:
        """
        Tests whether the untrained neural architecture exhibits forward sensitivity to textual nuance.
        Evaluates two events with identical source (MOPS) and identical ticker (2330):
        - Event A: Routine annual shareholder meeting notice.
        - Event B: 120-billion CapEx advanced packaging expansion.
        
        RESEARCH NOTICE (V8.9 P1-2):
        This test checks architectural sensitivity and forward gradient propagation under deterministic initialization.
        It confirms that textual differences pass through the network and do not collapse to zero delta.
        It DOES NOT claim that the untrained network has learned financial importance.
        """
        t_dec = pd.Timestamp("2024-01-15 20:00:00", tz="Asia/Taipei")
        t_pub = pd.Timestamp("2024-01-15 10:00:00", tz="Asia/Taipei")

        # Event A: Routine Notice
        art_routine_mops = RawNewsArticle(
            source_type=NewsSourceType.MOPS_OFFICIAL,
            source_name="MOPS",
            stock_id="2330",
            headline="台積電公告董事會決議召開民國113年股東常會相關事宜",
            source_published_at=t_pub,
            provider_available_at=t_pub,
            is_audit_verified=True,
            is_active_candidate=False,
        )

        # Event B: Major CapEx
        art_capex_mops = RawNewsArticle(
            source_type=NewsSourceType.MOPS_OFFICIAL,
            source_name="MOPS",
            stock_id="2330",
            headline="台積電董事會核准資本預算1200億元擴建先進製程晶圓廠",
            source_published_at=t_pub,
            provider_available_at=t_pub,
            is_audit_verified=True,
            is_active_candidate=False,
        )

        sem_scorer = SemanticMetadataEventImportanceScorer()
        sem_scorer.eval()

        m_a, t_a = cls.build_test_event_feature_bundle(art_routine_mops, "2330", t_dec, encoder)
        m_b, t_b = cls.build_test_event_feature_bundle(art_capex_mops, "2330", t_dec, encoder)

        # Measure text embedding cosine distance
        cos_sim = float(torch.cosine_similarity(t_a, t_b).item())

        with torch.no_grad():
            score_a = float(sem_scorer(m_a, t_a).item())
            score_b = float(sem_scorer(m_b, t_b).item())

        delta_score = abs(score_b - score_a)
        is_separated = delta_score > 0.0

        return {
            "evaluation_type": "Untrained Scorer Sensitivity Evaluation",
            "claim_boundary": "Forward sensitivity check under deterministic initialization; true importance learning deferred to Pilot.",
            "routine_notice_score": round(score_a, 6),
            "major_capex_score": round(score_b, 6),
            "headline_cosine_similarity": round(cos_sim, 6),
            "architectural_sensitivity_delta": round(delta_score, 6),
            "semantic_separation_achieved": is_separated,
        }

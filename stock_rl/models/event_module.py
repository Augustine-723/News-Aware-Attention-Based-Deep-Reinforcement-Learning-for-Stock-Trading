"""
Event Projection, Age Encoding, and Conditioned Temporal Attention Module V8.0
Query condition formula per V8.0:
    q = BaseQuery + StockProjection(StockContext) + MarketProjection(MarketSummary)
Includes No Relevant Event (NRE) Token, Padding Mask, and Event Age Encoding.
"""

from typing import Tuple, Optional
import math
import torch
import torch.nn as nn
from stock_rl.config.schema import (
    TOTAL_EVENT_SLOTS,
    EVENT_RAW_FEATURE_DIM,
    EVENT_PROJECTION_INTERMEDIATE_DIM,
    EVENT_PROJECTION_OUTPUT_DIM,
    STOCK_CONTEXT_DIM,
    EVENT_AGE_METADATA_INDEX,
    get_event_age_index,
    get_feature_index,
)


class EventAgeEncoder(nn.Module):
    """
    Computes continuous age positional encoding for event timestamps.
    """
    def __init__(self, d_model: int = 128):
        super().__init__()
        self.d_model = d_model
        self.linear_age = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, age_days: torch.Tensor) -> torch.Tensor:
        # age_days: [B, 40, 1]
        return self.linear_age(age_days)


class EventModule(nn.Module):
    """
    Event processing module per V7.0/V8.0:
    - Projection: 160 -> 256 -> 128
    - Event-age encoding
    - No Relevant Event (NRE) learnable token
    - Stock & Market Conditioned Temporal Multi-Head Attention:
        q = BaseQuery + StockProjection(StockContext) + MarketProjection(MarketSummary)
    """
    def __init__(
        self,
        raw_dim: int = EVENT_RAW_FEATURE_DIM,               # 160
        hidden_dim: int = EVENT_PROJECTION_INTERMEDIATE_DIM,# 256
        out_dim: int = EVENT_PROJECTION_OUTPUT_DIM,         # 128
        stock_context_dim: int = STOCK_CONTEXT_DIM,         # 16
        market_summary_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        # Event Projection: 160 -> 256 -> 128
        self.projection = nn.Sequential(
            nn.Linear(raw_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

        self.age_encoder = EventAgeEncoder(d_model=out_dim)

        # No Relevant Event (NRE) Token [1, 1, out_dim]
        self.nre_token = nn.Parameter(torch.randn(1, 1, out_dim) * 0.02)

        # Base Query parameter [1, 1, out_dim]
        self.base_query = nn.Parameter(torch.randn(1, 1, out_dim) * 0.02)

        # Conditioning Projections for Stock Context and Market State
        self.stock_query_proj = nn.Sequential(
            nn.Linear(stock_context_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
        )
        self.market_query_proj = nn.Sequential(
            nn.Linear(market_summary_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
        )

        # Temporal Multi-Head Attention
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=out_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        event_text: torch.Tensor,                           # [B, 40, 128]
        event_meta: torch.Tensor,                           # [B, 40, 24]
        event_rel: torch.Tensor,                            # [B, 40, 8]
        event_mask: torch.Tensor,                           # [B, 40] bool (True = valid, False = pad)
        stock_context: Optional[torch.Tensor] = None,       # [B, 16]
        market_summary: Optional[torch.Tensor] = None,      # [B, 128]
        schema_version: Optional[str] = None,
        event_age_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            projected_events: [B, 40, 128] (individual projected event representations)
            aggregated_event_token: [B, 1, 128] (conditioned temporal attention summary)
        """
        B = event_text.shape[0]

        # 1. Concatenate raw features: [B, 40, 160]
        raw_features = torch.cat([event_text, event_meta, event_rel], dim=-1)

        # 2. Project: 160 -> 256 -> 128
        projected = self.projection(raw_features)  # [B, 40, 128]

        # 3. Add Event-Age encoding (age index retrieved dynamically from schema)
        if event_age_idx is not None:
            age_idx = event_age_idx
        else:
            age_idx = get_event_age_index(schema_version)
        event_age = event_meta[:, :, age_idx : age_idx + 1]  # [B, 40, 1]
        age_enc = self.age_encoder(event_age)
        projected_with_age = projected + age_enc

        # 4. Prepend NRE Token to key/value sequence -> [B, 41, 128]
        nre_expanded = self.nre_token.expand(B, 1, -1)  # [B, 1, 128]
        kv_sequence = torch.cat([nre_expanded, projected_with_age], dim=1)  # [B, 41, 128]

        # Construct key_padding_mask: True indicates value should be ignored
        # NRE token at index 0 is NEVER masked (so False)
        nre_mask = torch.zeros((B, 1), dtype=torch.bool, device=event_mask.device)
        events_key_mask = ~event_mask  # [B, 40]
        combined_key_mask = torch.cat([nre_mask, events_key_mask], dim=1)  # [B, 41]

        # 5. Conditioned Query: q = BaseQuery + StockProjection + MarketProjection
        query = self.base_query.expand(B, 1, -1)
        if stock_context is not None:
            stock_q = self.stock_query_proj(stock_context).unsqueeze(1)  # [B, 1, 128]
            query = query + stock_q
        if market_summary is not None:
            market_q = self.market_query_proj(market_summary).unsqueeze(1)  # [B, 1, 128]
            query = query + market_q

        # Temporal Attention using conditioned query
        attn_out, _ = self.temporal_attn(
            query=query,
            key=kv_sequence,
            value=kv_sequence,
            key_padding_mask=combined_key_mask,
        )
        aggregated_event_token = self.norm(query + attn_out)  # [B, 1, 128]

        return projected, aggregated_event_token

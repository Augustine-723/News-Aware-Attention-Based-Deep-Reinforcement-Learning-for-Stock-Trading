"""
High-Fidelity Synthetic Point-in-Time Data Generator
Generates verifiable multimodal market, macro, event, and label tensors strictly
matching Specification V8.0 schemas.
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
    novelty_labels: Optional[torch.Tensor] = None    # [B, 40, 1]
    return_targets: Optional[torch.Tensor] = None    # [B, 3]
    risk_targets: Optional[torch.Tensor] = None      # [B, 3]

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
        )


def generate_synthetic_batch(
    batch_size: int = 64,
    seed: int = 42,
    include_targets: bool = True
) -> MultimodalBatch:
    """
    Generates a synthetic multimodal batch strictly complying with V8.0 dimensions.
    """
    rng = np.random.default_rng(seed)

    price = torch.tensor(rng.standard_normal((batch_size, MARKET_WINDOW_LENGTH, PRICE_DIM)), dtype=torch.float32)
    technical = torch.tensor(rng.standard_normal((batch_size, MARKET_WINDOW_LENGTH, TECHNICAL_DIM)), dtype=torch.float32)
    institutional = torch.tensor(rng.standard_normal((batch_size, MARKET_WINDOW_LENGTH, INSTITUTIONAL_DIM)), dtype=torch.float32)
    macro = torch.tensor(rng.standard_normal((batch_size, MARKET_WINDOW_LENGTH, MACRO_DIM)), dtype=torch.float32)
    stock_context = torch.tensor(rng.standard_normal((batch_size, STOCK_CONTEXT_DIM)), dtype=torch.float32)

    # Event embeddings: unit-norm text vectors
    raw_recent = rng.standard_normal((batch_size, RECENT_EVENT_SLOTS, EVENT_TEXT_DIM))
    raw_active = rng.standard_normal((batch_size, ACTIVE_EVENT_SLOTS, EVENT_TEXT_DIM))
    recent_emb = torch.tensor(raw_recent / (np.linalg.norm(raw_recent, axis=-1, keepdims=True) + 1e-8), dtype=torch.float32)
    active_emb = torch.tensor(raw_active / (np.linalg.norm(raw_active, axis=-1, keepdims=True) + 1e-8), dtype=torch.float32)

    event_metadata = torch.tensor(rng.uniform(0.0, 1.0, (batch_size, TOTAL_EVENT_SLOTS, EVENT_METADATA_DIM)), dtype=torch.float32)
    event_stock_rel = torch.tensor(rng.uniform(0.0, 1.0, (batch_size, TOTAL_EVENT_SLOTS, EVENT_STOCK_RELATION_DIM)), dtype=torch.float32)

    # Event mask: simulate 5 to 30 active events per sample
    event_mask = torch.zeros((batch_size, TOTAL_EVENT_SLOTS), dtype=torch.bool)
    for b in range(batch_size):
        num_valid = rng.integers(5, TOTAL_EVENT_SLOTS + 1)
        event_mask[b, :num_valid] = True

    # Position features [target_position, pnl_pct, holding_days_norm, dist_limit]
    pos_actions = rng.choice([0.0, 1.0], size=(batch_size, 1))
    pos_pnl = rng.uniform(-0.15, 0.20, size=(batch_size, 1))
    pos_days = rng.uniform(0.0, 1.0, size=(batch_size, 1))
    pos_dist = rng.uniform(0.02, 0.10, size=(batch_size, 1))
    position = torch.tensor(np.hstack([pos_actions, pos_pnl, pos_days, pos_dist]), dtype=torch.float32)

    relevance_labels = None
    novelty_labels = None
    return_targets = None
    risk_targets = None

    if include_targets:
        # Relevance: binary classification for each event slot
        relevance_raw = (rng.uniform(0.0, 1.0, (batch_size, TOTAL_EVENT_SLOTS, 1)) > 0.4).astype(np.float32)
        relevance_labels = torch.tensor(relevance_raw, dtype=torch.float32)

        # Novelty: binary classification for each event slot
        novelty_raw = (rng.uniform(0.0, 1.0, (batch_size, TOTAL_EVENT_SLOTS, 1)) > 0.5).astype(np.float32)
        novelty_labels = torch.tensor(novelty_raw, dtype=torch.float32)

        # Excess Return targets over 1, 3, 5 days (centered around 0, approx daily std 1.5%)
        ret_1d = rng.normal(0.0005, 0.015, size=(batch_size, 1))
        ret_3d = rng.normal(0.0015, 0.025, size=(batch_size, 1))
        ret_5d = rng.normal(0.0025, 0.035, size=(batch_size, 1))
        return_targets = torch.tensor(np.hstack([ret_1d, ret_3d, ret_5d]), dtype=torch.float32)

        # Future Risk targets (realized return magnitude proxy > 0)
        risk_1d = np.abs(ret_1d) + rng.exponential(0.005, size=(batch_size, 1))
        risk_3d = np.abs(ret_3d) + rng.exponential(0.010, size=(batch_size, 1))
        risk_5d = np.abs(ret_5d) + rng.exponential(0.015, size=(batch_size, 1))
        risk_targets = torch.tensor(np.hstack([risk_1d, risk_3d, risk_5d]), dtype=torch.float32)

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
    )

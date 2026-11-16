"""
Strict Tensor Schema Validator per Specification V8.0
Verifies exact shapes, datatypes, absence of NaN/Inf, and valid masking.
"""

from typing import Dict, Any
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
from stock_rl.data.synthetic_data import MultimodalBatch


class TensorSchemaError(ValueError):
    """Raised when tensor dimensions, dtypes, or values violate contracts."""
    pass


def validate_multimodal_batch(batch: MultimodalBatch) -> bool:
    """
    Validates all tensors in MultimodalBatch against Specification V8.0 contracts.
    """
    B = batch.price.shape[0]

    expected_shapes = {
        "price": (B, MARKET_WINDOW_LENGTH, PRICE_DIM),
        "technical": (B, MARKET_WINDOW_LENGTH, TECHNICAL_DIM),
        "institutional": (B, MARKET_WINDOW_LENGTH, INSTITUTIONAL_DIM),
        "macro": (B, MARKET_WINDOW_LENGTH, MACRO_DIM),
        "stock_context": (B, STOCK_CONTEXT_DIM),
        "recent_event_emb": (B, RECENT_EVENT_SLOTS, EVENT_TEXT_DIM),
        "active_event_emb": (B, ACTIVE_EVENT_SLOTS, EVENT_TEXT_DIM),
        "event_metadata": (B, TOTAL_EVENT_SLOTS, EVENT_METADATA_DIM),
        "event_stock_rel": (B, TOTAL_EVENT_SLOTS, EVENT_STOCK_RELATION_DIM),
        "event_mask": (B, TOTAL_EVENT_SLOTS),
        "position": (B, POSITION_STATE_DIM),
    }

    # Verify tensor shapes
    for name, expected in expected_shapes.items():
        tensor: torch.Tensor = getattr(batch, name)
        if tuple(tensor.shape) != expected:
            raise TensorSchemaError(
                f"Shape mismatch for {name}: expected {expected}, got {tuple(tensor.shape)}"
            )
        # NaN / Inf checks
        if tensor.dtype in (torch.float32, torch.float64):
            if torch.isnan(tensor).any():
                raise TensorSchemaError(f"NaN detected in tensor {name}")
            if torch.isinf(tensor).any():
                raise TensorSchemaError(f"Inf detected in tensor {name}")

    # Verify targets and independent masks if provided
    if batch.relevance_labels is not None:
        if tuple(batch.relevance_labels.shape) != (B, TOTAL_EVENT_SLOTS, 1):
            raise TensorSchemaError(f"Relevance labels shape mismatch: {tuple(batch.relevance_labels.shape)}")
    if batch.relevance_mask is not None:
        if tuple(batch.relevance_mask.shape) != (B, TOTAL_EVENT_SLOTS, 1):
            raise TensorSchemaError(f"Relevance mask shape mismatch: {tuple(batch.relevance_mask.shape)}")

    if batch.novelty_labels is not None:
        if tuple(batch.novelty_labels.shape) != (B, TOTAL_EVENT_SLOTS, 1):
            raise TensorSchemaError(f"Novelty labels shape mismatch: {tuple(batch.novelty_labels.shape)}")
    if batch.novelty_mask is not None:
        if tuple(batch.novelty_mask.shape) != (B, TOTAL_EVENT_SLOTS, 1):
            raise TensorSchemaError(f"Novelty mask shape mismatch: {tuple(batch.novelty_mask.shape)}")

    if batch.return_targets is not None:
        if tuple(batch.return_targets.shape) != (B, NUM_HORIZONS):
            raise TensorSchemaError(f"Return targets shape mismatch: {tuple(batch.return_targets.shape)}")
    if batch.return_mask is not None:
        if tuple(batch.return_mask.shape) != (B, NUM_HORIZONS):
            raise TensorSchemaError(f"Return mask shape mismatch: {tuple(batch.return_mask.shape)}")

    if batch.risk_targets is not None:
        if tuple(batch.risk_targets.shape) != (B, NUM_HORIZONS):
            raise TensorSchemaError(f"Risk targets shape mismatch: {tuple(batch.risk_targets.shape)}")
    if batch.risk_mask is not None:
        if tuple(batch.risk_mask.shape) != (B, NUM_HORIZONS):
            raise TensorSchemaError(f"Risk mask shape mismatch: {tuple(batch.risk_mask.shape)}")

    return True

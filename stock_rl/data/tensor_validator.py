"""
Strict Tensor Schema & Semantic Validator per Specification V8.0 / V8.3
Verifies exact shapes, datatypes, absence of NaN/Inf, mask consistency,
value boundaries, and dynamic schema-specific Event Age indexing.
"""

from typing import Dict, Any, Optional
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
    validate_schema_version,
)
from stock_rl.data.synthetic_data import MultimodalBatch


class TensorSchemaError(ValueError):
    """Raised when tensor dimensions, dtypes, or values violate contracts."""
    pass


def validate_multimodal_batch(
    batch: MultimodalBatch,
    schema_version: Optional[str] = None,
    is_normalized: bool = False,
) -> bool:
    """
    Validates all tensors in MultimodalBatch against Specification V8.0 / V8.3 contracts.
    Enforces shape contracts, finite values, mask consistency, value bounds,
    and dynamic schema-dependent Event Age indices.
    When is_normalized=True, skips raw [0, 1] bounded interval checks on z-scored features.
    """
    effective_schema = schema_version or getattr(batch, "schema_version", CURRENT_ACTIVE_SCHEMA_VERSION)
    validate_schema_version(effective_schema)

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

    # 1. Verify tensor shapes & finite values
    for name, expected in expected_shapes.items():
        tensor: torch.Tensor = getattr(batch, name)
        if tuple(tensor.shape) != expected:
            raise TensorSchemaError(
                f"Shape mismatch for {name}: expected {expected}, got {tuple(tensor.shape)}"
            )
        if tensor.dtype in (torch.float32, torch.float64):
            if torch.isnan(tensor).any():
                raise TensorSchemaError(f"NaN detected in tensor {name}")
            if torch.isinf(tensor).any():
                raise TensorSchemaError(f"Inf detected in tensor {name}")

    # 2. Verify targets and independent masks
    if batch.relevance_labels is not None:
        if tuple(batch.relevance_labels.shape) != (B, TOTAL_EVENT_SLOTS, 1):
            raise TensorSchemaError(f"Relevance labels shape mismatch: {tuple(batch.relevance_labels.shape)}")
    if batch.relevance_mask is not None:
        if tuple(batch.relevance_mask.shape) != (B, TOTAL_EVENT_SLOTS, 1):
            raise TensorSchemaError(f"Relevance mask shape mismatch: {tuple(batch.relevance_mask.shape)}")
        # Masked event slots cannot have valid relevance targets
        invalid_relevance = batch.relevance_mask & ~batch.event_mask.unsqueeze(-1)
        if invalid_relevance.any():
            raise TensorSchemaError("Relevance mask violates event mask: inactive slots marked as valid relevance.")

    if batch.novelty_labels is not None:
        if tuple(batch.novelty_labels.shape) != (B, TOTAL_EVENT_SLOTS, 1):
            raise TensorSchemaError(f"Novelty labels shape mismatch: {tuple(batch.novelty_labels.shape)}")
    if batch.novelty_mask is not None:
        if tuple(batch.novelty_mask.shape) != (B, TOTAL_EVENT_SLOTS, 1):
            raise TensorSchemaError(f"Novelty mask shape mismatch: {tuple(batch.novelty_mask.shape)}")
        # Masked event slots cannot have valid novelty targets
        invalid_novelty = batch.novelty_mask & ~batch.event_mask.unsqueeze(-1)
        if invalid_novelty.any():
            raise TensorSchemaError("Novelty mask violates event mask: inactive slots marked as valid novelty.")

    if batch.return_targets is not None:
        if tuple(batch.return_targets.shape) != (B, NUM_HORIZONS):
            raise TensorSchemaError(f"Return targets shape mismatch: {tuple(batch.return_targets.shape)}")
    if batch.return_mask is not None:
        if tuple(batch.return_mask.shape) != (B, NUM_HORIZONS):
            raise TensorSchemaError(f"Return mask shape mismatch: {tuple(batch.return_mask.shape)}")

    if batch.risk_targets is not None:
        if tuple(batch.risk_targets.shape) != (B, NUM_HORIZONS):
            raise TensorSchemaError(f"Risk targets shape mismatch: {tuple(batch.risk_targets.shape)}")
        # Risk target magnitude must be strictly non-negative
        if (batch.risk_targets < 0.0).any():
            raise TensorSchemaError("Risk targets must be strictly non-negative.")
    if batch.risk_mask is not None:
        if tuple(batch.risk_mask.shape) != (B, NUM_HORIZONS):
            raise TensorSchemaError(f"Risk mask shape mismatch: {tuple(batch.risk_mask.shape)}")

    # 3. Dynamic Event Age Index Validation
    event_age_idx = get_event_age_index(effective_schema)
    if effective_schema in (FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value, "v8.0-original", "v8.0"):
        if event_age_idx != 22:
            raise TensorSchemaError(f"Schema {effective_schema} expects Event Age at index 22, got {event_age_idx}")
    else:
        if event_age_idx != 12:
            raise TensorSchemaError(f"Schema {effective_schema} expects Event Age at index 12, got {event_age_idx}")

    # Event age must be non-negative for valid event slots
    active_ages = batch.event_metadata[:, :, event_age_idx][batch.event_mask]
    if (active_ages < 0.0).any():
        raise TensorSchemaError(f"Negative event age detected at index {event_age_idx} in active event slots.")

    # 4. Semantic field boundaries for Baseline V8.0 (on raw features)
    if not is_normalized and effective_schema in (FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value, "v8.0-original", "v8.0"):
        # P07 Close Location in [0.0, 1.0]
        close_loc = batch.price[:, :, 6]
        if (close_loc < -1e-4).any() or (close_loc > 1.0 + 1e-4).any():
            raise TensorSchemaError("Price Close Location (P07) exceeds bounded interval [0.0, 1.0].")

        # T14 Bollinger Bandwidth >= 0
        bb_width = batch.technical[:, :, 13]
        if (bb_width < -1e-4).any():
            raise TensorSchemaError("Technical Bollinger Bandwidth (T14) cannot be negative.")

        # T16 ADX14 in [0.0, 1.0]
        adx = batch.technical[:, :, 15]
        if (adx < -1e-4).any() or (adx > 1.0 + 1e-4).any():
            raise TensorSchemaError("Technical ADX14 (T16) exceeds bounded interval [0.0, 1.0].")

        # Stock Context Percentiles C11, C12, C14 in [0.0, 1.0]
        for col_idx, col_name in [(10, "volatility_percentile"), (11, "liquidity_percentile"), (13, "industry_market_cap_percentile")]:
            col_val = batch.stock_context[:, col_idx]
            if (col_val < -1e-4).any() or (col_val > 1.0 + 1e-4).any():
                raise TensorSchemaError(f"Stock Context {col_name} exceeds percentile interval [0.0, 1.0].")

        # Position S01 in {0.0, 1.0}
        pos = batch.position[:, 0]
        if not torch.all((pos == 0.0) | (pos == 1.0)):
            raise TensorSchemaError("Position State target position (S01) must be binary 0.0 or 1.0.")

    # 5. Missing & Epistemic Tracking Masks Validation (P1-4)
    if batch.macro_missing_mask is not None:
        if tuple(batch.macro_missing_mask.shape) != (B, MARKET_WINDOW_LENGTH, MACRO_DIM):
            raise TensorSchemaError(f"Macro missing mask shape mismatch: expected {(B, MARKET_WINDOW_LENGTH, MACRO_DIM)}, got {tuple(batch.macro_missing_mask.shape)}")
        if batch.macro_missing_mask.dtype != torch.bool:
            raise TensorSchemaError("Macro missing mask must have dtype torch.bool")

    if batch.institutional_missing_mask is not None:
        if tuple(batch.institutional_missing_mask.shape) != (B, MARKET_WINDOW_LENGTH, INSTITUTIONAL_DIM):
            raise TensorSchemaError(f"Institutional missing mask shape mismatch: expected {(B, MARKET_WINDOW_LENGTH, INSTITUTIONAL_DIM)}, got {tuple(batch.institutional_missing_mask.shape)}")
        if batch.institutional_missing_mask.dtype != torch.bool:
            raise TensorSchemaError("Institutional missing mask must have dtype torch.bool")

    if batch.stock_context_missing_mask is not None:
        if tuple(batch.stock_context_missing_mask.shape) != (B, STOCK_CONTEXT_DIM):
            raise TensorSchemaError(f"Stock context missing mask shape mismatch: expected {(B, STOCK_CONTEXT_DIM)}, got {tuple(batch.stock_context_missing_mask.shape)}")
        if batch.stock_context_missing_mask.dtype != torch.bool:
            raise TensorSchemaError("Stock context missing mask must have dtype torch.bool")

    if batch.event_metadata_missing_mask is not None:
        if tuple(batch.event_metadata_missing_mask.shape) != (B, TOTAL_EVENT_SLOTS, EVENT_METADATA_DIM):
            raise TensorSchemaError(f"Event metadata missing mask shape mismatch: expected {(B, TOTAL_EVENT_SLOTS, EVENT_METADATA_DIM)}, got {tuple(batch.event_metadata_missing_mask.shape)}")
        if batch.event_metadata_missing_mask.dtype != torch.bool:
            raise TensorSchemaError("Event metadata missing mask must have dtype torch.bool")

    return True

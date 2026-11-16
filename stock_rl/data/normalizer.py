"""
Point-in-Time Feature Normalization Pipeline V8.0 / V8.3
Strictly fits normalization statistics on Training Split ONLY.
Applies frozen statistics to Validation and Test splits to prevent lookahead leakage.
Supports both Baseline 'v8.0-original' and Ablation 'v8.1-candidate'.
"""

from typing import Dict, Any, List, Optional, Union
import copy
import numpy as np
import torch

from stock_rl.config.schema import (
    CURRENT_ACTIVE_SCHEMA_VERSION,
    FeatureSchemaVersion,
    validate_schema_version,
)
from stock_rl.data.synthetic_data import MultimodalBatch


class FeatureNormalizer:
    """
    Stateful point-in-time feature normalizer.
    Guarantees that normalization statistics (mean, std) are computed exclusively
    on training data and frozen thereafter.
    """
    def __init__(self, schema_version: Optional[str] = None):
        self.schema_version = schema_version or CURRENT_ACTIVE_SCHEMA_VERSION
        validate_schema_version(self.schema_version)
        self.is_fitted: bool = False
        self.stats: Dict[str, Dict[str, torch.Tensor]] = {}

    def fit(
        self,
        batch_or_list: Union[MultimodalBatch, List[MultimodalBatch]],
        is_train_split: bool = True,
    ) -> "FeatureNormalizer":
        """
        Computes mean and std statistics exclusively on the Training Split.
        Raises ValueError if attempted on Validation or Test splits.
        """
        if not is_train_split:
            raise ValueError(
                "PIT Violation: FeatureNormalizer can ONLY be fitted on the Training Split! "
                "Fitting on Validation or Test splits leaks future distribution statistics."
            )
        if self.is_fitted:
            raise RuntimeError(
                "FeatureNormalizer is already fitted and frozen. "
                "Re-fitting a frozen normalizer is prohibited to prevent statistical leakage."
            )

        if isinstance(batch_or_list, list):
            # Concatenate list of batches
            price = torch.cat([b.price for b in batch_or_list], dim=0)
            tech = torch.cat([b.technical for b in batch_or_list], dim=0)
            inst = torch.cat([b.institutional for b in batch_or_list], dim=0)
            macro = torch.cat([b.macro for b in batch_or_list], dim=0)
            ctx = torch.cat([b.stock_context for b in batch_or_list], dim=0)
        else:
            price = batch_or_list.price
            tech = batch_or_list.technical
            inst = batch_or_list.institutional
            macro = batch_or_list.macro
            ctx = batch_or_list.stock_context

        # Compute statistics over batch and time dimensions: [dim]
        self.stats = {
            "price": {
                "mean": price.mean(dim=(0, 1), keepdim=True),
                "std": torch.clamp(price.std(dim=(0, 1), keepdim=True), min=1e-4),
            },
            "technical": {
                "mean": tech.mean(dim=(0, 1), keepdim=True),
                "std": torch.clamp(tech.std(dim=(0, 1), keepdim=True), min=1e-4),
            },
            "institutional": {
                "mean": inst.mean(dim=(0, 1), keepdim=True),
                "std": torch.clamp(inst.std(dim=(0, 1), keepdim=True), min=1e-4),
            },
            "macro": {
                "mean": macro.mean(dim=(0, 1), keepdim=True),
                "std": torch.clamp(macro.std(dim=(0, 1), keepdim=True), min=1e-4),
            },
            "stock_context": {
                "mean": ctx.mean(dim=0, keepdim=True),
                "std": torch.clamp(ctx.std(dim=0, keepdim=True), min=1e-4),
            },
        }

        self.is_fitted = True
        return self

    def transform(self, batch: MultimodalBatch) -> MultimodalBatch:
        """
        Transforms features in MultimodalBatch using the FROZEN training statistics.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "FeatureNormalizer must be fitted on training data before calling transform()!"
            )
        if batch.schema_version != self.schema_version:
            raise ValueError(
                f"Normalizer schema mismatch: normalizer was fitted for '{self.schema_version}', "
                f"but received batch with '{batch.schema_version}'."
            )

        norm_price = (batch.price - self.stats["price"]["mean"]) / self.stats["price"]["std"]
        norm_tech = (batch.technical - self.stats["technical"]["mean"]) / self.stats["technical"]["std"]
        norm_inst = (batch.institutional - self.stats["institutional"]["mean"]) / self.stats["institutional"]["std"]
        norm_macro = (batch.macro - self.stats["macro"]["mean"]) / self.stats["macro"]["std"]
        norm_ctx = (batch.stock_context - self.stats["stock_context"]["mean"]) / self.stats["stock_context"]["std"]

        # Deep copy other fields unchanged
        return MultimodalBatch(
            price=norm_price,
            technical=norm_tech,
            institutional=norm_inst,
            macro=norm_macro,
            stock_context=norm_ctx,
            recent_event_emb=batch.recent_event_emb.clone(),
            active_event_emb=batch.active_event_emb.clone(),
            event_metadata=batch.event_metadata.clone(),
            event_stock_rel=batch.event_stock_rel.clone(),
            event_mask=batch.event_mask.clone(),
            position=batch.position.clone(),
            relevance_labels=batch.relevance_labels.clone() if batch.relevance_labels is not None else None,
            novelty_labels=batch.novelty_labels.clone() if batch.novelty_labels is not None else None,
            return_targets=batch.return_targets.clone() if batch.return_targets is not None else None,
            risk_targets=batch.risk_targets.clone() if batch.risk_targets is not None else None,
            relevance_mask=batch.relevance_mask.clone() if batch.relevance_mask is not None else None,
            novelty_mask=batch.novelty_mask.clone() if batch.novelty_mask is not None else None,
            return_mask=batch.return_mask.clone() if batch.return_mask is not None else None,
            risk_mask=batch.risk_mask.clone() if batch.risk_mask is not None else None,
            schema_version=self.schema_version,
        )

    def fit_transform(
        self,
        train_batch: MultimodalBatch,
        is_train_split: bool = True
    ) -> MultimodalBatch:
        """Fits on training batch and returns the transformed batch."""
        self.fit(train_batch, is_train_split=is_train_split)
        return self.transform(train_batch)

    def state_dict(self) -> Dict[str, Any]:
        """Exports normalizer statistics."""
        if not self.is_fitted:
            raise RuntimeError("Cannot export state_dict of an unfitted normalizer.")
        return {
            "schema_version": self.schema_version,
            "is_fitted": self.is_fitted,
            "stats": self.stats,
        }

    def load_state_dict(self, state: Dict[str, Any]):
        """Loads frozen normalizer statistics."""
        saved_schema = state.get("schema_version")
        if saved_schema != self.schema_version:
            raise ValueError(
                f"Normalizer schema mismatch: state dict schema '{saved_schema}' != '{self.schema_version}'"
            )
        self.stats = state["stats"]
        self.is_fitted = state["is_fitted"]

    def save(self, filepath: str):
        torch.save(self.state_dict(), filepath)

    def load(self, filepath: str):
        state = torch.load(filepath, map_location="cpu")
        self.load_state_dict(state)

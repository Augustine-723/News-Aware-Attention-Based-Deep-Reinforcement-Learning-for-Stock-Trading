"""
Point-in-Time Feature Normalization Pipeline V8.0 / V8.3
Strictly fits normalization statistics on Training Split ONLY.
Applies frozen statistics to Validation and Test splits to prevent lookahead leakage.
Supports both Baseline 'v8.0-original' and Ablation 'v8.1-candidate'.
Integrates SplitManifest provenance verification per V8.3 Code Review 1 (P1-5).
"""

from dataclasses import dataclass
import hashlib
from typing import Dict, Any, List, Optional, Union
import copy
import numpy as np
import pandas as pd
import torch

from stock_rl.config.schema import (
    CURRENT_ACTIVE_SCHEMA_VERSION,
    FeatureSchemaVersion,
    validate_schema_version,
)
from stock_rl.data.synthetic_data import MultimodalBatch


@dataclass(frozen=True)
class SplitManifest:
    """
    Cryptographic and chronological manifest for dataset split boundaries (P1-5).
    Guarantees train/validation/test temporal isolation and provenance tracking.
    """
    dataset_id: str
    manifest_hash: str
    schema_version: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    @classmethod
    def create(
        cls,
        dataset_id: str,
        schema_version: str,
        train_start: Union[str, pd.Timestamp],
        train_end: Union[str, pd.Timestamp],
        val_start: Union[str, pd.Timestamp],
        val_end: Union[str, pd.Timestamp],
        test_start: Union[str, pd.Timestamp],
        test_end: Union[str, pd.Timestamp],
    ) -> "SplitManifest":
        t_start = pd.Timestamp(train_start)
        t_end = pd.Timestamp(train_end)
        v_start = pd.Timestamp(val_start)
        v_end = pd.Timestamp(val_end)
        te_start = pd.Timestamp(test_start)
        te_end = pd.Timestamp(test_end)

        if not (t_start <= t_end < v_start <= v_end < te_start <= te_end):
            raise ValueError(
                f"SplitManifest chronological violation: train ({t_start}..{t_end}), "
                f"val ({v_start}..{v_end}), test ({te_start}..{te_end}) must be strictly sequential."
            )

        content = f"{dataset_id}|{schema_version}|{t_start.isoformat()}|{t_end.isoformat()}|{v_start.isoformat()}|{v_end.isoformat()}|{te_start.isoformat()}|{te_end.isoformat()}"
        manifest_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        return cls(
            dataset_id=dataset_id,
            manifest_hash=manifest_hash,
            schema_version=schema_version,
            train_start=t_start,
            train_end=t_end,
            val_start=v_start,
            val_end=v_end,
            test_start=te_start,
            test_end=te_end,
        )

    def validate_sample_timestamps(
        self,
        timestamps: Union[pd.DatetimeIndex, List[pd.Timestamp]],
        split_name: str = "train",
    ) -> bool:
        """
        Validates that all sample timestamps fall strictly within the specified split range.
        Raises ValueError if any timestamp falls outside the split boundaries.
        """
        if split_name == "train":
            s_start, s_end = self.train_start, self.train_end
        elif split_name in ("val", "validation"):
            s_start, s_end = self.val_start, self.val_end
        elif split_name == "test":
            s_start, s_end = self.test_start, self.test_end
        else:
            raise ValueError(f"Unknown split name: {split_name}")

        ts_series = pd.DatetimeIndex(timestamps)
        if ts_series.tz is not None and s_start.tz is not None:
            ts_norm = ts_series.tz_convert(s_start.tz)
        elif ts_series.tz is not None and s_start.tz is None:
            ts_norm = ts_series.tz_localize(None)
        elif ts_series.tz is None and s_start.tz is not None:
            ts_norm = ts_series.tz_localize(s_start.tz)
        else:
            ts_norm = ts_series

        out_of_bounds = (ts_norm < s_start) | (ts_norm > s_end)
        if out_of_bounds.any():
            invalid_ts = ts_series[out_of_bounds]
            raise ValueError(
                f"Split provenance violation: {len(invalid_ts)} sample timestamps "
                f"fall outside '{split_name}' range [{s_start} .. {s_end}]. "
                f"First offending timestamp: {invalid_ts[0]}."
            )
        return True


class FeatureNormalizer:
    """
    Stateful point-in-time feature normalizer.
    Guarantees that normalization statistics (mean, std) are computed exclusively
    on training data and frozen thereafter.
    Enforces SplitManifest boundary provenance checks (P1-5).
    """
    def __init__(
        self,
        schema_version: Optional[str] = None,
        split_manifest: Optional[SplitManifest] = None,
    ):
        self.schema_version = schema_version or CURRENT_ACTIVE_SCHEMA_VERSION
        validate_schema_version(self.schema_version)
        self.split_manifest = split_manifest
        if self.split_manifest is not None and self.split_manifest.schema_version != self.schema_version:
            raise ValueError(
                f"SplitManifest schema mismatch: manifest configured for '{self.split_manifest.schema_version}', "
                f"but normalizer configured for '{self.schema_version}'."
            )
        self.is_fitted: bool = False
        self.stats: Dict[str, Dict[str, torch.Tensor]] = {}

    def fit(
        self,
        batch_or_list: Union[MultimodalBatch, List[MultimodalBatch]],
        is_train_split: bool = True,
        sample_timestamps: Optional[Union[pd.DatetimeIndex, List[pd.Timestamp]]] = None,
        split_manifest: Optional[SplitManifest] = None,
    ) -> "FeatureNormalizer":
        """
        Computes mean and std statistics exclusively on the Training Split.
        Raises ValueError if attempted on Validation or Test splits.
        If sample_timestamps and manifest are available, rejects fitting if any timestamps
        fall outside train_start..train_end even if is_train_split=True (P1-5).
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

        manifest = split_manifest or self.split_manifest
        if manifest is not None and sample_timestamps is not None:
            manifest.validate_sample_timestamps(sample_timestamps, split_name="train")

        if isinstance(batch_or_list, list):
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
        is_train_split: bool = True,
        sample_timestamps: Optional[Union[pd.DatetimeIndex, List[pd.Timestamp]]] = None,
        split_manifest: Optional[SplitManifest] = None,
    ) -> MultimodalBatch:
        """Fits on training batch and returns the transformed batch."""
        self.fit(train_batch, is_train_split=is_train_split, sample_timestamps=sample_timestamps, split_manifest=split_manifest)
        return self.transform(train_batch)

    def state_dict(self) -> Dict[str, Any]:
        """Exports normalizer statistics."""
        if not self.is_fitted:
            raise RuntimeError("Cannot export state_dict of an unfitted normalizer.")
        return {
            "schema_version": self.schema_version,
            "manifest_hash": self.split_manifest.manifest_hash if self.split_manifest else None,
            "stats": copy.deepcopy(self.stats),
        }

    def load_state_dict(self, state: Dict[str, Any]):
        """Loads and freezes normalizer statistics from state dict."""
        schema_ver = state.get("schema_version", CURRENT_ACTIVE_SCHEMA_VERSION)
        if schema_ver != self.schema_version:
            raise ValueError(
                f"Cannot load normalizer state: target normalizer is '{self.schema_version}', "
                f"but state dict has '{schema_ver}'."
            )
        self.stats = copy.deepcopy(state["stats"])
        self.is_fitted = True

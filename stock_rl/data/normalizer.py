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
    Explicitly distinguishes Manifest Definition Hash from Dataset Content Hash.
    """
    dataset_id: str
    manifest_hash: str                  # Kept as alias of manifest_definition_hash for backwards compatibility
    manifest_definition_hash: str       # Hash of split definition metadata
    schema_version: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    raw_dataset_content_hash: Optional[str] = None     # SHA-256 of raw underlying input data (P1-5)
    feature_snapshot_hash: Optional[str] = None        # SHA-256 of computed feature tensors (P1-5)
    normalizer_fitting_data_hash: Optional[str] = None # SHA-256 of training batch used for fit() (P1-5)
    feature_builder_version: str = "8.3.0"

    @staticmethod
    def compute_content_hash(data: Any) -> str:
        """Computes deterministic cryptographic SHA-256 hash across DataFrames, numpy arrays, torch tensors, or dicts."""
        hasher = hashlib.sha256()
        if isinstance(data, pd.DataFrame):
            hasher.update(data.to_csv(index=True).encode("utf-8"))
        elif isinstance(data, np.ndarray):
            hasher.update(data.tobytes())
        elif isinstance(data, torch.Tensor):
            hasher.update(data.detach().cpu().numpy().tobytes())
        elif isinstance(data, dict):
            for k in sorted(data.keys()):
                hasher.update(str(k).encode("utf-8"))
                hasher.update(SplitManifest.compute_content_hash(data[k]).encode("utf-8"))
        elif isinstance(data, (list, tuple)):
            for item in data:
                hasher.update(SplitManifest.compute_content_hash(item).encode("utf-8"))
        elif isinstance(data, (str, int, float, bool)):
            hasher.update(str(data).encode("utf-8"))
        elif data is None:
            hasher.update(b"__NONE__")
        else:
            hasher.update(repr(data).encode("utf-8"))
        return hasher.hexdigest()

    @staticmethod
    def compute_batch_content_hash(batch: MultimodalBatch) -> str:
        """Computes cryptographic SHA-256 across all primary tensors in a MultimodalBatch."""
        hasher = hashlib.sha256()
        for tensor_name in [
            "price", "technical", "institutional", "macro",
            "stock_context", "recent_event_emb", "active_event_emb",
            "event_metadata", "event_stock_rel", "event_mask", "position"
        ]:
            t = getattr(batch, tensor_name, None)
            if t is not None:
                hasher.update(tensor_name.encode("utf-8"))
                hasher.update(t.detach().cpu().numpy().tobytes())
        return hasher.hexdigest()

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
        raw_dataset_content_hash: Optional[str] = None,
        feature_snapshot_hash: Optional[str] = None,
        normalizer_fitting_data_hash: Optional[str] = None,
        feature_builder_version: str = "8.3.0",
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

        content = (
            f"{dataset_id}|{schema_version}|{t_start.isoformat()}|{t_end.isoformat()}|"
            f"{v_start.isoformat()}|{v_end.isoformat()}|{te_start.isoformat()}|{te_end.isoformat()}"
        )
        manifest_def_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        return cls(
            dataset_id=dataset_id,
            manifest_hash=manifest_def_hash,
            manifest_definition_hash=manifest_def_hash,
            schema_version=schema_version,
            train_start=t_start,
            train_end=t_end,
            val_start=v_start,
            val_end=v_end,
            test_start=te_start,
            test_end=te_end,
            raw_dataset_content_hash=raw_dataset_content_hash,
            feature_snapshot_hash=feature_snapshot_hash,
            normalizer_fitting_data_hash=normalizer_fitting_data_hash,
            feature_builder_version=feature_builder_version,
        )

    def with_content_hashes(
        self,
        raw_dataset_content_hash: Optional[str] = None,
        feature_snapshot_hash: Optional[str] = None,
        normalizer_fitting_data_hash: Optional[str] = None,
    ) -> "SplitManifest":
        """Returns an updated immutable SplitManifest with actual content hashes attached."""
        return SplitManifest(
            dataset_id=self.dataset_id,
            manifest_hash=self.manifest_hash,
            manifest_definition_hash=self.manifest_definition_hash,
            schema_version=self.schema_version,
            train_start=self.train_start,
            train_end=self.train_end,
            val_start=self.val_start,
            val_end=self.val_end,
            test_start=self.test_start,
            test_end=self.test_end,
            raw_dataset_content_hash=raw_dataset_content_hash or self.raw_dataset_content_hash,
            feature_snapshot_hash=feature_snapshot_hash or self.feature_snapshot_hash,
            normalizer_fitting_data_hash=normalizer_fitting_data_hash or self.normalizer_fitting_data_hash,
            feature_builder_version=self.feature_builder_version,
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
    Enforces SplitManifest boundary provenance checks and timestamp alignment (P1-3, P1-5).
    """
    def __init__(
        self,
        schema_version: Optional[str] = None,
        split_manifest: Optional[SplitManifest] = None,
        mode: str = "formal",
    ):
        self.schema_version = schema_version or CURRENT_ACTIVE_SCHEMA_VERSION
        validate_schema_version(self.schema_version)
        self.split_manifest = split_manifest
        self.mode = mode
        if self.split_manifest is not None and self.split_manifest.schema_version != self.schema_version:
            raise ValueError(
                f"SplitManifest schema mismatch: manifest configured for '{self.split_manifest.schema_version}', "
                f"but normalizer configured for '{self.schema_version}'."
            )
        self.is_fitted: bool = False
        self.stats: Dict[str, Dict[str, torch.Tensor]] = {}
        self.fitted_dataset_id: Optional[str] = None
        self.fitted_manifest_hash: Optional[str] = None
        self.fitted_train_window: Optional[tuple] = None
        self.fitted_fitting_data_hash: Optional[str] = None
        self.fitted_raw_content_hash: Optional[str] = None
        self.fitted_feature_snapshot_hash: Optional[str] = None
        self.feature_builder_version: str = "8.3.0"

    def fit(
        self,
        batch_or_list: Union[MultimodalBatch, List[MultimodalBatch]],
        is_train_split: bool = True,
        sample_timestamps: Optional[Union[pd.DatetimeIndex, List[pd.Timestamp]]] = None,
        split_manifest: Optional[SplitManifest] = None,
        mode: Optional[str] = None,
    ) -> "FeatureNormalizer":
        """
        Computes mean and std statistics exclusively on the Training Split.
        Raises ValueError if attempted on Validation or Test splits.

        Formal Mode (mode='formal', default):
        Mandates all five provenance items (P1-3):
        1. SplitManifest
        2. sample_timestamps
        3. dataset_id / hash
        4. schema_version alignment
        5. train_fitting_window
        Also verifies len(sample_timestamps) == batch.price.shape[0].
        Missing any item strictly rejects fitting.

        Isolated Test Mode (mode='test' or 'synthetic'):
        Permits synthetic smoke tests without manifest, but validates if provided.
        """
        effective_mode = mode or self.mode

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

        # Determine batch count B
        if isinstance(batch_or_list, list):
            b_count = sum(b.price.shape[0] for b in batch_or_list)
        else:
            b_count = batch_or_list.price.shape[0]

        manifest = split_manifest or self.split_manifest

        if effective_mode == "formal":
            if manifest is None:
                raise ValueError(
                    "Formal Normalizer Provenance Error (P1-3): SplitManifest is mandatory in formal mode. "
                    "Cannot fit FeatureNormalizer without an audited SplitManifest."
                )
            if sample_timestamps is None:
                raise ValueError(
                    "Formal Normalizer Provenance Error (P1-3): sample_timestamps is mandatory in formal mode. "
                    "Cannot fit FeatureNormalizer without sample decision timestamps."
                )
            if not manifest.dataset_id or not manifest.manifest_hash:
                raise ValueError(
                    "Formal Normalizer Provenance Error (P1-3): SplitManifest must contain non-empty "
                    "dataset_id and manifest_hash."
                )
            if manifest.schema_version != self.schema_version:
                raise ValueError(
                    f"Formal Normalizer Provenance Error (P1-3): SplitManifest schema_version '{manifest.schema_version}' "
                    f"does not match normalizer schema_version '{self.schema_version}'."
                )
            if manifest.train_start is None or manifest.train_end is None:
                raise ValueError(
                    "Formal Normalizer Provenance Error (P1-3): SplitManifest must contain valid train_start "
                    "and train_end fitting window boundaries."
                )
            if len(sample_timestamps) != b_count:
                raise ValueError(
                    f"Formal Normalizer Provenance Error (P1-3): Length of sample_timestamps ({len(sample_timestamps)}) "
                    f"does not match total batch size ({b_count})."
                )
            manifest.validate_sample_timestamps(sample_timestamps, split_name="train")
            self.fitted_dataset_id = manifest.dataset_id
            self.fitted_manifest_hash = manifest.manifest_hash
            self.fitted_train_window = (manifest.train_start, manifest.train_end)
        elif effective_mode in ("synthetic", "test"):
            # Isolated test/synthetic mode
            if manifest is not None and sample_timestamps is not None:
                if len(sample_timestamps) != b_count:
                    raise ValueError(
                        f"Normalizer timestamp length mismatch: len(sample_timestamps) ({len(sample_timestamps)}) != {b_count}"
                    )
                manifest.validate_sample_timestamps(sample_timestamps, split_name="train")
            if manifest is not None:
                self.fitted_dataset_id = manifest.dataset_id
                self.fitted_manifest_hash = manifest.manifest_hash
                self.fitted_train_window = (manifest.train_start, manifest.train_end)
        else:
            raise ValueError(
                f"Unknown FeatureNormalizer mode: {effective_mode}. Expected 'formal', 'synthetic', or 'test'."
            )

        # P1-5: Compute content hash of fitting data
        if isinstance(batch_or_list, list):
            fitting_hash = SplitManifest.compute_content_hash([SplitManifest.compute_batch_content_hash(b) for b in batch_or_list])
        else:
            fitting_hash = SplitManifest.compute_batch_content_hash(batch_or_list)
        self.fitted_fitting_data_hash = fitting_hash

        if manifest is not None and manifest.normalizer_fitting_data_hash is not None:
            if self.fitted_fitting_data_hash != manifest.normalizer_fitting_data_hash:
                raise ValueError(
                    f"Dataset Content Hash Mismatch (P1-5): Normalizer fitting data hash '{self.fitted_fitting_data_hash}' "
                    f"does not match SplitManifest contract normalizer_fitting_data_hash '{manifest.normalizer_fitting_data_hash}'. "
                    "Underlying training batch data has been modified!"
                )
        if manifest is not None:
            self.fitted_raw_content_hash = manifest.raw_dataset_content_hash
            self.fitted_feature_snapshot_hash = manifest.feature_snapshot_hash
            self.feature_builder_version = manifest.feature_builder_version

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
        b_dim = ctx.shape[0]
        ctx_std = ctx.std(dim=0, keepdim=True, correction=0 if b_dim == 1 else 1)
        self.stats = {
            "price": {
                "mean": price.mean(dim=(0, 1), keepdim=True),
                "std": torch.clamp(torch.nan_to_num(price.std(dim=(0, 1), keepdim=True), nan=1.0), min=1e-4),
            },
            "technical": {
                "mean": tech.mean(dim=(0, 1), keepdim=True),
                "std": torch.clamp(torch.nan_to_num(tech.std(dim=(0, 1), keepdim=True), nan=1.0), min=1e-4),
            },
            "institutional": {
                "mean": inst.mean(dim=(0, 1), keepdim=True),
                "std": torch.clamp(torch.nan_to_num(inst.std(dim=(0, 1), keepdim=True), nan=1.0), min=1e-4),
            },
            "macro": {
                "mean": macro.mean(dim=(0, 1), keepdim=True),
                "std": torch.clamp(torch.nan_to_num(macro.std(dim=(0, 1), keepdim=True), nan=1.0), min=1e-4),
            },
            "stock_context": {
                "mean": ctx.mean(dim=0, keepdim=True),
                "std": torch.clamp(torch.nan_to_num(ctx_std, nan=1.0), min=1e-4),
            },
        }

        self.is_fitted = True
        return self

    def transform(self, batch: MultimodalBatch) -> MultimodalBatch:
        """
        Transforms features in MultimodalBatch using the FROZEN training statistics.
        Preserves missing masks without modification (P1-4).
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
            macro_missing_mask=batch.macro_missing_mask.clone() if batch.macro_missing_mask is not None else None,
            institutional_missing_mask=batch.institutional_missing_mask.clone() if batch.institutional_missing_mask is not None else None,
            stock_context_missing_mask=batch.stock_context_missing_mask.clone() if batch.stock_context_missing_mask is not None else None,
            event_metadata_missing_mask=batch.event_metadata_missing_mask.clone() if batch.event_metadata_missing_mask is not None else None,
            schema_version=self.schema_version,
        )

    def fit_transform(
        self,
        train_batch: MultimodalBatch,
        is_train_split: bool = True,
        sample_timestamps: Optional[Union[pd.DatetimeIndex, List[pd.Timestamp]]] = None,
        split_manifest: Optional[SplitManifest] = None,
        mode: Optional[str] = None,
    ) -> MultimodalBatch:
        """Fits on training batch and returns the transformed batch."""
        self.fit(
            train_batch,
            is_train_split=is_train_split,
            sample_timestamps=sample_timestamps,
            split_manifest=split_manifest,
            mode=mode,
        )
        return self.transform(train_batch)

    def state_dict(self) -> Dict[str, Any]:
        """Exports normalizer statistics and comprehensive provenance (P1-5)."""
        if not self.is_fitted:
            raise RuntimeError("Cannot export state_dict of an unfitted normalizer.")
        return {
            "schema_version": self.schema_version,
            "manifest_hash": self.fitted_manifest_hash or (self.split_manifest.manifest_hash if self.split_manifest else None),
            "manifest_definition_hash": self.fitted_manifest_hash or (self.split_manifest.manifest_definition_hash if self.split_manifest else None),
            "fitting_data_hash": self.fitted_fitting_data_hash,
            "raw_dataset_content_hash": self.fitted_raw_content_hash or (self.split_manifest.raw_dataset_content_hash if self.split_manifest else None),
            "feature_snapshot_hash": self.fitted_feature_snapshot_hash or (self.split_manifest.feature_snapshot_hash if self.split_manifest else None),
            "feature_builder_version": self.feature_builder_version,
            "dataset_id": self.fitted_dataset_id or (self.split_manifest.dataset_id if self.split_manifest else None),
            "stats": copy.deepcopy(self.stats),
            "mode": self.mode,
        }

    def load_state_dict(
        self,
        state: Dict[str, Any],
        expected_manifest: Optional[SplitManifest] = None,
    ):
        """Loads and freezes normalizer statistics from state dict with contract verification (P1-5)."""
        schema_ver = state.get("schema_version", CURRENT_ACTIVE_SCHEMA_VERSION)
        if schema_ver != self.schema_version:
            raise ValueError(
                f"Cannot load normalizer state: target normalizer is '{self.schema_version}', "
                f"but state dict has '{schema_ver}'."
            )
        if expected_manifest is not None:
            state_def_hash = state.get("manifest_definition_hash") or state.get("manifest_hash")
            if state_def_hash and state_def_hash != expected_manifest.manifest_definition_hash:
                raise ValueError(
                    f"Manifest Definition Hash Mismatch (P1-5): State has '{state_def_hash}', "
                    f"expected '{expected_manifest.manifest_definition_hash}'."
                )
            if expected_manifest.normalizer_fitting_data_hash and state.get("fitting_data_hash"):
                if state["fitting_data_hash"] != expected_manifest.normalizer_fitting_data_hash:
                    raise ValueError(
                        f"Fitting Data Hash Mismatch (P1-5): State has '{state.get('fitting_data_hash')}', "
                        f"expected '{expected_manifest.normalizer_fitting_data_hash}'."
                    )
        self.stats = copy.deepcopy(state["stats"])
        self.fitted_dataset_id = state.get("dataset_id")
        self.fitted_manifest_hash = state.get("manifest_hash")
        self.fitted_fitting_data_hash = state.get("fitting_data_hash")
        self.fitted_raw_content_hash = state.get("raw_dataset_content_hash")
        self.fitted_feature_snapshot_hash = state.get("feature_snapshot_hash")
        self.feature_builder_version = state.get("feature_builder_version", "8.3.0")
        self.is_fitted = True

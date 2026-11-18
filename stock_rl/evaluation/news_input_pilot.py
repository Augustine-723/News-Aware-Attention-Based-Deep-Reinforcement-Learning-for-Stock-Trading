"""Smoke-test M0–M3 inputs through one frozen V8.22 checkpoint."""

import argparse
import hashlib
import io
import json
from pathlib import Path

import torch

from stock_rl.config.schema import ACTION_CASH, CURRENT_ACTIVE_SCHEMA_VERSION, get_event_age_index
from stock_rl.data.normalizer import FeatureNormalizer
from stock_rl.data.tensor_validator import validate_multimodal_batch
from stock_rl.evaluation.paired_observation import build_report
from stock_rl.models.full_model import StockRLFullModel


def mask_event_inputs(batch, method):
    """Return an observation with only the selected method's event inputs."""
    if method not in ("M0", "M1", "M2", "M3"):
        raise ValueError(f"unsupported event input method: {method}")
    masked = batch.clone()
    if method == "M3":
        return masked
    masked.recent_event_emb.zero_()
    masked.active_event_emb.zero_()
    if method == "M2":
        return masked

    age_index = get_event_age_index(batch.schema_version)
    masked.event_metadata.zero_()
    masked.event_stock_rel.zero_()
    if masked.event_metadata_missing_mask is not None:
        masked.event_metadata_missing_mask.fill_(True)
    if method == "M1":
        masked.event_metadata[..., age_index] = batch.event_metadata[..., age_index]
        if masked.event_metadata_missing_mask is not None:
            masked.event_metadata_missing_mask[..., age_index] = batch.event_metadata_missing_mask[..., age_index]
    else:
        masked.event_mask.zero_()
        for key in ("relevance_labels", "novelty_labels", "relevance_mask", "novelty_mask"):
            field = getattr(masked, key)
            if field is not None:
                field.zero_()
    return masked


def run_checkpoint(checkpoint_path):
    raw = Path(checkpoint_path).read_bytes()
    state = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if state["schema_version"] != CURRENT_ACTIVE_SCHEMA_VERSION:
        raise ValueError("checkpoint schema does not match active snapshot schema")
    if state["protocol"]["protocol"] != "v8.22-pilot":
        raise ValueError("expected a V8.22 M0 checkpoint")
    normalizer = FeatureNormalizer(schema_version=state["schema_version"])
    normalizer.load_state_dict(state["normalizer"])
    model = StockRLFullModel()
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    def q_values(batch):
        batch = normalizer.transform(batch)
        validate_multimodal_batch(batch, is_normalized=True)
        with torch.inference_mode():
            q, _, _ = model(batch)
        if not torch.isfinite(q).all():
            raise RuntimeError("non-finite checkpoint output")
        return q[0]

    def observe(empty, fixture):
        reference = q_values(empty)
        readings = {}
        for method in ("M0", "M1", "M2", "M3"):
            batch = mask_event_inputs(fixture, method)
            q = q_values(batch)
            if (method == "M0" or not fixture.event_mask.any()) and not torch.equal(reference, q):
                raise RuntimeError("equivalent model inputs gave different Q values")
            readings[method] = {
                "event_count": int(batch.event_mask.sum()),
                "q_values_cash_long": q.tolist(),
                "max_abs_q_delta_from_m0": (q - reference).abs().max().item(),
                "greedy_action": "cash" if int(q.argmax()) == ACTION_CASH else "long",
            }
        return readings

    report = build_report(observe=observe)
    report["purpose"] = "FROZEN_M0_CHECKPOINT_SYNTHETIC_INPUT_SMOKE"
    report["checkpoint"] = {
        "seed": state["seed"], "selected_epoch": state["selected_epoch"],
        "sha256": hashlib.sha256(raw).hexdigest(),
        "fitting_data_hash": normalizer.fitted_fitting_data_hash,
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(run_checkpoint(args.checkpoint), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

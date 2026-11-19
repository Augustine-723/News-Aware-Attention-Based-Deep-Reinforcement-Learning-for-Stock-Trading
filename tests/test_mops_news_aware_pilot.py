import json
from pathlib import Path
import numpy as np
import pytest
import torch

from stock_rl.evaluation.mops_news_aware_pilot import (
    MOPS_JSON,
    RESULTS_JSON,
    load_mops_disclosures,
    build_mops_event_ledgers,
    CONSTANT_SENTENCE,
)
from stock_rl.evaluation.provenance import load_trusted_pretrained_embedding_engine
from stock_rl.evaluation.news_input_pilot import mask_event_inputs
from stock_rl.data.snapshot_builder import HistoricalSnapshotBuilder
from stock_rl.evaluation.market_drl_pilot import load_real_market_feeds, STOCK_METADATA
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def test_load_mops_disclosures_count_and_keys():
    disclosures = load_mops_disclosures(MOPS_JSON)
    assert len(disclosures) == 404
    for d in disclosures:
        assert "stock_id" in d
        assert "filing_datetime" in d
        assert "effective_decision_day" in d
        assert "headline" in d


def test_mops_event_ledgers_and_bge_embeddings():
    engine = load_trusted_pretrained_embedding_engine()
    if engine is None:
        pytest.skip("BGE snapshot not available locally")

    disclosures = load_mops_disclosures(MOPS_JSON)[:10]
    events_real, events_const = build_mops_event_ledgers(disclosures, engine)

    for stock in ("2330", "2603", "2881"):
        assert stock in events_real
        assert stock in events_const

    # Check embedding shapes and normalization
    all_events = [e for group in events_real.values() for e in group]
    if all_events:
        emb = all_events[0]["text_embedding"]
        assert emb.shape == (128,)
        assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-5)


def test_treatment_groups_masking_invariants():
    prices, inst, macro, foreign, context, _ = load_real_market_feeds("2330", mode="estimated")
    builder = HistoricalSnapshotBuilder(
        prices, inst, macro, foreign, stock_context_records=context,
        event_records=[], shares_outstanding=STOCK_METADATA["2330"]["shares_outstanding"],
        mode="estimated",
    )
    snap = builder.build_snapshot(pd.Timestamp("2024-02-15 20:00:00+08:00"))

    # Group A: M0
    snap_a = mask_event_inputs(snap, "M0")
    assert snap_a.event_mask.sum().item() == 0
    assert snap_a.recent_event_emb.norm().item() == 0.0

    # Group B: M1
    snap_b = mask_event_inputs(snap, "M1")
    assert snap_b.recent_event_emb.norm().item() == 0.0


def test_compiled_mops_news_aware_results_integrity():
    if not RESULTS_JSON.exists():
        pytest.skip("v8_47_mops_news_aware_experiment.json not found")

    data = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    assert "retrained_models" in data
    assert "action_audit" in data

    models = data["retrained_models"]
    for g in ("A", "B", "C", "D"):
        assert g in models
        assert "test" in models[g]
        assert "holdout" in models[g]
        for seed in ("42", "100", "2024"):
            assert seed in models[g]["test"]
            assert seed in models[g]["holdout"]

    audit = data["action_audit"]
    assert "C_vs_D" in audit
    cd_holdout = audit["C_vs_D"]["holdout_diffs"]
    # Check that in April holdout, C vs D has high exact match rate (>= 98%)
    assert cd_holdout["exact_match_rate"] >= 0.98

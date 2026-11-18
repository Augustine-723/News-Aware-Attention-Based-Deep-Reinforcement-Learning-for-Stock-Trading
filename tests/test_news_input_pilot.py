import torch

from stock_rl.config.schema import get_event_age_index
from stock_rl.evaluation.news_input_pilot import mask_event_inputs
from stock_rl.evaluation.paired_observation import MARKET_INPUTS, build_report


def test_event_input_ablation_retains_only_requested_observation_fields():
    def check(empty, full):
        m0, m1, m2, m3 = (mask_event_inputs(full, method) for method in ("M0", "M1", "M2", "M3"))
        assert all(torch.equal(getattr(empty, key), getattr(batch, key))
                   for key in MARKET_INPUTS for batch in (m0, m1, m2, m3))
        assert torch.equal(m0.event_mask, empty.event_mask)
        assert torch.equal(m0.event_metadata, empty.event_metadata)
        assert torch.equal(m0.event_stock_rel, empty.event_stock_rel)
        assert torch.equal(m0.event_metadata_missing_mask, empty.event_metadata_missing_mask)
        for batch in (m0, m1, m2):
            assert not batch.recent_event_emb.any()
            assert not batch.active_event_emb.any()
        for batch in (m1, m2, m3):
            assert torch.equal(batch.event_mask, full.event_mask)

        age = get_event_age_index(full.schema_version)
        expected = torch.zeros_like(full.event_metadata)
        expected[..., age] = full.event_metadata[..., age]
        assert torch.equal(m1.event_metadata, expected)
        assert not m1.event_stock_rel.any()
        assert torch.equal(m2.event_metadata, full.event_metadata)
        assert torch.equal(m2.event_stock_rel, full.event_stock_rel)
        assert torch.equal(m3.recent_event_emb, full.recent_event_emb)
        if full.event_mask.any():
            assert m2.event_stock_rel.any()
            assert m3.recent_event_emb.any()
        return True

    report = build_report(observe=check)
    assert [row["fixture_event_count"] for row in report["observations"]] == [0, 1, 3]
    assert all(row["frozen_inference"] for row in report["observations"])

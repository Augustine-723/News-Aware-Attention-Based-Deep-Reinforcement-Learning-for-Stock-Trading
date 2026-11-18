from pathlib import Path
import pytest

from stock_rl.evaluation.historical_import import read_snapshot
from stock_rl.evaluation.pilot_dataset import PilotSourceManifest, RAW_PILOT_EVENTS

FIXTURE = Path(__file__).parent / "fixtures/v8_19/synthetic.csv"
URL = "https://example.invalid/synthetic"
OBSERVED = "2026-11-18T12:00:00+08:00"


def test_fixture_manifest():
    entry, = read_snapshot(FIXTURE, URL, OBSERVED, fixture=True)
    assert entry.verification_status.value == "SYNTHETIC_FIXTURE"
    assert entry.verified_provider_available_at is None
    assert entry.source_locator == "CSV_RECORD[0]"
    assert entry.filing_datetime.isoformat() == "2024-01-02T20:15:01+08:00"


def test_missing_seconds(tmp_path):
    path = tmp_path / "sample.csv"
    path.write_bytes(FIXTURE.read_bytes().replace(b"201501", b"2015"))
    with pytest.raises(ValueError, match="exact filing time"):
        read_snapshot(path, URL, OBSERVED, fixture=True)


def test_self_asserted_receipt():
    event = dict(RAW_PILOT_EVENTS[0], event_id="external-candidate",
                 verification_status="SOURCE_EXISTENCE_VERIFIED",
                 verification_evidence={"verified_provider_available_at": OBSERVED,
                                        "is_formal_pit_validated": True})
    rec = PilotSourceManifest.generate_manifest([event])[event["event_id"]]
    assert rec.verified_provider_available_at is None
    assert not rec.is_formal_pit_validated
    assert rec.verification_status.value == "CURATED_UNVERIFIED"


def test_mismatch_rejected_before_parse():
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        read_snapshot(FIXTURE, URL, OBSERVED, expected_hash="0" * 64)


def test_imported_entry_reaches_formal_gate():
    from stock_rl.evaluation.official_source_collector import execute_minimal_pipeline_integration

    entry, = read_snapshot(FIXTURE, URL, OBSERVED, fixture=True)
    result = execute_minimal_pipeline_integration(
        entry, target_stock_id=entry.stock_id, decision_date_str="2024-01-03",
        trading_calendar=["2024-01-03"], pit_mode="EVIDENCE_VERIFIED_ASOF",
    )
    assert not result["slot_ingested"]
    assert not result["is_formal_pit_validated"]


def test_dataset_18415_csv_accepts_unpadded_hour(tmp_path):
    path = tmp_path / "sample.csv"
    path.write_bytes(FIXTURE.read_bytes().replace(b"201501", b"70004"))
    entry, = read_snapshot(path, URL, OBSERVED, fixture=True)
    assert entry.filing_datetime.isoformat() == "2024-01-02T07:00:04+08:00"
    with pytest.raises(ValueError, match="exact filing time"):
        read_snapshot(path, URL, OBSERVED, fixture=True, licensed=True)

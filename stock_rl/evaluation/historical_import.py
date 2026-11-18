"""Inspect a privately held Dataset 18415 CSV or TWSE-schema JSON snapshot."""
import argparse
import csv
import hashlib
import io
import json
import os
import tempfile
import re
from urllib.parse import urlsplit
from pathlib import Path

import pandas as pd

from .official_source_collector import (
    EvidenceManifestEntry, OfficialSourceChannel, _parse_raw_record_publication_datetime,
)
from .pilot_dataset import DisclosurePITStatus, LicenseMetadata, SourceVerificationStatus
from .semantic_eval import compute_normalized_content_hash


def read_snapshot(path, source_url, observed, *, fixture=False, licensed=False, expected_hash=None):
    """Return existing manifest entries for the three-stock Q1 pilot."""
    url = urlsplit(source_url)
    if url.scheme != "https" or not url.hostname or url.username or url.password:
        raise ValueError("HTTPS source URL required")
    retrieved = pd.Timestamp(observed)
    if pd.isna(retrieved) or retrieved.tzinfo is None:
        raise ValueError("retrieval timestamp requires a timezone")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_hash is not None and digest != expected_hash:
        raise ValueError("raw snapshot SHA-256 mismatch")
    content = raw.decode("utf-8-sig")
    if path.suffix.lower() == ".csv":
        rows = list(csv.DictReader(io.StringIO(content, newline="")))
        kind = "CSV"
    elif path.suffix.lower() == ".json":
        rows = json.loads(content)
        kind = "JSON"
    else:
        raise ValueError("expected CSV or JSON")
    if not isinstance(rows, list):
        raise ValueError("expected an array of records")
    entries = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not all(isinstance(k, str) for k in row):
            raise ValueError(f"record {index}: expected named fields")
        fields = {key.strip(): value for key, value in row.items()}
        if not {"公司代號", "發言日期", "發言時間", "主旨", "說明"} <= fields.keys():
            raise ValueError(f"record {index}: missing disclosure fields")
        stock = str(fields["公司代號"]).strip()
        if stock not in {"2330", "2603", "2881"}:
            continue
        if not re.fullmatch(r"(?:[0-9]{6}|[0-9]{2}:[0-9]{2}:[0-9]{2})", str(fields["發言時間"])):
            raise ValueError(f"record {index}: exact filing time required")
        filing, ok = _parse_raw_record_publication_datetime(
            {key: fields[key] for key in ("發言日期", "發言時間")}
        )
        if not ok:
            raise ValueError(f"record {index}: invalid filing datetime")
        if filing.strftime("%Y-%m") not in {"2024-01", "2024-02", "2024-03"}:
            continue
        headline, body = fields["主旨"], fields["說明"]
        if retrieved < filing or not all(isinstance(s, str) and s.strip() for s in (headline, body)):
            raise ValueError(f"record {index}: invalid retrieval chronology or empty text")
        local_id = f"LOCAL-{digest}-{index}"
        entries.append(EvidenceManifestEntry(
            announcement_id=local_id, local_record_id=local_id, is_local_record_id=True,
            stock_id=stock, filing_datetime=filing, headline=headline, raw_content=body,
            source_url=source_url, query_parameters={}, retrieved_at=retrieved,
            raw_response_bytes=raw, raw_response_hash=digest,
            parsed_payload_hash=compute_normalized_content_hash(headline, body),
            verification_status=(SourceVerificationStatus.SYNTHETIC_FIXTURE if fixture
                                 else SourceVerificationStatus.MOPS_CANDIDATE_UNVERIFIED),
            pit_status=DisclosurePITStatus.UNVERIFIED_CANDIDATE,
            license_metadata=LicenseMetadata(license_name="Pending export-specific license review"),
            source_channel=(OfficialSourceChannel.COMMERCIAL_VENDOR_FEEDS if licensed
                            else OfficialSourceChannel.TAIWAN_GOV_OPEN_DATA),
            source_locator=f"{kind}_RECORD[{index}]", parser_version="v8.19-twse-export-1",
            raw_response_snapshot_file=str(path),
        ))
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--observed-at", required=True)
    parser.add_argument("--licensed-export", action="store_true", help="vendor export using TWSE field names")
    parser.add_argument("--expected-sha256", help="compare with an independently supplied export digest")
    parser.add_argument("--fixture", action="store_true")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--output", type=Path, help="private candidate JSONL, create only")
    parser.add_argument("--license-document", type=Path, help="local terms reviewed for private retention")
    action.add_argument("--dry-run", action="store_true", help="inspect only (default)")
    args = parser.parse_args()
    entries = read_snapshot(args.snapshot, args.source_url, args.observed_at, fixture=args.fixture,
                            licensed=args.licensed_export, expected_hash=args.expected_sha256)
    license_hash = None
    if args.output:
        if not args.fixture and not (args.license_document and args.expected_sha256):
            parser.error("private export requires --license-document and --expected-sha256")
        if args.license_document:
            terms = args.license_document.read_bytes()
            if not terms.strip():
                parser.error("empty license document")
            license_hash = hashlib.sha256(terms).hexdigest()
    private_rows = []
    for entry in entries:
        row = entry.to_dict()
        del row["headline"], row["raw_content"]
        row.update(observed_retrieved_at=row["retrieved_at"],
                   estimated_provider_available_at=(entry.filing_datetime + pd.Timedelta(minutes=5)).isoformat(),
                   verified_provider_available_at=None,
                   license_status=entry.license_metadata.verification_status.value)
        if args.output:
            private_rows.append(dict(row, headline=entry.headline, body_text=entry.raw_content,
                                     license_document_sha256=license_hash))
        print(json.dumps(row, ensure_ascii=False))
    if args.output:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=args.output.parent) as temp:
            for record in private_rows:
                temp.write(json.dumps(record, ensure_ascii=False) + "\n")
            temp.flush()
            os.link(temp.name, args.output)
    print(json.dumps({"matched_candidates": len(entries), "verified_historical": 0}))


if __name__ == "__main__":
    main()

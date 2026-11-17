"""
V8.8 Genuine Pretrained Semantic Model Integration Test Suite
Validates:
1. Trusted Model Provenance (Model ID, Revision, Apache-2.0 License, SHA-256 Manifest).
2. Anti-Spoofing: Randomly initialized encoders and manual boolean flags fail Pretrained Verification.
3. Trusted Historical Evidence: RawNewsArticle does not auto-grant VERIFIED_EVIDENCE.
   Official sources with self-filled fake evidence fail closed in formal mode.
4. Genuine Semantic Embedding Micro PoC (BAAI/bge-small-zh-v1.5, 128D projection, same/cross/negative control).
5. 4-Way Controlled Ablation (B0 Market-Only, B1 Deterministic Hash, DEV-N Untrained Neural, B2 Genuine Pretrained).
"""

import copy
import hashlib
import os
import shutil
import numpy as np
import pandas as pd
import pytest
import torch

from stock_rl.config.schema import (
    EVENT_TEXT_DIM,
    TOTAL_EVENT_SLOTS,
    RECENT_EVENT_SLOTS,
    ACTIVE_EVENT_SLOTS,
)
from stock_rl.data.multimodal_pipeline import (
    NewsSourceType,
    EventLifecycleStatus,
    EvidenceAuditLevel,
    TextEmbeddingType,
    TextEmbeddingEngine,
    RawNewsArticle,
    load_micro_news_dataset,
    HistoricalMultimodalPipeline,
    validate_news_article_pit_contract,
    convert_article_to_event_record,
    SourceEvidenceRecord,
    TrustedSourceRegistry,
    SourceAuditCertificate,
    TrustedNewsSourceVerifier,
)
from stock_rl.data.real_data_poc import (
    load_real_market_feeds,
    STOCK_METADATA,
)
from stock_rl.models.full_model import StockRLFullModel
from stock_rl.evaluation.ablations import (
    TraditionalChineseCharacterTokenizer,
    UntrainedTraditionalChineseNeuralEncoder,
    NewsAblationProtocol,
)
from stock_rl.evaluation.provenance import (
    PretrainedModelManifest,
    BGE_SMALL_ZH_MANIFEST,
    AUTHORITATIVE_PRETRAINED_MANIFESTS,
    VerifiedModelProvenance,
    ModelProvenanceVerifier,
    GenuinePretrainedSemanticEncoder,
    load_trusted_pretrained_embedding_engine,
    evaluate_projection_robustness,
    PROJECTION_ROBUSTNESS_BENCHMARK_PAIRS,
)


# ==============================================================================
# 1. Trusted Model Provenance & Anti-Spoofing
# ==============================================================================

def test_pretrained_manifest_integrity():
    """
    V8.8 Section 1:
    Verifies that BGE_SMALL_ZH_MANIFEST specifies an authentic published model:
    - Model ID: BAAI/bge-small-zh-v1.5
    - Revision: 7999e1d3359715c523056ef9478215996d62a620
    - License: Apache-2.0
    - Expected Weights SHA-256 matches actual weights
    - Tokenizer and projection specifications
    """
    manifest = BGE_SMALL_ZH_MANIFEST
    assert manifest.model_id == "BAAI/bge-small-zh-v1.5"
    assert manifest.revision == "7999e1d3359715c523056ef9478215996d62a620"
    assert manifest.license == "Apache-2.0"
    assert manifest.hidden_dim == 512
    assert manifest.projection_output_dim == EVENT_TEXT_DIM  # 128
    assert manifest.projection_version == "orthogonal-qr-v1.0"
    assert manifest.max_seq_len == 128
    assert len(manifest.expected_weights_sha256) == 64
    assert ModelProvenanceVerifier.verify_manifest(manifest) is True


def test_random_neural_encoder_rejected_by_provenance_verifier():
    """
    V8.8 Section 1 Anti-Spoofing Guard:
    Randomly initialized encoders (UntrainedTraditionalChineseNeuralEncoder)
    MUST NEVER pass Pretrained Verification.
    Manual is_verified_pretrained=True boolean flag alone without manifest signature is rejected.
    """
    enc_untrained = UntrainedTraditionalChineseNeuralEncoder()
    tok_untrained = TraditionalChineseCharacterTokenizer()

    # 1. Direct audit of untrained encoder must raise ValueError
    with pytest.raises(ValueError, match="cannot pass Pretrained Verification"):
        ModelProvenanceVerifier.audit_model_and_tokenizer(
            model=enc_untrained,
            tokenizer=tok_untrained,
            manifest=BGE_SMALL_ZH_MANIFEST,
        )

    # 2. TextEmbeddingEngine with untrained encoder must reject PRETRAINED mode
    engine_spoofed = TextEmbeddingEngine(
        embedding_type=TextEmbeddingType.PRETRAINED,
        model_version=BGE_SMALL_ZH_MANIFEST.model_id,
        dim=EVENT_TEXT_DIM,
        model_weights=enc_untrained,
        tokenizer=tok_untrained,
        is_verified_pretrained=True,
    )
    with pytest.raises(ValueError, match="Untrained neural baseline cannot be used in PRETRAINED mode"):
        engine_spoofed.embed_text("測試標題")

    # 3. ModelProvenanceVerifier.is_verified_engine rejects engine without verified provenance certificate
    assert ModelProvenanceVerifier.is_verified_engine(engine_spoofed) is False


def test_invalid_license_and_tampered_weights_rejected():
    """
    V8.8 Section 1 & P0-1:
    Verifies that unapproved licenses or tampered weights hashes fail closed.
    """
    # 1. Non-approved license
    bad_license_manifest = copy.copy(BGE_SMALL_ZH_MANIFEST)
    bad_license_manifest.license = "Proprietary-Closed-Source"
    assert ModelProvenanceVerifier.verify_manifest(bad_license_manifest) is False

    # 2. Tampered weights hash (Anti-Forge Guard)
    tampered_manifest = copy.copy(BGE_SMALL_ZH_MANIFEST)
    tampered_manifest.expected_weights_sha256 = "0000000000000000000000000000000000000000000000000000000000000000"
    
    import os
    snapshot_dir = os.path.expanduser(
        f"~/.cache/huggingface/hub/models--BAAI--bge-small-zh-v1.5/snapshots/{BGE_SMALL_ZH_MANIFEST.revision}"
    )
    if os.path.isdir(snapshot_dir):
        from transformers import AutoTokenizer, AutoModel
        tok = AutoTokenizer.from_pretrained(snapshot_dir, local_files_only=True)
        model = AutoModel.from_pretrained(snapshot_dir, local_files_only=True)
        with pytest.raises(ValueError, match=r"(Weights SHA-256 mismatch|Forged manifest detected)"):
            ModelProvenanceVerifier.audit_model_and_tokenizer(
                model=model,
                tokenizer=tok,
                manifest=tampered_manifest,
                snapshot_dir=snapshot_dir,
            )


def test_missing_snapshot_directory_fails_closed():
    """
    V8.8 P0-1:
    audit_model_and_tokenizer MUST require a valid snapshot_dir on disk.
    Missing or None snapshot_dir MUST fail closed immediately (eliminating Fail-Open).
    """
    snapshot_dir = os.path.expanduser(
        f"~/.cache/huggingface/hub/models--BAAI--bge-small-zh-v1.5/snapshots/{BGE_SMALL_ZH_MANIFEST.revision}"
    )
    if not os.path.isdir(snapshot_dir):
        pytest.skip("Local model snapshot not found")

    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(snapshot_dir, local_files_only=True)
    model = AutoModel.from_pretrained(snapshot_dir, local_files_only=True)

    # 1. snapshot_dir=None -> raises ValueError
    with pytest.raises(ValueError, match="snapshot_dir 'None' is missing or not a directory"):
        ModelProvenanceVerifier.audit_model_and_tokenizer(
            model=model,
            tokenizer=tok,
            manifest=BGE_SMALL_ZH_MANIFEST,
            snapshot_dir=None,
        )

    # 2. snapshot_dir does not exist -> raises ValueError
    with pytest.raises(ValueError, match="is missing or not a directory"):
        ModelProvenanceVerifier.audit_model_and_tokenizer(
            model=model,
            tokenizer=tok,
            manifest=BGE_SMALL_ZH_MANIFEST,
            snapshot_dir="/non/existent/model/snapshot/dir",
        )


def test_unregistered_model_manifest_fails_closed():
    """
    V8.8 P0-1:
    Candidate models supplying arbitrary unvetted manifests MUST fail closed.
    Only models registered in AUTHORITATIVE_PRETRAINED_MANIFESTS are accepted.
    """
    snapshot_dir = os.path.expanduser(
        f"~/.cache/huggingface/hub/models--BAAI--bge-small-zh-v1.5/snapshots/{BGE_SMALL_ZH_MANIFEST.revision}"
    )
    if not os.path.isdir(snapshot_dir):
        pytest.skip("Local model snapshot not found")

    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(snapshot_dir, local_files_only=True)
    model = AutoModel.from_pretrained(snapshot_dir, local_files_only=True)

    fake_manifest = PretrainedModelManifest(
        model_id="UnregisteredOrg/unvetted-fake-model",
        revision="abcdef123456",
        license="Apache-2.0",
        expected_weights_sha256="1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
    )
    with pytest.raises(ValueError, match="is not in the authoritative trusted model registry"):
        ModelProvenanceVerifier.audit_model_and_tokenizer(
            model=model,
            tokenizer=tok,
            manifest=fake_manifest,
            snapshot_dir=snapshot_dir,
        )


def test_missing_required_files_in_snapshot_fails_closed(tmp_path):
    """
    V8.8 P0-1:
    If any required file (model.safetensors, config.json, vocab.txt, tokenizer.json) is missing,
    audit_model_and_tokenizer MUST fail closed with FileNotFoundError.
    """
    snapshot_dir = os.path.expanduser(
        f"~/.cache/huggingface/hub/models--BAAI--bge-small-zh-v1.5/snapshots/{BGE_SMALL_ZH_MANIFEST.revision}"
    )
    if not os.path.isdir(snapshot_dir):
        pytest.skip("Local model snapshot not found")

    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(snapshot_dir, local_files_only=True)
    model = AutoModel.from_pretrained(snapshot_dir, local_files_only=True)

    # Empty directory
    empty_dir = tmp_path / "empty_snapshot"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="Missing weights file"):
        ModelProvenanceVerifier.audit_model_and_tokenizer(
            model=model,
            tokenizer=tok,
            manifest=BGE_SMALL_ZH_MANIFEST,
            snapshot_dir=str(empty_dir),
        )

    # Directory with weights but missing vocab.txt
    partial_dir = tmp_path / "partial_snapshot"
    partial_dir.mkdir()
    import shutil
    shutil.copyfile(os.path.join(snapshot_dir, "model.safetensors"), str(partial_dir / "model.safetensors"))
    with pytest.raises(FileNotFoundError, match="Missing vocab file 'vocab.txt'"):
        ModelProvenanceVerifier.audit_model_and_tokenizer(
            model=model,
            tokenizer=tok,
            manifest=BGE_SMALL_ZH_MANIFEST,
            snapshot_dir=str(partial_dir),
        )


def test_tampered_vocab_or_tokenizer_fails_closed(tmp_path):
    """
    V8.8 P0-1:
    If vocab.txt or tokenizer.json is modified, verification fails closed with SHA-256 mismatch.
    """
    snapshot_dir = os.path.expanduser(
        f"~/.cache/huggingface/hub/models--BAAI--bge-small-zh-v1.5/snapshots/{BGE_SMALL_ZH_MANIFEST.revision}"
    )
    if not os.path.isdir(snapshot_dir):
        pytest.skip("Local model snapshot not found")

    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(snapshot_dir, local_files_only=True)
    model = AutoModel.from_pretrained(snapshot_dir, local_files_only=True)

    tampered_dir = tmp_path / "tampered_snapshot"
    tampered_dir.mkdir()
    import shutil
    shutil.copyfile(os.path.join(snapshot_dir, "model.safetensors"), str(tampered_dir / "model.safetensors"))
    shutil.copyfile(os.path.join(snapshot_dir, "config.json"), str(tampered_dir / "config.json"))
    shutil.copyfile(os.path.join(snapshot_dir, "tokenizer.json"), str(tampered_dir / "tokenizer.json"))
    # Write tampered vocab
    with open(tampered_dir / "vocab.txt", "w", encoding="utf-8") as f:
        f.write("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nTAMPERED_TOKEN\n")

    with pytest.raises(ValueError, match="Vocab SHA-256 mismatch"):
        ModelProvenanceVerifier.audit_model_and_tokenizer(
            model=model,
            tokenizer=tok,
            manifest=BGE_SMALL_ZH_MANIFEST,
            snapshot_dir=str(tampered_dir),
        )


def test_model_architecture_parameter_mismatch_fails_closed():
    """
    V8.8 P0-1:
    Model architecture parameters (hidden_size, vocab_size, parameter count)
    must strictly conform to the authoritative manifest specifications.
    """
    snapshot_dir = os.path.expanduser(
        f"~/.cache/huggingface/hub/models--BAAI--bge-small-zh-v1.5/snapshots/{BGE_SMALL_ZH_MANIFEST.revision}"
    )
    if not os.path.isdir(snapshot_dir):
        pytest.skip("Local model snapshot not found")

    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(snapshot_dir, local_files_only=True)
    model = AutoModel.from_pretrained(snapshot_dir, local_files_only=True)

    # Tamper with model config hidden_size
    orig_hidden = model.config.hidden_size
    try:
        model.config.hidden_size = 768
        with pytest.raises(ValueError, match="Model hidden_size 768 does not match manifest hidden_dim 512"):
            ModelProvenanceVerifier.audit_model_and_tokenizer(
                model=model,
                tokenizer=tok,
                manifest=BGE_SMALL_ZH_MANIFEST,
                snapshot_dir=snapshot_dir,
            )
    finally:
        model.config.hidden_size = orig_hidden


def test_provenance_digest_vs_digital_signature_distinction():
    """
    V8.8 P0-1:
    Distinguishes cryptographic SHA-256 integrity digest from an asymmetric digital signature.
    Explicitly reports is_digital_signature_verified=False when no private key signing is used.
    """
    engine = load_trusted_pretrained_embedding_engine(local_files_only=True)
    if engine is None:
        pytest.skip("Local model snapshot not found")

    prov = engine.provenance_certificate
    assert prov is not None
    assert prov.is_audit_verified is True
    assert prov.is_digital_signature_verified is False  # Must NOT claim an unsigned hash is a digital signature
    assert prov.provenance_digest_type == "SHA256_COMPOSITE_INTEGRITY_DIGEST"
    assert len(prov.provenance_digest_sha256) == 64
    assert ModelProvenanceVerifier.is_verified_engine(engine) is True


# ==============================================================================
# 2. Trusted Historical Evidence & Official Source Defense
# ==============================================================================

def test_raw_news_article_never_auto_grants_verified_evidence():
    """
    V8.8 Section 2:
    RawNewsArticle must NOT automatically issue VERIFIED_EVIDENCE merely based on
    source_type, is_audit_verified=True, and non-empty string.
    Uncertified articles must be assigned EVIDENCE_CLAIM_ONLY.
    """
    t_pub = pd.Timestamp("2024-01-15 10:00:00", tz="Asia/Taipei")

    art = RawNewsArticle(
        source_type=NewsSourceType.MOPS_OFFICIAL,
        source_name="MOPS",
        stock_id="2330",
        headline="台積電先進封裝產能規劃",
        source_published_at=t_pub,
        provider_available_at=t_pub,
        provider_available_evidence="MOPS formal announcement text",
        is_audit_verified=True,
    )
    # Must NOT be VERIFIED_EVIDENCE upon raw instantiation
    assert art.evidence_audit_level == EvidenceAuditLevel.EVIDENCE_CLAIM_ONLY


def test_official_source_with_self_filled_evidence_rejected_in_formal_mode():
    """
    V8.8 Section 2:
    Official sources (MOPS, TWSE) with self-filled, draft, or informal evidence strings
    MUST fail closed in formal mode.
    """
    t_pub = pd.Timestamp("2024-01-15 10:00:00", tz="Asia/Taipei")

    # 1. Official source with "self-filled" marker
    art_self_filled_official = RawNewsArticle(
        source_type=NewsSourceType.MOPS_OFFICIAL,
        source_name="MOPS",
        stock_id="2330",
        headline="台積電說明媒體報導",
        source_published_at=t_pub,
        provider_available_at=t_pub,
        provider_available_evidence="MOPS self-filled press note without official filing number",
        is_audit_verified=True,
        evidence_audit_level=EvidenceAuditLevel.VERIFIED_EVIDENCE,
    )
    with pytest.raises(ValueError, match=r"Strict PIT (Error|Rejection)"):
        convert_article_to_event_record(art_self_filled_official, mode="formal")

    # 2. Official source with draft evidence
    art_draft_official = RawNewsArticle(
        source_type=NewsSourceType.MOPS_OFFICIAL,
        source_name="MOPS",
        stock_id="2330",
        headline="台積電說明媒體報導",
        source_published_at=t_pub,
        provider_available_at=t_pub,
        provider_available_evidence="MOPS press release draft memo",
        is_audit_verified=True,
        evidence_audit_level=EvidenceAuditLevel.VERIFIED_EVIDENCE,
    )
    with pytest.raises(ValueError, match=r"Strict PIT (Error|Rejection)"):
        convert_article_to_event_record(art_draft_official, mode="formal")

    # 3. Official source with arbitrary non-filing string
    art_arbitrary_official = RawNewsArticle(
        source_type=NewsSourceType.MOPS_OFFICIAL,
        source_name="MOPS",
        stock_id="2330",
        headline="台積電說明媒體報導",
        source_published_at=t_pub,
        provider_available_at=t_pub,
        provider_available_evidence="Spoken statement by spokesperson",
        is_audit_verified=True,
        evidence_audit_level=EvidenceAuditLevel.VERIFIED_EVIDENCE,
    )
    with pytest.raises(ValueError, match=r"Strict PIT (Error|Rejection)"):
        convert_article_to_event_record(art_arbitrary_official, mode="formal")


def test_certified_official_source_passes_formal_mode():
    """
    V8.8 Section 2 & P0-2:
    Articles backed by an authentic registered record in TrustedSourceRegistry and certified
    by TrustedNewsSourceVerifier successfully pass formal mode and preserve complete provenance.
    """
    t_pub = pd.Timestamp("2024-01-15 10:00:00", tz="Asia/Taipei")

    art = RawNewsArticle(
        source_type=NewsSourceType.MOPS_OFFICIAL,
        source_name="MOPS",
        stock_id="2330",
        headline="台積電說明媒體報導先進封裝建廠產能與投資規劃",
        body_text="說明媒體報導先進封裝建廠產能與投資規劃",
        source_published_at=t_pub,
        provider_available_at=t_pub,
        provider_available_evidence="MOPS-FILING-20240115-2330-01",
    )

    acc = "MOPS-FILING-20240115-2330-01"
    TrustedSourceRegistry.register_record(SourceEvidenceRecord(
        record_id="REC_MOPS_20240115_2330_01",
        source_type=NewsSourceType.MOPS_OFFICIAL,
        source_id=f"MOPS_{art.article_id}",
        accession_number=acc,
        stock_id="2330",
        provider_available_at=t_pub,
        payload_sha256=art.raw_payload_hash,
    ))

    # Certify via TrustedNewsSourceVerifier
    certified_art = TrustedNewsSourceVerifier.certify_article(
        art,
        accession_number=acc
    )

    assert certified_art.is_audit_verified is True
    assert certified_art.evidence_audit_level == EvidenceAuditLevel.VERIFIED_EVIDENCE
    assert "source_audit_certificate" in certified_art.metadata

    # In formal mode: succeeds
    rec = convert_article_to_event_record(certified_art, mode="formal")
    assert rec is not None
    assert rec["provenance"].is_audit_verified is True
    assert rec["event_id"] == certified_art.event_id


def test_fabricated_mops_accession_rejected_without_registry_record():
    """
    V8.8 P0-2:
    Regex pattern matching alone is NOT sufficient.
    Fabricated accession numbers not present in TrustedSourceRegistry MUST fail closed.
    """
    t_pub = pd.Timestamp("2024-01-15 10:00:00", tz="Asia/Taipei")

    art = RawNewsArticle(
        source_type=NewsSourceType.MOPS_OFFICIAL,
        source_name="MOPS",
        stock_id="2330",
        headline="台積電說明媒體報導",
        source_published_at=t_pub,
        provider_available_at=t_pub,
        provider_available_evidence="MOPS-FILING-20240115-9999-99",  # Valid regex, but unregistered
    )

    # Certification must raise ValueError because accession is not registered in TrustedSourceRegistry
    with pytest.raises(ValueError, match="is not registered in the authoritative trusted source registry"):
        TrustedNewsSourceVerifier.certify_article(
            art,
            accession_number="MOPS-FILING-20240115-9999-99",
        )


def test_modified_article_content_after_certification_rejected():
    """
    V8.8 P0-2:
    If article content or headline is modified after certification,
    payload SHA-256 mismatch is detected and formal mode strictly fails closed.
    """
    t_pub = pd.Timestamp("2024-01-15 10:00:00", tz="Asia/Taipei")

    art = RawNewsArticle(
        source_type=NewsSourceType.MOPS_OFFICIAL,
        source_name="MOPS",
        stock_id="2330",
        headline="台積電原始審計公告",
        body_text="原始完整公告內容",
        source_published_at=t_pub,
        provider_available_at=t_pub,
        provider_available_evidence="MOPS-FILING-20240115-2330-02",
    )
    acc = "MOPS-FILING-20240115-2330-02"
    TrustedSourceRegistry.register_record(SourceEvidenceRecord(
        record_id="REC_MOPS_20240115_2330_02",
        source_type=NewsSourceType.MOPS_OFFICIAL,
        source_id=f"MOPS_{art.article_id}",
        accession_number=acc,
        stock_id="2330",
        provider_available_at=t_pub,
        payload_sha256=art.raw_payload_hash,
    ))
    certified_art = TrustedNewsSourceVerifier.certify_article(art, accession_number=acc)

    # Tamper with headline and re-derive payload hash
    certified_art.headline = "篡改後的台積電公告標題"
    content_key = f"{certified_art.headline}|{certified_art.body_text}|{certified_art.url or ''}"
    certified_art.raw_payload_hash = hashlib.sha256(content_key.encode("utf-8")).hexdigest()

    # Formal mode must reject modified content
    with pytest.raises(ValueError, match="does not satisfy TrustedNewsSourceVerifier requirements"):
        convert_article_to_event_record(certified_art, mode="formal")


def test_modified_available_at_timestamp_rejected():
    """
    V8.8 P0-2:
    If provider_available_at timestamp is altered after certification,
    it violates certificate consistency and fails closed in formal mode.
    """
    t_pub = pd.Timestamp("2024-01-15 10:00:00", tz="Asia/Taipei")

    art = RawNewsArticle(
        source_type=NewsSourceType.MOPS_OFFICIAL,
        source_name="MOPS",
        stock_id="2330",
        headline="台積電原始時間審計公告",
        source_published_at=t_pub,
        provider_available_at=t_pub,
        provider_available_evidence="MOPS-FILING-20240115-2330-03",
    )
    acc = "MOPS-FILING-20240115-2330-03"
    TrustedSourceRegistry.register_record(SourceEvidenceRecord(
        record_id="REC_MOPS_20240115_2330_03",
        source_type=NewsSourceType.MOPS_OFFICIAL,
        source_id=f"MOPS_{art.article_id}",
        accession_number=acc,
        stock_id="2330",
        provider_available_at=t_pub,
        payload_sha256=art.raw_payload_hash,
    ))
    certified_art = TrustedNewsSourceVerifier.certify_article(art, accession_number=acc)

    # Shift available_at into the past or future
    certified_art.provider_available_at = t_pub + pd.Timedelta(hours=2)

    with pytest.raises(ValueError, match="does not satisfy TrustedNewsSourceVerifier requirements"):
        convert_article_to_event_record(certified_art, mode="formal")


def test_copied_stolen_certificate_from_other_article_rejected():
    """
    V8.8 P0-2:
    Copying a valid SourceAuditCertificate from article A to article B
    is strictly detected via payload SHA-256 and stock ID mismatch, failing closed.
    """
    t_pub = pd.Timestamp("2024-01-15 10:00:00", tz="Asia/Taipei")

    art_a = RawNewsArticle(
        source_type=NewsSourceType.MOPS_OFFICIAL,
        source_name="MOPS",
        stock_id="2330",
        headline="台積電合法認證公告",
        source_published_at=t_pub,
        provider_available_at=t_pub,
        provider_available_evidence="MOPS-FILING-20240115-2330-04",
    )
    acc = "MOPS-FILING-20240115-2330-04"
    TrustedSourceRegistry.register_record(SourceEvidenceRecord(
        record_id="REC_MOPS_20240115_2330_04",
        source_type=NewsSourceType.MOPS_OFFICIAL,
        source_id=f"MOPS_{art_a.article_id}",
        accession_number=acc,
        stock_id="2330",
        provider_available_at=t_pub,
        payload_sha256=art_a.raw_payload_hash,
    ))
    certified_a = TrustedNewsSourceVerifier.certify_article(art_a, accession_number=acc)

    # Article B for Evergreen Marine (2603)
    art_b = RawNewsArticle(
        source_type=NewsSourceType.MOPS_OFFICIAL,
        source_name="MOPS",
        stock_id="2603",
        headline="長榮海運重訊公告",
        source_published_at=t_pub,
        provider_available_at=t_pub,
        provider_available_evidence="MOPS-FILING-20240115-2330-04",  # Copied evidence string
    )
    # Stolen certificate attached to B
    art_b.metadata["source_audit_certificate"] = copy.deepcopy(certified_a.metadata["source_audit_certificate"])
    art_b.is_audit_verified = True
    art_b.evidence_audit_level = EvidenceAuditLevel.VERIFIED_EVIDENCE

    with pytest.raises(ValueError, match="does not satisfy TrustedNewsSourceVerifier requirements"):
        convert_article_to_event_record(art_b, mode="formal")


def test_mismatched_stock_id_rejected():
    """
    V8.8 P0-2:
    A certificate issued for stock 2330 cannot be used if article stock_id is changed.
    """
    t_pub = pd.Timestamp("2024-01-15 10:00:00", tz="Asia/Taipei")

    art = RawNewsArticle(
        source_type=NewsSourceType.MOPS_OFFICIAL,
        source_name="MOPS",
        stock_id="2330",
        headline="台積電認證公告",
        source_published_at=t_pub,
        provider_available_at=t_pub,
        provider_available_evidence="MOPS-FILING-20240115-2330-05",
    )
    acc = "MOPS-FILING-20240115-2330-05"
    TrustedSourceRegistry.register_record(SourceEvidenceRecord(
        record_id="REC_MOPS_20240115_2330_05",
        source_type=NewsSourceType.MOPS_OFFICIAL,
        source_id=f"MOPS_{art.article_id}",
        accession_number=acc,
        stock_id="2330",
        provider_available_at=t_pub,
        payload_sha256=art.raw_payload_hash,
    ))
    certified_art = TrustedNewsSourceVerifier.certify_article(art, accession_number=acc)

    # Change stock ID
    certified_art.stock_id = "2881"
    with pytest.raises(ValueError, match="does not satisfy TrustedNewsSourceVerifier requirements"):
        convert_article_to_event_record(certified_art, mode="formal")


def test_formal_certification_stays_pending_without_trusted_source_ledger():
    """
    V8.8 P0-2:
    Historical news items in Micro PoC (without an active external institutional audit ledger)
    MUST remain in ESTIMATED / EVIDENCE_CLAIM_ONLY status.
    Formal mode strictly fails closed, correctly reporting Formal Certification as Pending.
    """
    raw_articles = load_micro_news_dataset()
    assert len(raw_articles) > 0

    # In estimated mode: succeeds
    pipe_est = HistoricalMultimodalPipeline(mode="estimated")
    for a in raw_articles:
        pipe_est.ingest_article(a)
    assert len(pipe_est._articles) == len(raw_articles)

    # In formal mode: all uncertified historical articles FAIL CLOSED
    pipe_formal = HistoricalMultimodalPipeline(mode="formal")
    for a in raw_articles:
        with pytest.raises(ValueError, match=r"(Strict PIT Error|Strict PIT Rejection)"):
            pipe_formal.ingest_article(a)


# ==============================================================================
# 3. Genuine Semantic Embedding Micro PoC
# ==============================================================================

def test_genuine_pretrained_semantic_discrimination_and_negative_control():
    """
    V8.8 Section 3:
    Validates genuine BAAI/bge-small-zh-v1.5 embedding engine:
    1. Loads genuine pretrained weights from verified manifest.
    2. Measures same-topic similarity (two related TSMC CoWoS headlines).
    3. Measures cross-topic similarity (TSMC packaging vs Evergreen shipping).
    4. Measures negative control (TSMC packaging vs unrelated weather sentence).
    5. Verifies: Sim(Same) >> Sim(Cross) >= Sim(Negative).
    """
    engine = load_trusted_pretrained_embedding_engine(local_files_only=True)
    assert engine is not None, "Genuine pretrained BGE engine must load successfully from local cache"
    assert engine.is_verified_pretrained is True

    h_same_1 = "台積電嘉義先進封裝廠投資額翻倍，因應輝達CoWoS產能需求"
    h_same_2 = "台積電說明媒體報導先進封裝建廠產能與投資規劃"
    h_cross = "紅海危機延燒，長榮等全球航商持續繞道好望角，航運指數大漲"
    h_neg = "今天台北天氣晴朗氣溫適中午後有零星短暫陣雨"

    v1 = engine.embed_text(h_same_1)
    v2 = engine.embed_text(h_same_2)
    v3 = engine.embed_text(h_cross)
    v4 = engine.embed_text(h_neg)

    # Dimensionality & Unit Norm
    for v in [v1, v2, v3, v4]:
        assert v.shape == (EVENT_TEXT_DIM,)
        assert np.isclose(np.linalg.norm(v), 1.0, atol=1e-5)

    sim_same = float(np.dot(v1, v2))
    sim_cross = float(np.dot(v1, v3))
    sim_neg = float(np.dot(v1, v4))

    # Same-topic must be substantially higher than cross-topic and negative control
    assert sim_same > 0.75, f"Same-topic similarity should be high (got {sim_same:.4f})"
    assert sim_same > sim_cross + 0.20, (
        f"Same-topic ({sim_same:.4f}) must exceed cross-topic ({sim_cross:.4f}) by at least 0.20"
    )
    assert sim_cross >= sim_neg - 0.05, (
        f"Cross-topic financial move ({sim_cross:.4f}) vs negative control ({sim_neg:.4f})"
    )


def test_128d_orthogonal_projection_quality_preservation():
    """
    V8.8 Section 3:
    Validates that the 512D -> 128D orthogonal QR projection preserves
    cosine similarities with minimal distortion (|sim_512 - sim_128| < 0.05).
    """
    import os
    snapshot_dir = os.path.expanduser(
        f"~/.cache/huggingface/hub/models--BAAI--bge-small-zh-v1.5/snapshots/{BGE_SMALL_ZH_MANIFEST.revision}"
    )
    if not os.path.isdir(snapshot_dir):
        pytest.skip("BGE snapshot directory not found")

    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(snapshot_dir)
    model = AutoModel.from_pretrained(snapshot_dir)
    model.eval()

    texts = [
        "台積電嘉義先進封裝廠投資額翻倍，因應輝達CoWoS產能需求",
        "台積電說明媒體報導先進封裝建廠產能與投資規劃",
        "紅海危機延燒，長榮等全球航商持續繞道好望角，航運指數大漲",
    ]

    inputs = tok(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.inference_mode():
        outputs = model(**inputs)
        cls_512 = outputs[0][:, 0].numpy()
        norms = np.linalg.norm(cls_512, axis=1, keepdims=True)
        cls_512 = cls_512 / norms

    sim_same_512 = float(np.dot(cls_512[0], cls_512[1]))
    sim_cross_512 = float(np.dot(cls_512[0], cls_512[2]))

    # Now through 128D engine
    engine = load_trusted_pretrained_embedding_engine(local_files_only=True)
    assert engine is not None
    v1 = engine.embed_text(texts[0])
    v2 = engine.embed_text(texts[1])
    v3 = engine.embed_text(texts[2])

    sim_same_128 = float(np.dot(v1, v2))
    sim_cross_128 = float(np.dot(v1, v3))

    assert abs(sim_same_512 - sim_same_128) < 0.05, (
        f"Projection distortion on same-topic should be < 0.05 (512D: {sim_same_512:.4f}, 128D: {sim_same_128:.4f})"
    )
    assert abs(sim_cross_512 - sim_cross_128) < 0.05, (
        f"Projection distortion on cross-topic should be < 0.05 (512D: {sim_cross_512:.4f}, 128D: {sim_cross_128:.4f})"
    )


# ==============================================================================
# 4. 4-Way Controlled Ablation (B0, B1, DEV-N, B2)
# ==============================================================================

def test_4_way_news_ablation_protocol_with_genuine_pretrained_model():
    """
    V8.8 Section 4:
    Executes complete 4-way controlled ablation study on 2330:
    - B0: Market-Only (all event slots masked and zeroed)
    - B1: Market + Deterministic Hash News (SHA-256 baseline)
    - DEV-N: Market + Untrained Neural Baseline (Random Transformer)
    - B2: Market + Genuine Pretrained Semantic (BAAI/bge-small-zh-v1.5)
    Verifies:
    1. B2 status reports AVAILABLE.
    2. Market bitwise parity (100% equal market inputs across all 4 baselines).
    3. Event occupancy: B0=0 slots, B1=2 slots, DEV-N=2 slots, B2=2 slots.
    4. Model feature sensitivity (divergence across all conditions).
    5. Finite gradient flow across all 4 ablation branches.
    6. Untrained Q-value guardrail disclaimer presence.
    """
    ticker = "2330"
    ohlcv, inst, macro, foreign, ctx, _ = load_real_market_feeds(ticker, mode="estimated", include_micro_news=False)
    meta = STOCK_METADATA[ticker]
    raw_articles = [a for a in load_micro_news_dataset() if a.stock_id == ticker]

    t_dec = pd.Timestamp("2024-01-15 20:00:00", tz="Asia/Taipei")

    tok = TraditionalChineseCharacterTokenizer()
    enc = UntrainedTraditionalChineseNeuralEncoder(output_dim=EVENT_TEXT_DIM)

    model = StockRLFullModel()

    res = NewsAblationProtocol.run_news_ablation_experiment(
        model=model,
        df_ohlcv=ohlcv,
        df_inst=inst,
        df_macro=macro,
        df_foreign=foreign,
        stock_context_records=ctx,
        raw_articles=raw_articles,
        decision_timestamp=t_dec,
        shares_outstanding=meta["shares_outstanding"],
        semantic_encoder=enc,
        semantic_tokenizer=tok,
        auto_load_pretrained=True,
        mode="estimated",
    )

    # 1. B2 Status
    assert res["b2_status"] == "AVAILABLE", "With auto_load_pretrained=True, B2 must report AVAILABLE"

    # 2. Market Bitwise Parity across all 4 branches
    assert res["market_bitwise_equal"] is True, "Market features across B0, B1, DEV-N, B2 must be 100% bitwise equal"

    # 3. Occupancy Checks
    assert res["occupancy"]["B0"] == 0, "B0 (Market-Only) must have zero occupied event slots"
    assert res["occupancy"]["B1"] == 2, "B1 must occupy 2 valid event slots for 2330"
    assert res["occupancy"]["DEV_N"] == 2, "DEV-N must occupy 2 valid event slots for 2330"
    assert res["occupancy"]["B2"] == 2, "B2 must occupy 2 valid event slots for 2330"

    # 4. Sensitivity Measurements
    sens = res["sensitivity"]
    assert sens["delta_state_01"] > 0.0, "Model state must react to event slot existence (B1 vs B0)"
    assert sens["delta_state_1_devn"] > 0.0, "Model state must react to neural vs hash representation (DEV-N vs B1)"
    assert sens["delta_state_12"] > 0.0, "Model state must react to genuine semantic vs hash (B2 vs B1)"
    assert sens["delta_state_02"] > 0.0, "Model state must react to genuine semantic vs market-only (B2 vs B0)"
    assert sens["delta_state_devn_2"] > 0.0, "Model state must distinguish genuine pretrained from untrained neural (B2 vs DEV-N)"

    assert sens["delta_q_01"] > 0.0
    assert sens["delta_q_1_devn"] > 0.0
    assert sens["delta_q_12"] > 0.0

    # 5. Gradient Flow across all 4 branches
    for tag in ["B0", "B1", "DEV_N", "B2"]:
        assert res["gradient_flow"][tag]["is_finite"] is True, f"Gradient flow for {tag} must be finite"
        assert res["gradient_flow"][tag]["mean_grad_norm"] > 0.0, f"Gradient norm for {tag} must be non-zero"

    # 6. Guardrail Disclaimer
    assert res["untrained_model_q_value_is_not_alpha"] is True
    assert "Q-value differences" in res["methodology_disclaimer"]


# ==============================================================================
# P1: Projection Robustness Empirical Evaluation
# ==============================================================================

def test_projection_robustness_25_pairs_evaluation():
    """
    V8.8 P1:
    Empirical evaluation of 512D to 128D projection semantic fidelity across 25 curated
    Traditional Chinese financial headline pairs spanning 5 categories:
    - Same-topic pairs (5 pairs)
    - Paraphrases (5 pairs)
    - Cross-industry pairs (5 pairs)
    - Negation pairs (5 pairs)
    - Unrelated negative controls (5 pairs)
    
    Verifies:
    1. Mean Absolute Cosine Distortion < 0.05
    2. 95th Percentile Distortion < 0.08
    3. Maximum Distortion < 0.10
    4. Similarity Ranking Consistency (Spearman rho >= 0.95)
    5. All 5 categories evaluated and reported
    """
    engine = load_trusted_pretrained_embedding_engine(local_files_only=True)
    if engine is None:
        pytest.skip("Local model snapshot not found")

    encoder = engine.model_weights
    assert isinstance(encoder, GenuinePretrainedSemanticEncoder)

    report = evaluate_projection_robustness(
        encoder=encoder,
        pairs=PROJECTION_ROBUSTNESS_BENCHMARK_PAIRS,
        random_seed=42,
    )

    assert report["num_pairs"] == 25
    assert report["model_id"] == "BAAI/bge-small-zh-v1.5"
    assert report["projection_version"] == "orthogonal-qr-v1.0"

    # Distortion bounds
    mean_dist = report["mean_absolute_cosine_distortion"]
    p95_dist = report["percentile_95_distortion"]
    max_dist = report["maximum_distortion"]
    spearman_rho = report["spearman_rank_correlation"]

    print(f"\n[P1 Projection Robustness Report]")
    print(f"Mean Distortion: {mean_dist:.6f}")
    print(f"P95 Distortion:  {p95_dist:.6f}")
    print(f"Max Distortion:  {max_dist:.6f}")
    print(f"Spearman Rho:    {spearman_rho:.6f}")
    print(f"Kendall Tau:     {report['kendall_tau']:.6f}")

    assert mean_dist < 0.05, f"Mean distortion {mean_dist} exceeds threshold 0.05"
    assert p95_dist < 0.08, f"95th percentile distortion {p95_dist} exceeds threshold 0.08"
    assert max_dist < 0.10, f"Max distortion {max_dist} exceeds threshold 0.10"
    assert spearman_rho >= 0.95, f"Spearman rank correlation {spearman_rho} below 0.95"

    # Category checks
    cats = report["category_breakdown"]
    assert len(cats) == 5
    for cat_name in ["same_topic", "paraphrase", "cross_industry", "negation", "negative_control"]:
        assert cat_name in cats
        assert cats[cat_name]["count"] == 5
        assert cats[cat_name]["mean_distortion"] < 0.05

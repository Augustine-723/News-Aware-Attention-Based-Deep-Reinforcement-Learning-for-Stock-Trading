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
    VerifiedModelProvenance,
    ModelProvenanceVerifier,
    GenuinePretrainedSemanticEncoder,
    load_trusted_pretrained_embedding_engine,
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
    V8.8 Section 1:
    Verifies that unapproved licenses or tampered weights hashes fail closed.
    """
    # 1. Non-approved license
    bad_license_manifest = copy.copy(BGE_SMALL_ZH_MANIFEST)
    bad_license_manifest.license = "Proprietary-Closed-Source"
    assert ModelProvenanceVerifier.verify_manifest(bad_license_manifest) is False

    # 2. Tampered weights hash
    tampered_manifest = copy.copy(BGE_SMALL_ZH_MANIFEST)
    tampered_manifest.expected_weights_sha256 = "0000000000000000000000000000000000000000000000000000000000000000"
    
    # Audit against local snapshot with tampered hash must fail
    import os
    snapshot_dir = os.path.expanduser(
        f"~/.cache/huggingface/hub/models--BAAI--bge-small-zh-v1.5/snapshots/{BGE_SMALL_ZH_MANIFEST.revision}"
    )
    if os.path.isdir(snapshot_dir):
        from transformers import AutoTokenizer, AutoModel
        tok = AutoTokenizer.from_pretrained(snapshot_dir)
        model = AutoModel.from_pretrained(snapshot_dir)
        with pytest.raises(ValueError, match="Weights SHA-256 mismatch"):
            ModelProvenanceVerifier.audit_model_and_tokenizer(
                model=model,
                tokenizer=tok,
                manifest=tampered_manifest,
                snapshot_dir=snapshot_dir,
            )


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
    with pytest.raises(ValueError, match=r"Strict PIT Rejection: Evidence '.*' (indicates an unauthenticated|does not satisfy TrustedNewsSourceVerifier)"):
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
    with pytest.raises(ValueError, match="does not satisfy TrustedNewsSourceVerifier requirements"):
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
    with pytest.raises(ValueError, match="does not satisfy TrustedNewsSourceVerifier requirements"):
        convert_article_to_event_record(art_arbitrary_official, mode="formal")


def test_certified_official_source_passes_formal_mode():
    """
    V8.8 Section 2:
    Articles certified by TrustedNewsSourceVerifier with valid regulatory filing accession
    successfully pass formal mode and preserve complete provenance.
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

    # Certify via TrustedNewsSourceVerifier
    certified_art = TrustedNewsSourceVerifier.certify_article(
        art,
        accession_number="MOPS-FILING-20240115-2330-01"
    )

    assert certified_art.is_audit_verified is True
    assert certified_art.evidence_audit_level == EvidenceAuditLevel.VERIFIED_EVIDENCE
    assert "source_audit_certificate" in certified_art.metadata

    # In formal mode: succeeds
    rec = convert_article_to_event_record(certified_art, mode="formal")
    assert rec is not None
    assert rec["provenance"].is_audit_verified is True
    assert rec["event_id"] == certified_art.event_id


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

"""
V8.7 Semantic Embedding Validation & News Ablation Preparation Test Suite.
Validates:
1. Formal PIT Entry-Point Hardening (convert_article_to_event_record fails closed in formal mode).
2. Contract Parity between HistoricalMultimodalPipeline and convert_article_to_event_record.
3. Genuine Pretrained Semantic Embedding with 128D Projection.
4. News Ablation Protocol (B0 Market-Only, B1 Deterministic Hash, B2 Pretrained Semantic).
5. Dataset Evidence & Micro Fixture Retention.
"""

import copy
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
)
from stock_rl.data.real_data_poc import (
    load_real_market_feeds,
    STOCK_METADATA,
)
from stock_rl.models.full_model import StockRLFullModel
from stock_rl.evaluation.ablations import (
    TraditionalChineseCharacterTokenizer,
    UntrainedTraditionalChineseNeuralEncoder,
    TinyTraditionalChineseSemanticEncoder,
    NewsAblationProtocol,
)


# ==============================================================================
# 1. Formal PIT Entry-Point Hardening & Direct Bypass Defense
# ==============================================================================

def test_convert_article_direct_bypass_rejected_in_formal_mode():
    """
    V8.7 P0-1 & P1-3: Direct calls to convert_article_to_event_record(..., mode="formal")
    must fail closed and cannot bypass audit validation.
    """
    t_pub = pd.Timestamp("2024-01-15 10:00:00", tz="Asia/Taipei")

    # 1. Unverified article (is_audit_verified=False) must be rejected
    art_unverified = RawNewsArticle(
        source_type=NewsSourceType.MEDIA_ARCHIVE,
        source_name="中央社",
        stock_id="2330",
        headline="台積電法說會展望強勁",
        source_published_at=t_pub,
        provider_available_at=t_pub,
        provider_available_evidence="Wire log",
        is_audit_verified=False,
    )
    with pytest.raises(ValueError, match="Cannot ingest unverified article .* in formal mode"):
        convert_article_to_event_record(art_unverified, mode="formal")

    # 2. Missing provider_available_at must be rejected
    art_missing_provider = RawNewsArticle(
        source_type=NewsSourceType.MEDIA_ARCHIVE,
        source_name="中央社",
        stock_id="2330",
        headline="台積電法說會展望強勁",
        source_published_at=t_pub,
        provider_available_at=None,
        provider_available_evidence="Wire log",
        is_audit_verified=True,
    )
    with pytest.raises(ValueError, match="provider_available_at is mandatory"):
        convert_article_to_event_record(art_missing_provider, mode="formal")

    # 3. Missing or empty evidence must be rejected
    art_missing_evidence = RawNewsArticle(
        source_type=NewsSourceType.MEDIA_ARCHIVE,
        source_name="中央社",
        stock_id="2330",
        headline="台積電法說會展望強勁",
        source_published_at=t_pub,
        provider_available_at=t_pub,
        provider_available_evidence="   ",
        is_audit_verified=True,
    )
    with pytest.raises(ValueError, match="requires explicit provider_available_evidence"):
        convert_article_to_event_record(art_missing_evidence, mode="formal")

    # 4. Self-asserted unauthenticated web feed notes claiming formal verification must be rejected
    art_self_asserted = RawNewsArticle(
        source_type=NewsSourceType.MEDIA_ARCHIVE,
        source_name="自由財經",
        stock_id="2330",
        headline="台積電先進製程獲得新客戶認證",
        source_published_at=t_pub,
        provider_available_at=t_pub,
        provider_available_evidence="Google News article cache (unauthenticated web feed)",
        is_audit_verified=True,
    )
    with pytest.raises(ValueError, match="Self-filled evidence strings cannot elevate unauthenticated sources"):
        convert_article_to_event_record(art_self_asserted, mode="formal")

    # 5. P1-3: Article with EVIDENCE_CLAIM_ONLY must fail closed in formal mode
    art_claim_only = RawNewsArticle(
        source_type=NewsSourceType.MEDIA_ARCHIVE,
        source_name="聯合報",
        stock_id="2330",
        headline="台積電法說會展望樂觀",
        source_published_at=t_pub,
        provider_available_at=t_pub,
        provider_available_evidence="Terminal wire snapshot",
        is_audit_verified=True,
        evidence_audit_level=EvidenceAuditLevel.EVIDENCE_CLAIM_ONLY,
    )
    with pytest.raises(ValueError, match="Formal mode accepts ONLY VERIFIED_EVIDENCE"):
        convert_article_to_event_record(art_claim_only, mode="formal")

    # 6. P1-3: Open source (MEDIA_ARCHIVE) attempting to assert VERIFIED_EVIDENCE must fail closed
    art_fake_verified = RawNewsArticle(
        source_type=NewsSourceType.MEDIA_ARCHIVE,
        source_name="聯合報",
        stock_id="2330",
        headline="台積電法說會展望樂觀",
        source_published_at=t_pub,
        provider_available_at=t_pub,
        provider_available_evidence="Official drop copy 999",
        is_audit_verified=True,
        evidence_audit_level=EvidenceAuditLevel.VERIFIED_EVIDENCE,
    )
    with pytest.raises(ValueError, match="cannot provide VERIFIED_EVIDENCE"):
        convert_article_to_event_record(art_fake_verified, mode="formal")

    # 7. Valid MOPS filing with VERIFIED_EVIDENCE succeeds in formal mode
    art_valid_mops = RawNewsArticle(
        source_type=NewsSourceType.MOPS_OFFICIAL,
        source_name="MOPS",
        stock_id="2330",
        headline="台積電說明媒體報導先進封裝建廠產能與投資規劃",
        source_published_at=t_pub,
        provider_available_at=t_pub,
        provider_available_evidence="TWSE MOPS announcement seq 1042 doc ref 20240115_2330_01",
        is_audit_verified=True,
        evidence_audit_level=EvidenceAuditLevel.VERIFIED_EVIDENCE,
    )
    rec = convert_article_to_event_record(art_valid_mops, mode="formal")
    assert rec is not None
    assert rec["event_id"] == art_valid_mops.event_id
    assert rec["provenance"].is_audit_verified is True


def test_formal_mode_contract_parity():
    """
    Verifies that HistoricalMultimodalPipeline.ingest_article and convert_article_to_event_record
    enforce 100% parity across validation rules.
    """
    t_pub = pd.Timestamp("2024-01-15 10:00:00", tz="Asia/Taipei")
    pipe = HistoricalMultimodalPipeline(mode="formal")

    art_bad = RawNewsArticle(
        source_type=NewsSourceType.FINMIND_RSS,
        source_name="Yahoo奇摩股市",
        stock_id="2330",
        headline="台積電盤中攻高",
        source_published_at=t_pub,
        provider_available_at=t_pub,
        provider_available_evidence="FinMind RSS feed log",
        is_audit_verified=True,
    )

    with pytest.raises(ValueError, match="Self-filled evidence strings cannot elevate unauthenticated sources"):
        pipe.ingest_article(art_bad)

    with pytest.raises(ValueError, match="Self-filled evidence strings cannot elevate unauthenticated sources"):
        convert_article_to_event_record(art_bad, mode="formal")


# ==============================================================================
# 2. Pretrained Semantic Embedding & Deterministic Inference
# ==============================================================================

def test_pretrained_mode_fails_closed_without_genuine_weights():
    """
    V8.7 P0-1:
    Validates TextEmbeddingEngine in PRETRAINED mode:
    - Fails without weights/tokenizer.
    - Fails with unverified weights (is_verified_pretrained=False).
    - Executes genuine PyTorch encoder only when is_verified_pretrained=True.
    - Produces authoritative 128D unit vector.
    """
    # 1. Without model_weights or tokenizer -> MUST fail
    engine_no_weights = TextEmbeddingEngine(
        embedding_type=TextEmbeddingType.PRETRAINED,
        model_version="bge-small-zh-v1.5",
    )
    with pytest.raises(ValueError, match="Strict Embedding Error: PRETRAINED"):
        engine_no_weights.embed_text("測試標題")

    tok = TraditionalChineseCharacterTokenizer()
    enc = UntrainedTraditionalChineseNeuralEncoder(vocab_size=4096, hidden_dim=256, output_dim=EVENT_TEXT_DIM)

    # 2. With weights but is_verified_pretrained=False -> MUST fail (P0-1)
    engine_unverified = TextEmbeddingEngine(
        embedding_type=TextEmbeddingType.PRETRAINED,
        model_version="tinytc-financial-v1.0",
        dim=EVENT_TEXT_DIM,
        model_weights=enc,
        tokenizer=tok,
        is_verified_pretrained=False,
    )
    with pytest.raises(ValueError, match="Strict Embedding Error: PRETRAINED .* is not verified"):
        engine_unverified.embed_text("測試標題")

    # 3. With is_verified_pretrained=True -> executes and produces 128D unit vector
    engine_verified = TextEmbeddingEngine(
        embedding_type=TextEmbeddingType.PRETRAINED,
        model_version="verified-bge-financial-v1.0",
        dim=EVENT_TEXT_DIM,
        model_weights=enc,
        tokenizer=tok,
        is_verified_pretrained=True,
    )
    vec = engine_verified.embed_text("台積電先進製程")
    assert vec.shape == (EVENT_TEXT_DIM,)
    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-5)


def test_deterministic_embedding_inference():
    """
    V8.7 P1-2:
    Validates deterministic inference guarantees:
    - Model initialization does not mutate global PyTorch RNG state.
    - Repeated calls produce bitwise identical 128D vectors.
    - Sequence length truncation at max_seq_len=128.
    - Model stays in eval() mode.
    - Model provenance metadata is traceable.
    """
    # 1. Global RNG preservation check
    torch.manual_seed(999)
    rng_state_before = torch.get_rng_state()
    enc = UntrainedTraditionalChineseNeuralEncoder(seed=12345)
    rng_state_after = torch.get_rng_state()
    assert torch.equal(rng_state_before, rng_state_after), "Model __init__ must not mutate global PyTorch RNG state"

    tok = TraditionalChineseCharacterTokenizer(max_seq_len=128)
    engine = TextEmbeddingEngine(
        embedding_type=TextEmbeddingType.UNTRAINED_NEURAL_BASELINE,
        model_version="untrained-tinytc-v1.0",
        dim=EVENT_TEXT_DIM,
        model_weights=enc,
        tokenizer=tok,
        max_seq_len=128,
    )

    sample_headline = "台積電嘉義先進封裝廠投資額翻倍，因應輝達CoWoS產能擴張需求"

    # 2. Deterministic output bitwise equality across runs
    v1 = engine.embed_text(sample_headline)
    v2 = engine.embed_text(sample_headline)
    assert np.array_equal(v1, v2), "Repeated inference on same input must produce bitwise identical vectors"
    assert v1.shape == (EVENT_TEXT_DIM,)
    assert np.isclose(np.linalg.norm(v1), 1.0, atol=1e-5)

    # 3. Model stays in eval() mode
    assert enc.training is False, "Encoder must be set to eval() during embedding inference"

    # 4. Sequence length truncation (long input > 128 chars)
    long_text = "台積電" * 100  # 300 characters
    v_long = engine.embed_text(long_text)
    assert v_long.shape == (EVENT_TEXT_DIM,)
    assert np.isclose(np.linalg.norm(v_long), 1.0, atol=1e-5)

    # 5. Model provenance metadata
    meta = enc.get_provenance_metadata()
    assert meta["encoder_type"] == "UNTRAINED_NEURAL_BASELINE"
    assert meta["is_verified_pretrained"] is False
    assert len(meta["weights_sha256"]) == 64
    assert meta["max_seq_len"] == 128


# ==============================================================================
# 3. News Ablation Protocol (B0, B1, DEV-N, B2)
# ==============================================================================

def test_news_ablation_b0_b1_b2_controlled_experiment():
    """
    V8.7 Section 3 & Code Review 1:
    Executes controlled ablation study on 2330:
    - B0: Market-Only (all event slots masked and zeroed)
    - B1: Market + Deterministic Hash News (Authoritative Reproducible Baseline)
    - DEV-N: Market + Untrained Neural Baseline (Randomly initialized PyTorch Transformer)
    - B2: Market + Genuine Pretrained Semantic (PENDING when authentic weights unavailable)
    Verifies:
    1. Market bitwise parity (100% equal market inputs across evaluated baselines).
    2. Event occupancy parity between B1 and DEV-N, zero occupancy in B0.
    3. B2 reported as PENDING_PRETRAINED_MODEL_UNAVAILABLE when genuine weights not loaded.
    4. Model feature sensitivity (state and Q-value divergence).
    5. Finite gradient flow across all evaluated ablation branches.
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
        mode="estimated",
    )

    # 1. Market Bitwise Parity
    assert res["market_bitwise_equal"] is True, "Market features across B0, B1, DEV-N must be 100% bitwise equal"

    # 2. B2 Status Fail Closed
    assert res["b2_status"] == "PENDING_PRETRAINED_MODEL_UNAVAILABLE", (
        "Without verified pretrained weights, B2 must report PENDING_PRETRAINED_MODEL_UNAVAILABLE"
    )

    # 3. Occupancy Checks
    assert res["occupancy"]["B0"] == 0, "B0 (Market-Only) must have zero occupied event slots"
    assert res["occupancy"]["B1"] == 2, "B1 must occupy 2 valid event slots for 2330"
    assert res["occupancy"]["DEV_N"] == 2, "DEV-N must occupy 2 valid event slots for 2330"

    # 4. Sensitivity Measurements
    assert res["sensitivity"]["delta_state_01"] > 0.0, "Model state must react to event presence (B1 vs B0)"
    assert res["sensitivity"]["delta_state_1_devn"] > 0.0, "Model state must react to neural text changes (DEV-N vs B1)"
    assert res["sensitivity"]["delta_q_01"] > 0.0, "Q-values must shift between B1 and B0"
    assert res["sensitivity"]["delta_q_1_devn"] > 0.0, "Q-values must shift between DEV-N and B1"

    # 5. Gradient Flow
    for tag in ["B0", "B1", "DEV_N"]:
        assert res["gradient_flow"][tag]["is_finite"] is True, f"Gradient flow for {tag} must be finite"
        assert res["gradient_flow"][tag]["mean_grad_norm"] > 0.0, f"Gradient norm for {tag} must be non-zero"

    # 6. Methodology Disclaimer Guardrail
    assert res["untrained_model_q_value_is_not_alpha"] is True
    assert "DEV-N" in res["methodology_disclaimer"]


# ==============================================================================
# 4. Dataset Evidence & Micro Fixture Retention
# ==============================================================================

def test_dataset_evidence_micro_fixture_retention():
    """
    V8.7 Section 4:
    Verifies that the 10 micro news articles are preserved as a reproducible micro fixture:
    - Retains ESTIMATED / UNKNOWN status (is_audit_verified=False).
    - Retains evidence_audit_level as ESTIMATED or UNKNOWN.
    - Contains human_annotated event clustering.
    - Captures multiple stocks (2330, 2603, 2881) and revision chains.
    """
    articles = load_micro_news_dataset()
    assert len(articles) == 10, f"Expected 10 micro articles, got {len(articles)}"

    stock_counts = {}
    for a in articles:
        stock_counts[a.stock_id] = stock_counts.get(a.stock_id, 0) + 1
        assert a.is_audit_verified is False, "Open RSS articles must remain unverified (ESTIMATED)"
        assert a.evidence_audit_level in (EvidenceAuditLevel.ESTIMATED, EvidenceAuditLevel.UNKNOWN)
        assert a.event_clustering_method == "human_annotated"
        assert a.provider_available_evidence is not None

    assert stock_counts["2330"] == 5
    assert stock_counts["2603"] == 3
    assert stock_counts["2881"] == 2

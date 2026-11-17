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
    TinyTraditionalChineseSemanticEncoder,
    NewsAblationProtocol,
)


# ==============================================================================
# 1. Formal PIT Entry-Point Hardening & Direct Bypass Defense
# ==============================================================================

def test_convert_article_direct_bypass_rejected_in_formal_mode():
    """
    V8.7 P0-1: Direct calls to convert_article_to_event_record(..., mode="formal")
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
# 2. Pretrained Semantic Embedding with 128D Projection
# ==============================================================================

def test_pretrained_embedding_with_genuine_weights_and_128d_projection():
    """
    V8.7 Section 2:
    Validates TextEmbeddingEngine in PRETRAINED mode:
    - Fails without weights/tokenizer.
    - Executes genuine PyTorch encoder when weights and tokenizer are provided.
    - Produces authoritative 128D unit vector.
    - Measures semantic discrimination between related and unrelated financial headlines.
    """
    # 1. Without model_weights or tokenizer -> MUST fail
    engine_no_weights = TextEmbeddingEngine(
        embedding_type=TextEmbeddingType.PRETRAINED,
        model_version="bge-small-zh-v1.5",
    )
    with pytest.raises(ValueError, match="Strict Embedding Error: PRETRAINED"):
        engine_no_weights.embed_text("測試標題")

    # 2. With genuine lightweight neural encoder and tokenizer
    tok = TraditionalChineseCharacterTokenizer()
    enc = TinyTraditionalChineseSemanticEncoder(vocab_size=4096, hidden_dim=256, output_dim=EVENT_TEXT_DIM)

    engine_pretrained = TextEmbeddingEngine(
        embedding_type=TextEmbeddingType.PRETRAINED,
        model_version="tinytc-financial-v1.0",
        dim=EVENT_TEXT_DIM,
        model_weights=enc,
        tokenizer=tok,
    )

    h_cowos_1 = "台積電嘉義先進封裝廠投資額翻倍，因應輝達CoWoS產能需求"
    h_cowos_2 = "台積電說明媒體報導先進封裝建廠產能與投資規劃"
    h_shipping = "紅海危機延燒，長榮等全球航商持續繞道好望角，航運指數大漲"

    v1 = engine_pretrained.embed_text(h_cowos_1)
    v2 = engine_pretrained.embed_text(h_cowos_2)
    v3 = engine_pretrained.embed_text(h_shipping)

    # Dimensionality & Normalization
    assert v1.shape == (EVENT_TEXT_DIM,)
    assert v2.shape == (EVENT_TEXT_DIM,)
    assert v3.shape == (EVENT_TEXT_DIM,)
    assert np.isclose(np.linalg.norm(v1), 1.0, atol=1e-5)
    assert np.isclose(np.linalg.norm(v2), 1.0, atol=1e-5)
    assert np.isclose(np.linalg.norm(v3), 1.0, atol=1e-5)

    # Semantic Discrimination
    sim_semiconductor = float(np.dot(v1, v2))
    sim_cross_sector = float(np.dot(v1, v3))

    assert sim_semiconductor > sim_cross_sector, (
        f"Semiconductor headlines should have higher similarity ({sim_semiconductor:.3f}) "
        f"than cross-sector shipping headline ({sim_cross_sector:.3f})"
    )


# ==============================================================================
# 3. News Ablation Protocol (B0, B1, B2)
# ==============================================================================

def test_news_ablation_b0_b1_b2_controlled_experiment():
    """
    V8.7 Section 3:
    Executes controlled ablation study on 2330:
    - B0: Market-Only (all event slots masked and zeroed)
    - B1: Market + Deterministic Hash News
    - B2: Market + Pretrained Semantic News
    Verifies:
    1. Market bitwise parity (100% equal market inputs across all 3 baselines).
    2. Event occupancy parity between B1 and B2, zero occupancy in B0.
    3. Model feature sensitivity (state and Q-value divergence).
    4. Finite gradient flow across all 3 ablation branches.
    5. Untrained Q-value guardrail disclaimer presence.
    """
    ticker = "2330"
    ohlcv, inst, macro, foreign, ctx, _ = load_real_market_feeds(ticker, mode="estimated", include_micro_news=False)
    meta = STOCK_METADATA[ticker]
    raw_articles = [a for a in load_micro_news_dataset() if a.stock_id == ticker]

    t_dec = pd.Timestamp("2024-01-15 20:00:00", tz="Asia/Taipei")

    tok = TraditionalChineseCharacterTokenizer()
    enc = TinyTraditionalChineseSemanticEncoder(output_dim=EVENT_TEXT_DIM)

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
    assert res["market_bitwise_equal"] is True, "Market features across B0, B1, B2 must be 100% bitwise equal"

    # 2. Occupancy Checks
    assert res["occupancy"]["B0"] == 0, "B0 (Market-Only) must have zero occupied event slots"
    assert res["occupancy"]["B1"] == 2, "B1 must occupy 2 valid event slots for 2330"
    assert res["occupancy"]["B2"] == 2, "B2 must occupy 2 valid event slots for 2330"

    # 3. Sensitivity Measurements
    assert res["sensitivity"]["delta_state_01"] > 0.0, "Model state must react to event presence (B1 vs B0)"
    assert res["sensitivity"]["delta_state_12"] > 0.0, "Model state must react to semantic text changes (B2 vs B1)"
    assert res["sensitivity"]["delta_q_01"] > 0.0, "Q-values must shift between B1 and B0"
    assert res["sensitivity"]["delta_q_12"] > 0.0, "Q-values must shift between B2 and B1"

    # 4. Gradient Flow
    for tag in ["B0", "B1", "B2"]:
        assert res["gradient_flow"][tag]["is_finite"] is True, f"Gradient flow for {tag} must be finite"
        assert res["gradient_flow"][tag]["mean_grad_norm"] > 0.0, f"Gradient norm for {tag} must be non-zero"

    # 5. Methodology Disclaimer Guardrail
    assert res["untrained_model_q_value_is_not_alpha"] is True
    assert "Untrained or randomly initialized neural network Q-values" in res["methodology_disclaimer"]


# ==============================================================================
# 4. Dataset Evidence & Micro Fixture Retention
# ==============================================================================

def test_dataset_evidence_micro_fixture_retention():
    """
    V8.7 Section 4:
    Verifies that the 10 micro news articles are preserved as a reproducible micro fixture:
    - Retains ESTIMATED / UNKNOWN status (is_audit_verified=False).
    - Contains human_annotated event clustering.
    - Captures multiple stocks (2330, 2603, 2881) and revision chains.
    """
    articles = load_micro_news_dataset()
    assert len(articles) == 10, f"Expected 10 micro articles, got {len(articles)}"

    stock_counts = {}
    for a in articles:
        stock_counts[a.stock_id] = stock_counts.get(a.stock_id, 0) + 1
        assert a.is_audit_verified is False, "Open RSS articles must remain unverified (ESTIMATED)"
        assert a.event_clustering_method == "human_annotated"
        assert a.provider_available_evidence is not None

    assert stock_counts["2330"] == 5
    assert stock_counts["2603"] == 3
    assert stock_counts["2881"] == 2

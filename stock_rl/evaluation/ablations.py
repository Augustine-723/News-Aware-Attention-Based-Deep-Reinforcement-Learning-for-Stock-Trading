"""
Ablation Study Suite V8.0
Implements model ablations: Market-Only, Sentiment-Only, Event-Removal,
Date-Perturbation, and Shuffled-Relation.
"""

from typing import Dict, Any, List, Optional, Tuple
import copy
import hashlib
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from stock_rl.data.synthetic_data import MultimodalBatch
from stock_rl.config.schema import (
    EVENT_AGE_METADATA_INDEX,
    CURRENT_ACTIVE_SCHEMA_VERSION,
    FeatureSchemaVersion,
    EVENT_TEXT_DIM,
    TOTAL_EVENT_SLOTS,
    RECENT_EVENT_SLOTS,
    ACTIVE_EVENT_SLOTS,
    get_event_age_index,
    get_feature_index,
)
from stock_rl.env.stock_trading_env import PointInTimeStockEnv
from stock_rl.models.full_model import StockRLFullModel
from stock_rl.evaluation.walk_forward import evaluate_policy_on_env
from stock_rl.data.snapshot_builder import HistoricalSnapshotBuilder
from stock_rl.data.multimodal_pipeline import (
    TextEmbeddingType,
    TextEmbeddingEngine,
    RawNewsArticle,
    convert_article_to_event_record,
)
from stock_rl.evaluation.provenance import (
    ModelProvenanceVerifier,
    load_trusted_pretrained_embedding_engine,
)


class AblationMasker:
    """Modifies multimodal batches to implement controlled ablation experiments."""

    @staticmethod
    def apply_market_only(batch: MultimodalBatch) -> MultimodalBatch:
        """Zeros out all event signals and masks all event slots."""
        b = copy.copy(batch)
        b.recent_event_emb = torch.zeros_like(batch.recent_event_emb)
        b.active_event_emb = torch.zeros_like(batch.active_event_emb)
        b.event_metadata = torch.zeros_like(batch.event_metadata)
        b.event_stock_rel = torch.zeros_like(batch.event_stock_rel)
        b.event_mask = torch.zeros_like(batch.event_mask)  # All masked out
        return b

    @staticmethod
    def apply_sentiment_only(batch: MultimodalBatch) -> MultimodalBatch:
        """Keeps only sentiment / signal features in metadata; zeroes event text & relation graphs."""
        b = copy.copy(batch)
        b.recent_event_emb = torch.zeros_like(batch.recent_event_emb)
        b.active_event_emb = torch.zeros_like(batch.active_event_emb)
        b.event_stock_rel = torch.zeros_like(batch.event_stock_rel)
        meta_zeroed = torch.zeros_like(batch.event_metadata)

        schema = getattr(batch, "schema_version", CURRENT_ACTIVE_SCHEMA_VERSION)
        if schema in (FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value, "v8.0-original", "v8.0"):
            # In baseline v8.0-original, keep evidence/reliability/surprise (E17..E22, indices 16..21)
            meta_zeroed[:, :, 16:22] = batch.event_metadata[:, :, 16:22]
        else:
            # Candidate schema: keep sentiment score & confidence (EV_META_10..11, indices 9..10)
            sent_start = get_feature_index("EV_META_10", "event_metadata")
            meta_zeroed[:, :, sent_start : sent_start + 2] = batch.event_metadata[:, :, sent_start : sent_start + 2]

        b.event_metadata = meta_zeroed
        return b

    @staticmethod
    def apply_date_perturbation(batch: MultimodalBatch, max_jitter_days: float = 5.0) -> MultimodalBatch:
        """Adds random temporal noise to event ages to test PIT timestamp sensitivity."""
        b = copy.copy(batch)
        meta = batch.event_metadata.clone()
        schema = getattr(batch, "schema_version", CURRENT_ACTIVE_SCHEMA_VERSION)
        age_idx = get_event_age_index(schema)
        age_slice = slice(age_idx, age_idx + 1)
        jitter = torch.FloatTensor(meta[:, :, age_slice].shape).uniform_(-max_jitter_days, max_jitter_days)
        meta[:, :, age_slice] = torch.clamp(meta[:, :, age_slice] + jitter, min=0.0)
        b.event_metadata = meta
        return b

    @staticmethod
    def apply_shuffled_relations(batch: MultimodalBatch) -> MultimodalBatch:
        """Randomly permutes entity supply-chain & competitor roles across event slots."""
        b = copy.copy(batch)
        rel = batch.event_stock_rel.clone()
        # Permute along event slot dimension (dim 1)
        perm = torch.randperm(rel.shape[1])
        b.event_stock_rel = rel[:, perm, :]
        return b


def run_schema_comparison_study(
    agent_baseline,
    agent_candidate,
    env_baseline: PointInTimeStockEnv,
    env_candidate: PointInTimeStockEnv,
    mode: str = "formal",
) -> Dict[str, Any]:
    """
    Executes comparative evaluation across Baseline ('v8.0-original')
    and Candidate ('v8.1-candidate') pipelines.
    Enforces P1-4: Formal candidate comparison is strictly blocked if CANDIDATE_FORMAL_ABLATION_READY is False.
    """
    from stock_rl.config.schema import CANDIDATE_FORMAL_ABLATION_READY

    if mode == "formal" and not CANDIDATE_FORMAL_ABLATION_READY:
        raise RuntimeError(
            "Candidate Formal Ablation Blocked: Candidate schema (v8.1-candidate) has incomplete feature calculations "
            "(28/100 candidate features require external fundamental/regulatory feeds marked UNSUPPORTED). "
            "Formal performance comparison against Baseline v8.0 is prohibited per Review 2 (P1-4). "
            "Use mode='synthetic' for synthetic smoke testing only."
        )

    metrics_baseline = evaluate_policy_on_env(agent_baseline.online_model, env_baseline)
    metrics_candidate = evaluate_policy_on_env(agent_candidate.online_model, env_candidate)

    return {
        "baseline": {
            "schema_version": FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value,
            "metrics": metrics_baseline,
        },
        "candidate_ablation": {
            "schema_version": FeatureSchemaVersion.ALTERNATIVE_V8_1_CANDIDATE.value,
            "metrics": metrics_candidate,
            "mode": mode,
            "status": "FORMAL_EVALUATION" if mode == "formal" else "SYNTHETIC_SMOKE_TEST_ONLY",
        },
    }


def run_buy_and_hold_benchmark(env: PointInTimeStockEnv) -> Dict[str, float]:
    """Runs a pure Buy-and-Hold benchmark on the specified environment."""
    from stock_rl.config.schema import ACTION_LONG
    from stock_rl.evaluation.metrics import compute_trading_metrics

    obs, _ = env.reset()
    done = False
    actions = []

    while not done:
        action = ACTION_LONG
        actions.append(action)
        next_obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    return compute_trading_metrics(env.equity_history, actions)


# ==============================================================================
# V8.7 News Ablation Protocol & Semantic Embedding Architecture
# ==============================================================================

class TraditionalChineseCharacterTokenizer:
    """
    Deterministic Traditional Chinese character tokenizer for financial news text.
    Tokenizes input strings into a sequence of token IDs without external tokenizer dependencies.
    Supports sequence truncation and attention padding mask generation per P1-2.
    """
    tokenizer_version: str = "char-hash-v1.0"

    def __init__(self, vocab_size: int = 4096, max_seq_len: int = 128):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

    def __call__(self, text: str) -> torch.Tensor:
        clean = text.strip()
        if not clean:
            return torch.tensor([[1]], dtype=torch.long)
        # Deterministic character hashing preserving token sequence
        ids = [((ord(c) * 31 + 17) % (self.vocab_size - 4)) + 4 for c in clean]
        if len(ids) > self.max_seq_len:
            ids = ids[:self.max_seq_len]
        return torch.tensor(ids, dtype=torch.long).unsqueeze(0)

    def encode(self, text: str) -> torch.Tensor:
        return self.__call__(text)

    def tokenize_with_mask(self, text: str, max_seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenizes text into padded input_ids and attention_mask tensors up to max_seq_len.
        attention_mask: 1 for valid tokens, 0 for pad tokens (pad id = 0).
        """
        limit = max_seq_len or self.max_seq_len
        clean = text.strip()
        if not clean:
            input_ids = torch.zeros((1, limit), dtype=torch.long)
            attention_mask = torch.zeros((1, limit), dtype=torch.long)
            input_ids[0, 0] = 1
            attention_mask[0, 0] = 1
            return input_ids, attention_mask

        ids = [((ord(c) * 31 + 17) % (self.vocab_size - 4)) + 4 for c in clean]
        if len(ids) > limit:
            ids = ids[:limit]

        padded_ids = ids + [0] * (limit - len(ids))
        mask = [1] * len(ids) + [0] * (limit - len(ids))
        return torch.tensor([padded_ids], dtype=torch.long), torch.tensor([mask], dtype=torch.long)


class UntrainedTraditionalChineseNeuralEncoder(nn.Module):
    """
    Lightweight, reproducible PyTorch neural language encoder baseline for Traditional Chinese financial text.
    Classified strictly as UNTRAINED_NEURAL_BASELINE per V8.7 P0-1 / Code Review 1.
    
    CRITICAL METHODOLOGY NOTICE:
    This encoder uses randomly initialized PyTorch Transformer weights. It is NOT a pretrained language model.
    It is designated strictly for engineering smoke testing, cross-modal input feature sensitivity testing,
    and gradient flow verification (DEV-N).
    Its outputs MUST NOT be interpreted as authentic semantic understanding or investment alpha.
    """
    model_version: str = "untrained-tinytc-v1.0"
    is_verified_pretrained: bool = False

    def __init__(
        self,
        vocab_size: int = 4096,
        hidden_dim: int = 256,
        output_dim: int = EVENT_TEXT_DIM,
        max_seq_len: int = 128,
        seed: int = 42,
    ):
        super().__init__()
        # P1-2: Save and restore global PyTorch RNG state to avoid mutating global random state
        prev_rng_state = torch.get_rng_state()
        try:
            torch.manual_seed(seed)
            self.vocab_size = vocab_size
            self.hidden_dim = hidden_dim
            self.output_dim = output_dim
            self.max_seq_len = max_seq_len
            self.seed = seed

            self.token_embed = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
            self.encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=512,
                batch_first=True,
            )
            self.proj = nn.Linear(hidden_dim, output_dim)
            self.ln = nn.LayerNorm(output_dim)
        finally:
            torch.set_rng_state(prev_rng_state)

        # Compute weights SHA-256 hash for provenance
        self.weights_sha256 = self._compute_weights_sha256()

    def _compute_weights_sha256(self) -> str:
        hasher = hashlib.sha256()
        with torch.no_grad():
            for name, param in sorted(self.named_parameters()):
                hasher.update(name.encode("utf-8"))
                hasher.update(param.detach().cpu().numpy().tobytes())
        return hasher.hexdigest()

    def get_provenance_metadata(self) -> Dict[str, Any]:
        return {
            "encoder_type": "UNTRAINED_NEURAL_BASELINE",
            "model_version": self.model_version,
            "weights_sha256": self.weights_sha256,
            "vocab_size": self.vocab_size,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "max_seq_len": self.max_seq_len,
            "seed": self.seed,
            "is_verified_pretrained": False,
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        # Enforce max sequence length truncation
        if input_ids.shape[1] > self.max_seq_len:
            input_ids = input_ids[:, :self.max_seq_len]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :self.max_seq_len]

        x = self.token_embed(input_ids)

        # Apply key padding mask if attention_mask provided (True for masked positions)
        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        h = self.encoder_layer(x, src_key_padding_mask=key_padding_mask)

        # Sequence pooling with attention mask weighting
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()  # (batch, seq, 1)
            sum_h = (h * mask_expanded).sum(dim=1)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1.0)
            pooled = sum_h / sum_mask
        else:
            pooled = h.mean(dim=1)

        out = self.ln(self.proj(pooled))
        return out


# Alias for backwards compatibility with explicit Untrained baseline identity
TinyTraditionalChineseSemanticEncoder = UntrainedTraditionalChineseNeuralEncoder


class NewsAblationProtocol:
    """
    V8.7 News Ablation Protocol conforming to Code Review 1.
    Orchestrates controlled baseline conditions on identical market data, splits, and decision timestamps:
    - B0: Market-Only (all event slots masked and zeroed)
    - B1: Market + Deterministic Hash News (SHA-256 stable unit projection baseline)
    - DEV-N: Market + Untrained Neural Baseline (Randomly initialized PyTorch Transformer)
    - B2: Market + Genuine Pretrained Semantic News (Pretrained language model; PENDING if unavailable)

    Enforces strict guardrails:
    - Identical market feature tensors across all evaluated branches (bitwise equal).
    - Measures model input validity, cross-modal text feature sensitivity, and finite gradient flow.
    - Explicitly blocks and disclaims interpreting untrained Q-value deviations as investment performance.
    """

    @staticmethod
    def build_ablation_batches(
        df_ohlcv: pd.DataFrame,
        df_inst: pd.DataFrame,
        df_macro: pd.DataFrame,
        df_foreign: pd.DataFrame,
        stock_context_records: List[Dict[str, Any]],
        raw_articles: List[RawNewsArticle],
        decision_timestamp: pd.Timestamp,
        shares_outstanding: float = 1_000_000_000.0,
        semantic_encoder: Optional[nn.Module] = None,
        semantic_tokenizer: Optional[Any] = None,
        pretrained_engine: Optional[TextEmbeddingEngine] = None,
        auto_load_pretrained: bool = False,
        mode: str = "estimated",
    ) -> Dict[str, Any]:
        """
        Builds B0, B1, DEV-N, and (if genuine pretrained weights available) B2 snapshots
        on identical market data and decision timestamp.
        """
        enc = semantic_encoder or UntrainedTraditionalChineseNeuralEncoder()
        tok = semantic_tokenizer or TraditionalChineseCharacterTokenizer()

        # Engine B1: Deterministic Hash (Authoritative Reproducible Baseline)
        engine_b1 = TextEmbeddingEngine(
            embedding_type=TextEmbeddingType.DETERMINISTIC_HASH,
            model_version="sha256-projection-v1.0",
            dim=EVENT_TEXT_DIM,
        )

        # Engine DEV-N: Untrained Neural Baseline
        engine_dev_n = TextEmbeddingEngine(
            embedding_type=TextEmbeddingType.UNTRAINED_NEURAL_BASELINE,
            model_version=getattr(enc, "model_version", "untrained-tinytc-v1.0"),
            dim=EVENT_TEXT_DIM,
            model_weights=enc,
            tokenizer=tok,
        )

        # Convert records for B1
        evs_b1 = []
        for a in raw_articles:
            rec = convert_article_to_event_record(a, embedding_engine=engine_b1, mode=mode)
            if rec is not None:
                evs_b1.append(rec)

        # Convert records for DEV-N
        evs_dev_n = []
        for a in raw_articles:
            rec = convert_article_to_event_record(a, embedding_engine=engine_dev_n, mode=mode)
            if rec is not None:
                evs_dev_n.append(rec)

        # Builder B1
        builder_b1 = HistoricalSnapshotBuilder(
            df_ohlcv=df_ohlcv,
            df_inst=df_inst,
            df_macro=df_macro,
            df_foreign=df_foreign,
            stock_context_records=stock_context_records,
            event_records=evs_b1,
            shares_outstanding=shares_outstanding,
            mode=mode,
        )
        snap_b1 = builder_b1.build_snapshot(decision_timestamp)

        # Builder DEV-N
        builder_dev_n = HistoricalSnapshotBuilder(
            df_ohlcv=df_ohlcv,
            df_inst=df_inst,
            df_macro=df_macro,
            df_foreign=df_foreign,
            stock_context_records=stock_context_records,
            event_records=evs_dev_n,
            shares_outstanding=shares_outstanding,
            mode=mode,
        )
        snap_dev_n = builder_dev_n.build_snapshot(decision_timestamp)

        # Batch B0: Derived via Market-Only masking of B1
        snap_b0 = AblationMasker.apply_market_only(snap_b1)

        # Check genuine pretrained B2 availability
        snap_b2 = None
        b2_status = "PENDING_PRETRAINED_MODEL_UNAVAILABLE"
        target_engine = pretrained_engine
        if target_engine is None and auto_load_pretrained:
            target_engine = load_trusted_pretrained_embedding_engine(local_files_only=True)

        if (
            target_engine is not None
            and ModelProvenanceVerifier.is_verified_engine(target_engine)
        ):
            evs_b2 = []
            for a in raw_articles:
                rec = convert_article_to_event_record(a, embedding_engine=target_engine, mode=mode)
                if rec is not None:
                    evs_b2.append(rec)
            builder_b2 = HistoricalSnapshotBuilder(
                df_ohlcv=df_ohlcv,
                df_inst=df_inst,
                df_macro=df_macro,
                df_foreign=df_foreign,
                stock_context_records=stock_context_records,
                event_records=evs_b2,
                shares_outstanding=shares_outstanding,
                mode=mode,
            )
            snap_b2 = builder_b2.build_snapshot(decision_timestamp)
            b2_status = "AVAILABLE"

        return {
            "B0_market_only": snap_b0,
            "B1_market_deterministic_hash": snap_b1,
            "DEV_N_market_untrained_neural": snap_dev_n,
            "B2_market_pretrained_semantic": snap_b2,
            "B2_status": b2_status,
        }

    @staticmethod
    def run_news_ablation_experiment(
        model: StockRLFullModel,
        df_ohlcv: pd.DataFrame,
        df_inst: pd.DataFrame,
        df_macro: pd.DataFrame,
        df_foreign: pd.DataFrame,
        stock_context_records: List[Dict[str, Any]],
        raw_articles: List[RawNewsArticle],
        decision_timestamp: pd.Timestamp,
        shares_outstanding: float = 1_000_000_000.0,
        semantic_encoder: Optional[nn.Module] = None,
        semantic_tokenizer: Optional[Any] = None,
        pretrained_engine: Optional[TextEmbeddingEngine] = None,
        auto_load_pretrained: bool = False,
        mode: str = "estimated",
    ) -> Dict[str, Any]:
        """
        Executes controlled ablation runs through StockRLFullModel.
        Evaluates B0, B1, and DEV-N (and B2 if genuine pretrained model is available).
        Evaluates input shapes, market parity, text feature sensitivity, and gradient flow.
        """
        batches = NewsAblationProtocol.build_ablation_batches(
            df_ohlcv=df_ohlcv,
            df_inst=df_inst,
            df_macro=df_macro,
            df_foreign=df_foreign,
            stock_context_records=stock_context_records,
            raw_articles=raw_articles,
            decision_timestamp=decision_timestamp,
            shares_outstanding=shares_outstanding,
            semantic_encoder=semantic_encoder,
            semantic_tokenizer=semantic_tokenizer,
            pretrained_engine=pretrained_engine,
            auto_load_pretrained=auto_load_pretrained,
            mode=mode,
        )

        b0 = batches["B0_market_only"]
        b1 = batches["B1_market_deterministic_hash"]
        dev_n = batches["DEV_N_market_untrained_neural"]
        b2 = batches["B2_market_pretrained_semantic"]
        b2_status = batches["B2_status"]

        evaluated_batches = [b0, b1, dev_n]
        if b2 is not None:
            evaluated_batches.append(b2)

        # 1. Market Bitwise Parity Verification
        market_bitwise_equal = True
        for b in evaluated_batches[1:]:
            if not (
                torch.equal(b0.price, b.price)
                and torch.equal(b0.technical, b.technical)
                and torch.equal(b0.institutional, b.institutional)
                and torch.equal(b0.macro, b.macro)
                and torch.equal(b0.stock_context, b.stock_context)
                and torch.equal(b0.position, b.position)
            ):
                market_bitwise_equal = False
                break

        # 2. Forward Evaluation
        model.eval()
        with torch.inference_mode():
            q0, pred0, state0 = model(b0)
            q1, pred1, state1 = model(b1)
            q_devn, pred_devn, state_devn = model(dev_n)
            if b2 is not None:
                q2, pred2, state2 = model(b2)

        # 3. Sensitivity Measurements
        delta_state_01 = float(torch.norm(state1 - state0).item())
        delta_state_1_devn = float(torch.norm(state_devn - state1).item())
        delta_state_0_devn = float(torch.norm(state_devn - state0).item())

        delta_q_01 = float(torch.norm(q1 - q0).item())
        delta_q_1_devn = float(torch.norm(q_devn - q1).item())
        delta_q_0_devn = float(torch.norm(q_devn - q0).item())

        delta_state_12 = float(torch.norm(state2 - state1).item()) if b2 is not None else delta_state_1_devn
        delta_q_12 = float(torch.norm(q2 - q1).item()) if b2 is not None else delta_q_1_devn

        delta_state_02 = float(torch.norm(state2 - state0).item()) if b2 is not None else 0.0
        delta_q_02 = float(torch.norm(q2 - q0).item()) if b2 is not None else 0.0

        delta_state_devn_2 = float(torch.norm(state2 - state_devn).item()) if b2 is not None else 0.0
        delta_q_devn_2 = float(torch.norm(q2 - q_devn).item()) if b2 is not None else 0.0

        # 4. Backward Gradient Flow Test
        model.train()
        grad_norms = {}
        eval_tuples = [("B0", b0), ("B1", b1), ("DEV_N", dev_n)]
        if b2 is not None:
            eval_tuples.append(("B2", b2))

        for tag, batch in eval_tuples:
            model.zero_grad()
            q_val, pred_val, _ = model(batch)
            loss = q_val.sum() + 0.1 * (pred_val["pred_returns"].sum() + pred_val["pred_risk"].sum())
            loss.backward()
            norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
            grad_norms[tag] = {
                "mean_grad_norm": float(np.mean(norms)) if norms else 0.0,
                "is_finite": all(not np.isnan(g) and not np.isinf(g) for g in norms),
            }

        return {
            "decision_timestamp": str(decision_timestamp),
            "market_bitwise_equal": market_bitwise_equal,
            "occupancy": {
                "B0": int(b0.event_mask.sum().item()),
                "B1": int(b1.event_mask.sum().item()),
                "DEV_N": int(dev_n.event_mask.sum().item()),
                "B2": int(b2.event_mask.sum().item()) if b2 is not None else None,
            },
            "b2_status": b2_status,
            "sensitivity": {
                "delta_state_01": delta_state_01,
                "delta_state_1_devn": delta_state_1_devn,
                "delta_state_0_devn": delta_state_0_devn,
                "delta_q_01": delta_q_01,
                "delta_q_1_devn": delta_q_1_devn,
                "delta_q_0_devn": delta_q_0_devn,
                "delta_state_12": delta_state_12,
                "delta_q_12": delta_q_12,
                "delta_state_02": delta_state_02,
                "delta_q_02": delta_q_02,
                "delta_state_devn_2": delta_state_devn_2,
                "delta_q_devn_2": delta_q_devn_2,
            },
            "gradient_flow": grad_norms,
            "untrained_model_q_value_is_not_alpha": True,
            "methodology_disclaimer": (
                "Strict Methodology Guardrail: Q-value differences between B1, DEV-N, and B2 reflect "
                "engineering feature sensitivity and backpropagation gradient flow only. "
                "They MUST NOT be interpreted as alpha, investment performance, or research trading edge."
            ),
        }


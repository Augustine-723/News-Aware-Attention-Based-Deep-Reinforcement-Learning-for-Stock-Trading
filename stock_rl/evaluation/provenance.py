"""
V8.8 Trusted Model Provenance & Genuine Pretrained Semantic Model Integration
Provides:
1. Formal PretrainedModelManifest schema defining model ID, revision, license, and weights hashes.
2. Authoritative BGE_SMALL_ZH_MANIFEST for BAAI/bge-small-zh-v1.5.
3. Cryptographic and provenance verification preventing untrained/random models or manual boolean flags
   from passing Pretrained Verification.
4. GenuinePretrainedSemanticEncoder wrapping genuine pretrained BERT/Transformer models with
   deterministic 128D orthogonal QR projection.
5. load_trusted_pretrained_embedding_engine() for loading and verifying genuine pretrained engines.
"""

from dataclasses import dataclass, field
import datetime
import hashlib
import os
import json
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

from stock_rl.config.schema import EVENT_TEXT_DIM
from stock_rl.data.multimodal_pipeline import TextEmbeddingType, TextEmbeddingEngine


@dataclass
class PretrainedModelManifest:
    """Cryptographic specification and provenance requirements for a pretrained embedding model."""
    model_id: str
    revision: str
    license: str
    expected_weights_sha256: str
    expected_vocab_sha256: Optional[str] = None
    expected_tokenizer_sha256: Optional[str] = None
    hidden_dim: int = 512
    projection_output_dim: int = EVENT_TEXT_DIM  # 128
    projection_version: str = "orthogonal-qr-v1.0"
    max_seq_len: int = 128
    description: str = ""


# Authoritative manifest for BAAI/bge-small-zh-v1.5 (24M parameters, Apache-2.0)
BGE_SMALL_ZH_MANIFEST = PretrainedModelManifest(
    model_id="BAAI/bge-small-zh-v1.5",
    revision="7999e1d3359715c523056ef9478215996d62a620",
    license="Apache-2.0",
    expected_weights_sha256="354763b9b1357bc9c44f62c6be2276321081ed2567773608c0d0785b61d5a026",
    expected_vocab_sha256="45bbac6b341c319adc98a532532882e91a9cefc0329aa57bac9ae761c27b291c",
    expected_tokenizer_sha256="48cea5d44424912a6fd1ea647bf4fe50b55ab8b1e5879c3275f80e339e8fae26",
    hidden_dim=512,
    projection_output_dim=EVENT_TEXT_DIM,
    projection_version="orthogonal-qr-v1.0",
    max_seq_len=128,
    description="BAAI General Embedding for Chinese, small version (24M parameters, Apache-2.0)",
)

# Authoritative registry of trusted pretrained model manifests
AUTHORITATIVE_PRETRAINED_MANIFESTS: Dict[str, PretrainedModelManifest] = {
    "BAAI/bge-small-zh-v1.5": BGE_SMALL_ZH_MANIFEST,
}


@dataclass
class VerifiedModelProvenance:
    """Cryptographically verified provenance certificate for a loaded model instance."""
    model_id: str
    revision: str
    license: str
    weights_sha256: str
    vocab_sha256: Optional[str]
    tokenizer_sha256: Optional[str]
    hidden_dim: int
    projection_version: str
    projection_output_dim: int
    max_seq_len: int
    verified_at_timestamp: str
    is_audit_verified: bool
    manifest_signature: str  # Kept as alias for provenance_digest_sha256
    provenance_digest_sha256: str = ""
    is_digital_signature_verified: bool = False  # Explicitly False: this is an unsigned cryptographic SHA-256 digest
    provenance_digest_type: str = "SHA256_COMPOSITE_INTEGRITY_DIGEST"

    def __post_init__(self):
        if not self.provenance_digest_sha256 and self.manifest_signature:
            self.provenance_digest_sha256 = self.manifest_signature
        elif not self.manifest_signature and self.provenance_digest_sha256:
            self.manifest_signature = self.provenance_digest_sha256

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "license": self.license,
            "weights_sha256": self.weights_sha256,
            "vocab_sha256": self.vocab_sha256,
            "tokenizer_sha256": self.tokenizer_sha256,
            "hidden_dim": self.hidden_dim,
            "projection_version": self.projection_version,
            "projection_output_dim": self.projection_output_dim,
            "max_seq_len": self.max_seq_len,
            "verified_at_timestamp": self.verified_at_timestamp,
            "is_audit_verified": self.is_audit_verified,
            "manifest_signature": self.manifest_signature,
            "provenance_digest_sha256": self.provenance_digest_sha256,
            "is_digital_signature_verified": self.is_digital_signature_verified,
            "provenance_digest_type": self.provenance_digest_type,
        }


class ModelProvenanceVerifier:
    """
    Independent verifier ensuring only authentic pretrained models with valid provenance
    can be used in PRETRAINED mode.
    Guarantees:
    - Randomly initialized encoders (UNTRAINED_NEURAL_BASELINE) can NEVER pass Pretrained Verification.
    - Manual is_verified_pretrained=True boolean flag alone is rejected.
    - Model weights file SHA-256 and tokenizer files must strictly match trusted manifest.
    - Model license must be compliant (Apache-2.0, MIT).
    """

    @classmethod
    def verify_manifest(cls, manifest: PretrainedModelManifest) -> bool:
        if not manifest.model_id or not manifest.revision or not manifest.expected_weights_sha256:
            return False
        if manifest.license not in ("Apache-2.0", "MIT", "BSD-3-Clause"):
            return False
        return True

    @classmethod
    def verify_weights_file(cls, weights_path: str, expected_sha256: str) -> bool:
        if not os.path.exists(weights_path):
            return False
        hasher = hashlib.sha256()
        with open(weights_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest() == expected_sha256

    @classmethod
    def audit_model_and_tokenizer(
        cls,
        model: Any,
        tokenizer: Any,
        manifest: PretrainedModelManifest,
        snapshot_dir: Optional[str] = None,
    ) -> VerifiedModelProvenance:
        """
        Audits a model and tokenizer against an authoritative PretrainedModelManifest.
        Fails closed if:
        - Candidate model is untrained or randomly initialized (Anti-Spoofing).
        - Candidate manifest differs from authoritative manifest (Anti-Forgery).
        - snapshot_dir is missing, None, or not a physical directory on disk (Eliminates Fail-Open).
        - Required files (model.safetensors, config.json, vocab.txt, tokenizer.json) are missing.
        - File SHA-256 hashes mismatch authoritative manifest.
        - Model architecture, parameter count, or tensor shapes mismatch.
        - Model license is not an approved open-source license.
        """
        # 1. Anti-Spoofing Guard: Randomly initialized neural baseline can NEVER pass Pretrained Verification
        if getattr(model, "encoder_type", "") == "UNTRAINED_NEURAL_BASELINE":
            raise ValueError(
                f"Provenance Verification Error: Untrained neural baseline '{getattr(model, 'model_version', 'unknown')}' "
                f"is randomly initialized and cannot pass Pretrained Verification."
            )
        from stock_rl.evaluation.ablations import UntrainedTraditionalChineseNeuralEncoder
        if isinstance(model, UntrainedTraditionalChineseNeuralEncoder) or "Untrained" in model.__class__.__name__:
            raise ValueError(
                "Provenance Verification Error: UntrainedTraditionalChineseNeuralEncoder is an untrained "
                "baseline and cannot pass Pretrained Verification."
            )

        # 2. Authoritative Manifest Registry Check (Anti-Forge Guard)
        # Candidate models are NOT permitted to supply arbitrary expected hashes as authoritative evidence.
        if manifest.model_id not in AUTHORITATIVE_PRETRAINED_MANIFESTS:
            raise ValueError(
                f"Provenance Verification Error: Model '{manifest.model_id}' is not in the authoritative "
                f"trusted model registry. Only vetted models with pre-approved cryptographic manifests are accepted."
            )
        auth_manifest = AUTHORITATIVE_PRETRAINED_MANIFESTS[manifest.model_id]
        if manifest.expected_weights_sha256 != auth_manifest.expected_weights_sha256:
            raise ValueError(
                f"Provenance Verification Error: Forged manifest detected! Candidate expected_weights_sha256 "
                f"'{manifest.expected_weights_sha256}' does not match authoritative registry '{auth_manifest.expected_weights_sha256}'."
            )
        if manifest.revision != auth_manifest.revision or manifest.license != auth_manifest.license:
            raise ValueError(
                "Provenance Verification Error: Forged manifest parameters! Candidate revision/license differs from authoritative registry."
            )

        # 3. License verification
        if auth_manifest.license not in ("Apache-2.0", "MIT", "BSD-3-Clause"):
            raise ValueError(
                f"Provenance Verification Error: License '{auth_manifest.license}' is not an approved open-source license."
            )

        # 4. Snapshot Directory Requirement (Eliminate Fail-Open)
        if not snapshot_dir or not os.path.isdir(snapshot_dir):
            raise ValueError(
                f"Provenance Verification Error: snapshot_dir '{snapshot_dir}' is missing or not a directory. "
                f"Cannot verify model provenance without reading physical files from disk (Fail Closed)."
            )

        # 5. Required Files Existence Check
        weights_file = os.path.join(snapshot_dir, "model.safetensors")
        if not os.path.exists(weights_file):
            weights_file = os.path.join(snapshot_dir, "pytorch_model.bin")
        if not os.path.exists(weights_file):
            raise FileNotFoundError(
                f"Provenance Verification Error: Missing weights file ('model.safetensors' or 'pytorch_model.bin') in '{snapshot_dir}'."
            )

        vocab_file = os.path.join(snapshot_dir, "vocab.txt")
        if not os.path.exists(vocab_file):
            raise FileNotFoundError(f"Provenance Verification Error: Missing vocab file 'vocab.txt' in '{snapshot_dir}'.")

        tok_file = os.path.join(snapshot_dir, "tokenizer.json")
        if not os.path.exists(tok_file):
            raise FileNotFoundError(f"Provenance Verification Error: Missing tokenizer file 'tokenizer.json' in '{snapshot_dir}'.")

        config_file = os.path.join(snapshot_dir, "config.json")
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Provenance Verification Error: Missing config file 'config.json' in '{snapshot_dir}'.")

        # 6. Physical Hash Calculation & Validation
        hasher = hashlib.sha256()
        with open(weights_file, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        weights_sha = hasher.hexdigest()
        if weights_sha != auth_manifest.expected_weights_sha256:
            raise ValueError(
                f"Provenance Verification Error: Weights SHA-256 mismatch! "
                f"Expected {auth_manifest.expected_weights_sha256}, got {weights_sha}."
            )

        with open(vocab_file, "rb") as f:
            vocab_sha = hashlib.sha256(f.read()).hexdigest()
        if auth_manifest.expected_vocab_sha256 and vocab_sha != auth_manifest.expected_vocab_sha256:
            raise ValueError(
                f"Provenance Verification Error: Vocab SHA-256 mismatch! "
                f"Expected {auth_manifest.expected_vocab_sha256}, got {vocab_sha}."
            )

        with open(tok_file, "rb") as f:
            tok_sha = hashlib.sha256(f.read()).hexdigest()
        if auth_manifest.expected_tokenizer_sha256 and tok_sha != auth_manifest.expected_tokenizer_sha256:
            raise ValueError(
                f"Provenance Verification Error: Tokenizer SHA-256 mismatch! "
                f"Expected {auth_manifest.expected_tokenizer_sha256}, got {tok_sha}."
            )

        # 7. Model Architecture & Parameter Shapes Verification
        if not hasattr(model, "config"):
            raise ValueError("Provenance Verification Error: Candidate model lacks 'config' attribute.")
        cfg = model.config
        if getattr(cfg, "hidden_size", None) != auth_manifest.hidden_dim:
            raise ValueError(
                f"Provenance Verification Error: Model hidden_size {getattr(cfg, 'hidden_size', None)} "
                f"does not match manifest hidden_dim {auth_manifest.hidden_dim}."
            )
        if getattr(cfg, "vocab_size", None) != 21128:
            raise ValueError(
                f"Provenance Verification Error: Model vocab_size {getattr(cfg, 'vocab_size', None)} "
                f"does not match expected Chinese BERT vocab size (21128)."
            )

        # Check total parameter count (~24M parameters for bge-small-zh)
        total_params = sum(p.numel() for p in model.parameters()) if hasattr(model, "parameters") else 0
        if not (23_000_000 <= total_params <= 25_000_000):
            raise ValueError(
                f"Provenance Verification Error: Model parameter count ({total_params}) outside expected "
                f"range [23M, 25M] for BGE-small architecture."
            )

        # Check word embedding shape
        if hasattr(model, "embeddings") and hasattr(model.embeddings, "word_embeddings"):
            emb_weight = model.embeddings.word_embeddings.weight
            if emb_weight.shape != (21128, auth_manifest.hidden_dim):
                raise ValueError(
                    f"Provenance Verification Error: Word embeddings shape {emb_weight.shape} "
                    f"does not match expected (21128, {auth_manifest.hidden_dim})."
                )

        # 8. Cryptographic Composite Digest (Explicit Distinction from Digital Signature)
        # Note: This is an unsigned SHA-256 cryptographic digest of provenance records, NOT an asymmetric digital signature.
        digest_payload = f"{auth_manifest.model_id}|{auth_manifest.revision}|{weights_sha}|{vocab_sha}|{tok_sha}|{auth_manifest.license}"
        composite_digest = hashlib.sha256(digest_payload.encode("utf-8")).hexdigest()

        now_str = datetime.datetime.now(datetime.timezone.utc).isoformat()

        return VerifiedModelProvenance(
            model_id=auth_manifest.model_id,
            revision=auth_manifest.revision,
            license=auth_manifest.license,
            weights_sha256=weights_sha,
            vocab_sha256=vocab_sha,
            tokenizer_sha256=tok_sha,
            hidden_dim=auth_manifest.hidden_dim,
            projection_version=auth_manifest.projection_version,
            projection_output_dim=auth_manifest.projection_output_dim,
            max_seq_len=auth_manifest.max_seq_len,
            verified_at_timestamp=now_str,
            is_audit_verified=True,
            manifest_signature=composite_digest,
            provenance_digest_sha256=composite_digest,
            is_digital_signature_verified=False,
            provenance_digest_type="SHA256_COMPOSITE_INTEGRITY_DIGEST",
        )

    @classmethod
    def is_verified_engine(cls, engine: Any) -> bool:
        """
        Validates that a TextEmbeddingEngine holds legitimate, cryptographically verified pretrained provenance.
        Cross-checks against the fixed authoritative manifest registry.
        Rejects manual is_verified_pretrained=True boolean spoofing without valid manifest.
        """
        if not hasattr(engine, "embedding_type") or engine.embedding_type != TextEmbeddingType.PRETRAINED:
            return False
        if not getattr(engine, "is_verified_pretrained", False):
            return False
        prov = getattr(engine, "provenance_certificate", None)
        if prov is None or not isinstance(prov, VerifiedModelProvenance):
            return False
        if not prov.is_audit_verified or not getattr(prov, "provenance_digest_sha256", None):
            return False

        # Compare against fixed authoritative registry
        auth_manifest = AUTHORITATIVE_PRETRAINED_MANIFESTS.get(prov.model_id)
        if auth_manifest is None:
            return False
        if prov.revision != auth_manifest.revision:
            return False
        if prov.license != auth_manifest.license:
            return False
        if prov.weights_sha256 != auth_manifest.expected_weights_sha256:
            return False
        if prov.vocab_sha256 != auth_manifest.expected_vocab_sha256:
            return False
        if prov.tokenizer_sha256 != auth_manifest.expected_tokenizer_sha256:
            return False
        if prov.hidden_dim != auth_manifest.hidden_dim:
            return False
        if prov.projection_output_dim != auth_manifest.projection_output_dim:
            return False

        # Verify composite integrity digest
        digest_payload = f"{prov.model_id}|{prov.revision}|{prov.weights_sha256}|{prov.vocab_sha256}|{prov.tokenizer_sha256}|{prov.license}"
        expected_digest = hashlib.sha256(digest_payload.encode("utf-8")).hexdigest()
        return prov.provenance_digest_sha256 == expected_digest


class GenuinePretrainedSemanticEncoder(nn.Module):
    """
    Genuine Pretrained Semantic Encoder conforming to V8.8 specifications.
    Wraps genuine pretrained Transformer (e.g. BAAI/bge-small-zh-v1.5) with:
    - CLS token extraction & normalization
    - Deterministic orthogonal QR projection to authoritative 128D EVENT_TEXT_DIM
    - Deterministic inference: model.eval(), torch.inference_mode(), sequence truncation
    - Complete provenance traceability
    """
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        provenance: VerifiedModelProvenance,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.provenance = provenance
        self.output_dim = provenance.projection_output_dim  # 128
        self.hidden_dim = provenance.hidden_dim              # 512
        self.max_seq_len = provenance.max_seq_len            # 128
        self.model_version = provenance.model_id
        self.is_verified_pretrained = True
        self.weights_sha256 = provenance.weights_sha256

        # Build deterministic orthogonal QR projection matrix (512 -> 128)
        h = hashlib.sha256(f"PROJ_{provenance.model_id}_{self.hidden_dim}_{self.output_dim}".encode("utf-8")).digest()
        seed = int.from_bytes(h[:8], byteorder="big") % (2**32)
        rng = np.random.default_rng(seed)
        rand_mat = rng.standard_normal((self.hidden_dim, self.output_dim), dtype=np.float32)
        q, _ = np.linalg.qr(rand_mat)
        self.register_buffer("proj_matrix", torch.from_numpy(q[:, :self.output_dim]))

        if hasattr(self.model, "eval"):
            self.model.eval()

    def get_provenance_metadata(self) -> Dict[str, Any]:
        return self.provenance.to_dict()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass producing normalized 128D embedding.
        """
        if hasattr(self.model, "eval"):
            self.model.eval()

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if attention_mask is not None and attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)

        # Enforce max sequence length truncation
        if input_ids.shape[1] > self.max_seq_len:
            input_ids = input_ids[:, :self.max_seq_len]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :self.max_seq_len]

        if attention_mask is not None:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            outputs = self.model(input_ids=input_ids)

        # BGE recommendation: CLS token embedding (outputs[0][:, 0])
        cls_emb = outputs[0][:, 0]  # shape: (batch_size, hidden_dim)

        # Normalize 512D
        cls_norm = torch.nn.functional.normalize(cls_emb, p=2, dim=1)

        # Project to 128D
        proj_matrix = self.proj_matrix.to(cls_norm.device, dtype=cls_norm.dtype)
        projected = torch.matmul(cls_norm, proj_matrix)

        # Normalize 128D
        out = torch.nn.functional.normalize(projected, p=2, dim=1)
        return out

    def embed_text(self, text: str) -> np.ndarray:
        """
        Embeds a single string into a normalized 128D numpy float32 vector.
        """
        clean = (text or "").strip()
        if not clean:
            v = np.zeros(self.output_dim, dtype=np.float32)
            v[0] = 1.0
            return v

        if hasattr(self.model, "eval"):
            self.model.eval()

        with torch.inference_mode():
            inputs = self.tokenizer(
                clean,
                padding=True,
                truncation=True,
                max_length=self.max_seq_len,
                return_tensors="pt"
            )
            v = self.forward(input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask"))
            emb = v.squeeze(0).cpu().numpy().astype(np.float32)
            norm = float(np.linalg.norm(emb))
            return (emb / norm).astype(np.float32) if norm > 1e-6 else np.zeros(self.output_dim, dtype=np.float32)


def load_trusted_pretrained_embedding_engine(
    manifest: PretrainedModelManifest = BGE_SMALL_ZH_MANIFEST,
    local_files_only: bool = True,
    cache_dir: Optional[str] = None,
) -> Optional[TextEmbeddingEngine]:
    """
    Loads and cryptographically verifies a genuine pretrained embedding model per V8.8 specifications.
    Attempts local cache first.
    If model is unavailable or verification fails, returns None (reporting B2 as Pending).
    """
    try:
        from transformers import AutoTokenizer, AutoModel
    except ImportError:
        return None

    # Check local snapshot directory in huggingface cache
    snapshot_dir = None
    default_cache = cache_dir or os.path.expanduser("~/.cache/huggingface/hub")
    repo_folder = f"models--{manifest.model_id.replace('/', '--')}"
    candidate_dir = os.path.join(default_cache, repo_folder, "snapshots", manifest.revision)
    if os.path.isdir(candidate_dir):
        snapshot_dir = candidate_dir

    if not snapshot_dir:
        # Fails closed if local snapshot directory is missing
        return None

    try:
        tok = AutoTokenizer.from_pretrained(snapshot_dir, local_files_only=True)
        model = AutoModel.from_pretrained(snapshot_dir, local_files_only=True)

        # Perform strict provenance audit
        prov = ModelProvenanceVerifier.audit_model_and_tokenizer(
            model=model,
            tokenizer=tok,
            manifest=manifest,
            snapshot_dir=snapshot_dir,
        )

        encoder = GenuinePretrainedSemanticEncoder(
            model=model,
            tokenizer=tok,
            provenance=prov,
        )

        engine = TextEmbeddingEngine(
            embedding_type=TextEmbeddingType.PRETRAINED,
            model_version=manifest.model_id,
            dim=manifest.projection_output_dim,
            model_weights=encoder,
            tokenizer=tok,
            is_verified_pretrained=True,
            weights_sha256=prov.weights_sha256,
            tokenizer_version="bert-tokenizer-bge-v1.5",
            projection_version=manifest.projection_version,
            max_seq_len=manifest.max_seq_len,
        )
        engine.provenance_certificate = prov
        return engine
    except Exception as e:
        # Fails closed
        return None


# ==============================================================================
# P1: Projection Robustness Evaluation Benchmark Dataset & Metric Evaluator
# ==============================================================================

PROJECTION_ROBUSTNESS_BENCHMARK_PAIRS: List[Dict[str, str]] = [
    # 1. Same-topic pairs (5 pairs)
    {
        "category": "same_topic",
        "pair_id": "ST-01",
        "topic": "TSMC CapEx & Advanced Packaging Expansion",
        "text_a": "台積電董事會拍板12吋晶圓廠投資案，預計投資新台幣千億元擴充先進製程產能。",
        "text_b": "台積電加速CoWoS先進封裝擴產，中科新廠動工因應AI晶片強勁需求。",
    },
    {
        "category": "same_topic",
        "pair_id": "ST-02",
        "topic": "Evergreen & Shipping Freight Rates",
        "text_a": "長榮海運獲利創高，紅海危機帶動SCFI運價大漲，董事會決議配發高額現金股利。",
        "text_b": "陽明海運公布最新營收，受惠於歐洲航線運價翻倍，單月獲利呈現跳躍式成長。",
    },
    {
        "category": "same_topic",
        "pair_id": "ST-03",
        "topic": "Fubon & Cathay Financial Earnings",
        "text_a": "富邦金控自結全年累計稅後淨利突破千億元，蟬聯金控每股獲利王寶座。",
        "text_b": "國泰金控公布單月稅後淨利大幅增長，壽險與銀行雙引擎推升金控整體獲利表現。",
    },
    {
        "category": "same_topic",
        "pair_id": "ST-04",
        "topic": "MediaTek AI Mobile Processor",
        "text_a": "聯發科發表新一代旗艦天璣5G AI行動晶片，運算效能與功耗表現大幅躍進。",
        "text_b": "聯發科旗艦處理器獲全球一線智慧型手機大廠導入，AI邊緣運算晶片出貨強勁。",
    },
    {
        "category": "same_topic",
        "pair_id": "ST-05",
        "topic": "Foxconn AI Server & Global Supply Chain",
        "text_a": "鴻海擴大全球AI伺服器出貨量能，北美與墨西哥產線全面滿載運轉。",
        "text_b": "鴻海科技集團加速布局電動車與AI高速運算領域，全球供應鏈垂直整合效益浮現。",
    },

    # 2. Paraphrases (5 pairs)
    {
        "category": "paraphrase",
        "pair_id": "PA-01",
        "topic": "TSMC Annual Revenue Guidance Upgrade",
        "text_a": "台積電法說會調升今年全年美元營收成長預估至兩成以上。",
        "text_b": "台積電在法人說明會上修正全年營收展望，預期美元營收將實現超過20%年增長。",
    },
    {
        "category": "paraphrase",
        "pair_id": "PA-02",
        "topic": "Evergreen Board Cash Dividend Announcement",
        "text_a": "長榮海運宣布董事會通過每股配發現金股利10元。",
        "text_b": "長榮公告最新股利政策，董事會決議每股配息十元現金。",
    },
    {
        "category": "paraphrase",
        "pair_id": "PA-03",
        "topic": "MediaTek Dimensity GenAI Launch",
        "text_a": "聯發科推出天璣系列最新處理器，搶攻高階生成式AI手機市場。",
        "text_b": "聯發科發表全新天璣旗艦晶片，積極布局旗艦智慧手機生成式人工智慧應用。",
    },
    {
        "category": "paraphrase",
        "pair_id": "PA-04",
        "topic": "Fubon Financial Record High Profit",
        "text_a": "富邦金累計稅後淨利創歷史同期新高，資產品質維持穩健。",
        "text_b": "富邦金自結淨利改寫歷年同期最佳紀錄，各子公司資產結構健康。",
    },
    {
        "category": "paraphrase",
        "pair_id": "PA-05",
        "topic": "Largan Precision Gross Margin Recovery",
        "text_a": "大立光公布上季毛利率回升，高階潛望鏡鏡頭拉貨動能轉強。",
        "text_b": "大立光財報顯示單季營業毛利增溫，旗艦手機高階鏡頭需求帶動出貨。",
    },

    # 3. Cross-industry pairs (5 pairs)
    {
        "category": "cross_industry",
        "pair_id": "CI-01",
        "topic": "TSMC (Semiconductor) vs Evergreen (Shipping)",
        "text_a": "台積電中科2奈米新廠建置順利，預計明年進入量產階段。",
        "text_b": "長榮海運新造兩萬四千TEU超大型貨櫃輪交船，全面投入歐美長程主力航線。",
    },
    {
        "category": "cross_industry",
        "pair_id": "CI-02",
        "topic": "Fubon Financial (Banking) vs China Steel (Steel)",
        "text_a": "富邦金控擴大海外銀行綠色融資規模，首季淨利息收入穩健增長。",
        "text_b": "中鋼宣布開出最新月盤盤價，全球高爐鋼廠減產帶動熱軋鋼捲價格走揚。",
    },
    {
        "category": "cross_industry",
        "pair_id": "CI-03",
        "topic": "MediaTek (IC Design) vs Formosa Plastics (Petrochemical)",
        "text_a": "聯發科跨足客製化晶片ASIC與車用電子市場，擴大研發高階先進製程IP。",
        "text_b": "台塑四寶公布最新營收表現，石化下游需求疲弱導致乙烯丙烯利差收斂。",
    },
    {
        "category": "cross_industry",
        "pair_id": "CI-04",
        "topic": "Chunghwa Telecom (Telecom) vs E Ink (E-Paper)",
        "text_a": "中華電信積極推動5G專網與低軌衛星地面站建置，企業客戶營收持續增長。",
        "text_b": "元太科技彩色電子紙新產能開出，電子書閱讀器與智慧零售標籤出貨攀升。",
    },
    {
        "category": "cross_industry",
        "pair_id": "CI-05",
        "topic": "Uni-President (Food Retail) vs Delta Electronics (Power Electronics)",
        "text_a": "統一企業擴大亞洲生活品牌布局，超商與超大型物流園區陸續完工啟用。",
        "text_b": "台達電深耕伺服器電源與微電網儲能系統，工業自動化解決方案接單暢旺。",
    },

    # 4. Negation pairs (5 pairs)
    {
        "category": "negation",
        "pair_id": "NG-01",
        "topic": "TSMC Optimistic Guidance vs Slump & Order Cuts",
        "text_a": "台積電先進製程營收創歷史新高，全年展望樂觀且產能利用率持續回升。",
        "text_b": "台積電先進製程遭遇客戶砍單，下修全年展望且產能利用率驟跌至低谷。",
    },
    {
        "category": "negation",
        "pair_id": "NG-02",
        "topic": "Evergreen Surging Profit vs Loss & Collapse",
        "text_a": "長榮海運獲利爆發，運價大漲推升營業利益改寫歷史次高。",
        "text_b": "長榮海運營業利益重挫，全球運價崩跌導致單季營運轉為虧損。",
    },
    {
        "category": "negation",
        "pair_id": "NG-03",
        "topic": "Financial Sector Record Profit vs Severe Impairment",
        "text_a": "金控獲利創歷年新高，海外投資評價利得豐厚且逾放比降至新低。",
        "text_b": "金控獲利大幅衰退，海外投資提列巨額減損且壞帳風險急速升高。",
    },
    {
        "category": "negation",
        "pair_id": "NG-04",
        "topic": "Chip Certification Pass vs Architecture Defect Failure",
        "text_a": "旗艦AI晶片測試順利通過認證，良率超乎預期並啟動大規模量產交付。",
        "text_b": "旗艦AI晶片因重大架構缺陷未能通過認證，生產良率過低被迫延期量產。",
    },
    {
        "category": "negation",
        "pair_id": "NG-05",
        "topic": "Expansion & Massive Hiring vs Plant Closure & Layoffs",
        "text_a": "科技大廠宣布加碼百億元擴建新廠，全面招募數千名研發與製程工程師。",
        "text_b": "科技大廠宣布關閉現有老舊廠區，大幅裁撤員工並縮減所有資本支出計劃。",
    },

    # 5. Unrelated negative controls (5 pairs)
    {
        "category": "negative_control",
        "pair_id": "NC-01",
        "topic": "TSMC CapEx vs Cold Front Weather Alert",
        "text_a": "台積電董事會拍板12吋晶圓廠投資案，預計投資新台幣千億元擴充先進製程產能。",
        "text_b": "中央氣象署發布低溫特報，強烈大陸冷氣團來襲，各地氣溫驟降並出現局部短暫雨。",
    },
    {
        "category": "negative_control",
        "pair_id": "NC-02",
        "topic": "Fubon Net Profit vs Baseball WBC Win",
        "text_a": "富邦金自結全年累計獲利突破千億元，每股獲利蟬聯國內金控冠軍。",
        "text_b": "世界棒球經典賽八強淘汰賽今晚開打，中華隊打線全員安打擊敗強敵挺進準決賽。",
    },
    {
        "category": "negative_control",
        "pair_id": "NC-03",
        "topic": "Evergreen Dividend vs Highway Holiday Traffic",
        "text_a": "長榮海運獲利創高，紅海危機帶動SCFI運價大漲，董事會決議配發高額現金股利。",
        "text_b": "高速公路局公布春節連假交通疏導措施，初一至初五國道多個路段實施高乘載管制。",
    },
    {
        "category": "negative_control",
        "pair_id": "NC-04",
        "topic": "MediaTek 5G AI vs Braised Beef Noodle Recipe",
        "text_a": "聯發科發表新一代旗艦天璣5G AI行動晶片，運算效能大幅躍進。",
        "text_b": "傳統家常紅燒牛肉麵烹調秘訣，先將牛腱肉慢火煎至金黃再加入豆瓣醬與中藥包燉煮。",
    },
    {
        "category": "negative_control",
        "pair_id": "NC-05",
        "topic": "Foxconn Server Shipment vs Night Sky Meteor Shower",
        "text_a": "鴻海擴大全球AI伺服器出貨量能，北美與墨西哥產線全面滿載運轉。",
        "text_b": "台北市立天文科學教育館表示，本週末夜空將出現年度最大流星雨，每小時可見數十顆流星。",
    },
]


def evaluate_projection_robustness(
    encoder: GenuinePretrainedSemanticEncoder,
    pairs: Optional[List[Dict[str, str]]] = None,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Evaluates semantic preservation and distortion between raw 512D pretrained embeddings
    and projected 128D embeddings across benchmark pairs conforming to V8.8 P1 specifications.
    Reports:
    - Mean Absolute Cosine Distortion
    - 95th Percentile Distortion
    - Maximum Distortion
    - Similarity Ranking Consistency (Spearman rho & Kendall tau)
    - Per-category breakdown
    """
    import scipy.stats

    test_pairs = pairs or PROJECTION_ROBUSTNESS_BENCHMARK_PAIRS
    raw_model = encoder.model
    tokenizer = encoder.tokenizer
    max_len = encoder.max_seq_len

    if hasattr(raw_model, "eval"):
        raw_model.eval()

    def get_512d_cls(text: str) -> np.ndarray:
        with torch.inference_mode():
            inp = tokenizer(text, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
            out = raw_model(input_ids=inp["input_ids"], attention_mask=inp.get("attention_mask"))
            cls_vec = out[0][:, 0].squeeze(0).cpu().numpy().astype(np.float32)
            norm = float(np.linalg.norm(cls_vec))
            return (cls_vec / norm) if norm > 1e-6 else np.zeros(512, dtype=np.float32)

    sims_512 = []
    sims_128 = []
    distortions = []
    detailed_results = []
    categories: Dict[str, Dict[str, List[float]]] = {}

    for item in test_pairs:
        cat = item["category"]
        text_a = item["text_a"]
        text_b = item["text_b"]

        # 512D
        v512_a = get_512d_cls(text_a)
        v512_b = get_512d_cls(text_b)
        s512 = float(np.dot(v512_a, v512_b))

        # 128D
        v128_a = encoder.embed_text(text_a)
        v128_b = encoder.embed_text(text_b)
        s128 = float(np.dot(v128_a, v128_b))

        dist = abs(s512 - s128)

        sims_512.append(s512)
        sims_128.append(s128)
        distortions.append(dist)

        detailed_results.append({
            "pair_id": item.get("pair_id", ""),
            "category": cat,
            "topic": item.get("topic", ""),
            "sim_512": round(s512, 6),
            "sim_128": round(s128, 6),
            "distortion": round(dist, 6),
        })

        if cat not in categories:
            categories[cat] = {"sim_512": [], "sim_128": [], "dist": []}
        categories[cat]["sim_512"].append(s512)
        categories[cat]["sim_128"].append(s128)
        categories[cat]["dist"].append(dist)

    sims_512_arr = np.array(sims_512, dtype=np.float64)
    sims_128_arr = np.array(sims_128, dtype=np.float64)
    dist_arr = np.array(distortions, dtype=np.float64)

    mean_dist = float(np.mean(dist_arr))
    p95_dist = float(np.percentile(dist_arr, 95))
    max_dist = float(np.max(dist_arr))

    spearman_res = scipy.stats.spearmanr(sims_512_arr, sims_128_arr)
    spearman_rho = float(spearman_res.statistic)
    kendall_res = scipy.stats.kendalltau(sims_512_arr, sims_128_arr)
    kendall_tau = float(kendall_res.statistic)

    cat_breakdown = {}
    for cat, vals in categories.items():
        cat_breakdown[cat] = {
            "count": len(vals["dist"]),
            "mean_sim_512": round(float(np.mean(vals["sim_512"])), 6),
            "mean_sim_128": round(float(np.mean(vals["sim_128"])), 6),
            "mean_distortion": round(float(np.mean(vals["dist"])), 6),
            "max_distortion": round(float(np.max(vals["dist"])), 6),
        }

    return {
        "num_pairs": len(test_pairs),
        "random_seed": random_seed,
        "model_id": getattr(encoder.provenance, "model_id", "BAAI/bge-small-zh-v1.5"),
        "model_revision": getattr(encoder.provenance, "revision", ""),
        "projection_version": getattr(encoder.provenance, "projection_version", "orthogonal-qr-v1.0"),
        "mean_absolute_cosine_distortion": round(mean_dist, 6),
        "percentile_95_distortion": round(p95_dist, 6),
        "maximum_distortion": round(max_dist, 6),
        "spearman_rank_correlation": round(spearman_rho, 6),
        "kendall_tau": round(kendall_tau, 6),
        "category_breakdown": cat_breakdown,
        "pairs_detail": detailed_results,
        "test_limitations": (
            "Empirical evaluation based on 25 curated Traditional Chinese financial headline pairs across "
            "5 semantic categories. Represents short-text heading semantic preservation; multi-sentence and "
            "long-form documents require hierarchical chunking in future stages."
        ),
    }

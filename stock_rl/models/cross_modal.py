"""
Cross-Modal Attention Fusion Module V8.0
Integrates Market Tokens (Price, Technical, Institutional, Macro, Context) and Event Token.
"""

import torch
import torch.nn as nn
from stock_rl.config.schema import MULTIMODAL_FUSION_DIM


class CrossModalAttention(nn.Module):
    """
    Cross-Modal Attention across Price, Technical, Institutional, Macro, Context, and Event tokens.
    Fuses all modalities into a unified 128-dimensional embedding.
    """
    def __init__(
        self,
        d_model: int = MULTIMODAL_FUSION_DIM,  # 128
        num_heads: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

        # Learnable pooling / aggregation query
        self.fusion_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pool_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        market_tokens: torch.Tensor,  # [B, 5, 128]
        event_token: torch.Tensor     # [B, 1, 128]
    ) -> torch.Tensor:
        """
        Returns:
            fused_representation: [B, 128]
        """
        B = market_tokens.shape[0]

        # Combine tokens: [B, 6, 128]
        all_tokens = torch.cat([market_tokens, event_token], dim=1)

        # Cross-modal self-attention
        sa_out, _ = self.self_attn(all_tokens, all_tokens, all_tokens)
        tokens_normed = self.norm1(all_tokens + sa_out)

        # Feedforward
        ffn_out = self.ffn(tokens_normed)
        tokens_processed = self.norm2(tokens_normed + ffn_out)  # [B, 6, 128]

        # Aggregate across modality tokens into a single 128-d vector
        query = self.fusion_query.expand(B, 1, -1)
        pooled, _ = self.pool_attn(query, tokens_processed, tokens_processed)
        fused = self.final_norm((query + pooled).squeeze(1))  # [B, 128]

        return fused

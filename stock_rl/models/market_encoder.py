"""
Market Feature Encoders for Price, Technical, Institutional, and Macro Sequences
Specification V8.0
"""

import torch
import torch.nn as nn
from stock_rl.config.schema import (
    PRICE_DIM,
    TECHNICAL_DIM,
    INSTITUTIONAL_DIM,
    MACRO_DIM,
    STOCK_CONTEXT_DIM,
)


class SequenceEncoder(nn.Module):
    """
    Bidirectional/Unidirectional GRU encoder mapping [B, T, D_in] to [B, D_out].
    """
    def __init__(self, in_dim: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D_in]
        out, _ = self.gru(x)
        last_hidden = out[:, -1, :]  # [B, hidden_dim]
        return self.proj(last_hidden)


class MarketEncoders(nn.Module):
    """
    Processes Price, Technical, Institutional, Macro sequences and Stock Context
    into a set of discrete modality tokens [B, 5, 128].
    """
    def __init__(self, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.price_encoder = SequenceEncoder(PRICE_DIM, hidden_dim, num_layers, dropout)
        self.tech_encoder = SequenceEncoder(TECHNICAL_DIM, hidden_dim, num_layers, dropout)
        self.inst_encoder = SequenceEncoder(INSTITUTIONAL_DIM, hidden_dim, num_layers, dropout)
        self.macro_encoder = SequenceEncoder(MACRO_DIM, hidden_dim, num_layers, dropout)

        self.context_encoder = nn.Sequential(
            nn.Linear(STOCK_CONTEXT_DIM, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(
        self,
        price: torch.Tensor,
        tech: torch.Tensor,
        inst: torch.Tensor,
        macro: torch.Tensor,
        context: torch.Tensor
    ) -> torch.Tensor:
        """
        Returns stacked modality tokens of shape [B, 5, hidden_dim].
        Tokens order: [Price, Tech, Inst, Macro, Context]
        """
        p_tok = self.price_encoder(price).unsqueeze(1)    # [B, 1, 128]
        t_tok = self.tech_encoder(tech).unsqueeze(1)      # [B, 1, 128]
        i_tok = self.inst_encoder(inst).unsqueeze(1)      # [B, 1, 128]
        m_tok = self.macro_encoder(macro).unsqueeze(1)    # [B, 1, 128]
        c_tok = self.context_encoder(context).unsqueeze(1)# [B, 1, 128]

        return torch.cat([p_tok, t_tok, i_tok, m_tok, c_tok], dim=1)  # [B, 5, 128]

"""
Integrated Multimodal Stock RL Model V8.0
Unifies MarketEncoders, EventModule, CrossModalAttention, MultiTaskHeads, and DuelingDQN.
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn

from stock_rl.config.schema import (
    MULTIMODAL_FUSION_DIM,
    RL_STATE_DIM,
    POSITION_STATE_DIM,
)
from stock_rl.data.synthetic_data import MultimodalBatch
from stock_rl.models.market_encoder import MarketEncoders
from stock_rl.models.event_module import EventModule
from stock_rl.models.cross_modal import CrossModalAttention
from stock_rl.models.multi_task_heads import MultiTaskHeads
from stock_rl.models.dueling_dqn import DuelingDQN


class StockRLFullModel(nn.Module):
    """
    End-to-end neural system for point-in-time event-driven stock trading RL.
    """
    def __init__(self):
        super().__init__()
        self.market_encoders = MarketEncoders()
        self.event_module = EventModule()
        self.cross_modal_fusion = CrossModalAttention()
        self.multi_task_heads = MultiTaskHeads()
        self.dqn = DuelingDQN()

    def encode_multimodal_state(
        self,
        batch: MultimodalBatch
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Runs encoders and fusion.
        Returns:
            fused_representation: [B, 128]
            rl_state: [B, 132]
            projected_events: [B, 40, 128]
            context_token: [B, 128]
        """
        # 1. Market sequences -> [B, 5, 128]
        # Order: [Price, Tech, Inst, Macro, Context]
        market_tokens = self.market_encoders(
            price=batch.price,
            tech=batch.technical,
            inst=batch.institutional,
            macro=batch.macro,
            context=batch.stock_context,
        )
        context_token = market_tokens[:, 4, :]  # [B, 128]

        # 2. Event tokens: [B, 40, 128] and [B, 1, 128]
        # Combine recent (32) and active (8) text embeddings -> [B, 40, 128]
        event_text = torch.cat([batch.recent_event_emb, batch.active_event_emb], dim=1)
        projected_events, aggregated_event_token = self.event_module(
            event_text=event_text,
            event_meta=batch.event_metadata,
            event_rel=batch.event_stock_rel,
            event_mask=batch.event_mask,
        )

        # 3. Cross-Modal Fusion across 5 market tokens + 1 aggregated event token -> [B, 128]
        fused_rep = self.cross_modal_fusion(
            market_tokens=market_tokens,
            event_token=aggregated_event_token,
        )

        # 4. Construct 132-d RL state = [Fused (128) + Position (4)]
        rl_state = torch.cat([fused_rep, batch.position], dim=-1)  # [B, 132]

        return fused_rep, rl_state, projected_events, context_token

    def forward(
        self,
        batch: MultimodalBatch
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """
        Full forward pass returning Q-values, auxiliary predictions, and RL state.
        Returns:
            q_values: [B, 2]
            predictions: dict of multi-task outputs
            rl_state: [B, 132]
        """
        fused_rep, rl_state, projected_events, context_token = self.encode_multimodal_state(batch)

        # Multi-task auxiliary predictions
        predictions = self.multi_task_heads(
            projected_events=projected_events,
            fused_representation=fused_rep,
            context_token=context_token,
        )

        # Dueling DDQN Q-values: [B, 2]
        q_values = self.dqn(rl_state)

        return q_values, predictions, rl_state

    def freeze_encoders(self):
        """Freezes representation encoders for Stage 2 RL training."""
        for p in self.market_encoders.parameters():
            p.requires_grad = False
        for p in self.event_module.parameters():
            p.requires_grad = False
        for p in self.cross_modal_fusion.parameters():
            p.requires_grad = False
        for p in self.multi_task_heads.parameters():
            p.requires_grad = False

    def unfreeze_for_stage3(self):
        """
        Selectively unfreezes Temporal Attention, Cross-Modal Fusion,
        and Event Projection for Stage 3 fine-tuning.
        """
        # Event projection and temporal attention
        for p in self.event_module.parameters():
            p.requires_grad = True
        # Cross-Modal Fusion
        for p in self.cross_modal_fusion.parameters():
            p.requires_grad = True
        # Auxiliary heads
        for p in self.multi_task_heads.parameters():
            p.requires_grad = True
        # DQN heads
        for p in self.dqn.parameters():
            p.requires_grad = True

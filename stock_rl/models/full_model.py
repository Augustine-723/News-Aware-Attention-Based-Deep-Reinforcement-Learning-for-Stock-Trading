"""
Integrated Multimodal Stock RL Model V8.0
Unifies MarketEncoders, EventModule (with Conditioned Temporal Attention),
CrossModalAttention, MultiTaskHeads, and DuelingDQN.
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn

from stock_rl.config.schema import (
    MULTIMODAL_FUSION_DIM,
    RL_STATE_DIM,
    POSITION_STATE_DIM,
    TOTAL_EVENT_SLOTS,
    EVENT_AGE_METADATA_INDEX,
    CURRENT_ACTIVE_SCHEMA_VERSION,
    get_event_age_index,
    get_feature_index,
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
        batch: MultimodalBatch,
        schema_version: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Runs encoders and fusion with Conditioned Temporal Attention.
        Returns:
            fused_representation: [B, 128]
            rl_state: [B, 132]
            projected_events: [B, 40, 128]
            context_token: [B, 128]
            prior_memory: [B, 128]
        """
        B = batch.price.shape[0]
        effective_schema = schema_version or getattr(batch, "schema_version", CURRENT_ACTIVE_SCHEMA_VERSION)
        age_idx = get_event_age_index(effective_schema)

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
        market_summary = market_tokens[:, 0:4, :].mean(dim=1)  # [B, 128]

        # 2. Event tokens: [B, 40, 128] and [B, 1, 128]
        # Combine recent (32) and active (8) text embeddings -> [B, 40, 128]
        event_text = torch.cat([batch.recent_event_emb, batch.active_event_emb], dim=1)

        # Conditioned Temporal Attention: q = Base + StockProj + MarketProj
        projected_events, aggregated_event_token = self.event_module(
            event_text=event_text,
            event_meta=batch.event_metadata,
            event_rel=batch.event_stock_rel,
            event_mask=batch.event_mask,
            stock_context=batch.stock_context,
            market_summary=market_summary,
            schema_version=effective_schema,
            event_age_idx=age_idx,
        )

        # 3. Cross-Modal Fusion across 5 market tokens + 1 aggregated event token -> [B, 128]
        fused_rep = self.cross_modal_fusion(
            market_tokens=market_tokens,
            event_token=aggregated_event_token,
        )

        # 4. Construct 132-d RL state = [Fused (128) + Position (4)]
        rl_state = torch.cat([fused_rep, batch.position], dim=-1)  # [B, 132]

        # 5. Prior event memory for Novelty comparison per V8.0:
        # a) active_projected_memory with valid mask (slots 32..39), falling back to NRE token if empty
        active_mask = batch.event_mask[:, 32:].unsqueeze(-1)  # [B, 8, 1]
        active_counts = active_mask.sum(dim=1)                # [B, 1]
        active_sum = (projected_events[:, 32:, :] * active_mask.float()).sum(dim=1)  # [B, 128]
        nre_token = self.event_module.nre_token.squeeze(1).expand(B, -1)              # [B, 128]
        active_projected_memory = torch.where(
            active_counts > 0,
            active_sum / torch.clamp(active_counts, min=1e-8),
            nre_token,
        )  # [B, 128]

        # b) Causal prior memory per event slot:
        # Each event's novelty must only compare against information known prior to its publication.
        # Dynamically retrieve event age index from schema without hardcoding.
        # Since age = T_decision - T_available, older events (published before event j) have age_k > age_j.
        event_ages = batch.event_metadata[:, :, age_idx]  # [B, 40]
        ages_k = event_ages.unsqueeze(1)            # [B, 1, 40]
        ages_j = event_ages.unsqueeze(2)            # [B, 40, 1]
        # Slot k was published strictly before slot j and is a valid event
        is_prior = (ages_k > ages_j) & batch.event_mask.unsqueeze(1)  # [B, 40, 40]
        prior_weights = is_prior.float()                              # [B, 40, 40]
        prior_counts = prior_weights.sum(dim=-1, keepdim=True)        # [B, 40, 1]
        prior_sum = torch.bmm(prior_weights, projected_events)        # [B, 40, 128]

        # Fallback to NRE / No-Prior-Memory token when no strictly prior event exists
        nre_token_prior = self.event_module.nre_token.squeeze(1).unsqueeze(1).expand(B, TOTAL_EVENT_SLOTS, -1)
        causal_prior_memory = torch.where(
            prior_counts > 0,
            prior_sum / torch.clamp(prior_counts, min=1e-8),
            nre_token_prior,
        )  # [B, 40, 128]

        return fused_rep, rl_state, projected_events, context_token, causal_prior_memory

    def forward(
        self,
        batch: MultimodalBatch,
        schema_version: Optional[str] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """
        Full forward pass returning Q-values, auxiliary predictions, and RL state.
        Returns:
            q_values: [B, 2]
            predictions: dict of multi-task outputs
            rl_state: [B, 132]
        """
        fused_rep, rl_state, projected_events, context_token, prior_memory = self.encode_multimodal_state(
            batch, schema_version=schema_version
        )

        # Multi-task auxiliary predictions with active memory comparison
        predictions = self.multi_task_heads(
            projected_events=projected_events,
            prior_memory=prior_memory,
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
        for p in self.dqn.parameters():
            p.requires_grad = True

    def unfreeze_for_stage3(self):
        """
        Selectively unfreezes Temporal Attention, Cross-Modal Fusion,
        and Event Projection for Stage 3 fine-tuning.
        """
        for p in self.event_module.parameters():
            p.requires_grad = True
        for p in self.cross_modal_fusion.parameters():
            p.requires_grad = True
        for p in self.multi_task_heads.parameters():
            p.requires_grad = True
        for p in self.dqn.parameters():
            p.requires_grad = True

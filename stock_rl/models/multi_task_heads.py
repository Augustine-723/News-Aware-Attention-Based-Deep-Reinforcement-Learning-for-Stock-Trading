"""
Multi-Task Supervised Heads V8.0
1. Event-Stock Relevance Head (128 -> 64 -> 1)
2. Information Novelty Head (256 -> 64 -> 1)
3. Multi-Horizon Excess Return Head (256 -> 128 -> 3)
4. Multi-Horizon Future Risk Head (256 -> 128 -> 3)
Includes Uncertainty-Weighted Multi-Task Loss.
"""

import math
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from stock_rl.config.schema import NUM_HORIZONS, TOTAL_EVENT_SLOTS


class MultiTaskHeads(nn.Module):
    """
    Supervised predictive heads operating on event representations and stock-date multimodal embeddings.
    """
    def __init__(self, event_dim: int = 128, fusion_dim: int = 128):
        super().__init__()

        # Task 1: Stock Relevance Head: 128 -> 64 -> 1
        self.relevance_head = nn.Sequential(
            nn.Linear(event_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        # Task 2: Information Novelty Head: 256 -> 64 -> 1
        # Evaluates novelty of an event given the stock-date context
        self.novelty_head = nn.Sequential(
            nn.Linear(event_dim * 2, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        # Task 3: Daily Return Head: 256 -> 128 -> 3
        # Operates strictly on stock-date representation (fused representation + context)
        self.return_head = nn.Sequential(
            nn.Linear(fusion_dim * 2, 128),
            nn.GELU(),
            nn.Linear(128, NUM_HORIZONS),
        )

        # Task 4: Daily Risk Head: 256 -> 128 -> 3
        # Predicts future return magnitude / volatility proxy across 1, 3, 5 days
        self.risk_head = nn.Sequential(
            nn.Linear(fusion_dim * 2, 128),
            nn.GELU(),
            nn.Linear(128, NUM_HORIZONS),
            nn.Softplus(),  # Risk proxy is strictly non-negative
        )

    def forward(
        self,
        projected_events: torch.Tensor,       # [B, 40, 128]
        fused_representation: torch.Tensor,   # [B, 128]
        context_token: torch.Tensor           # [B, 128]
    ) -> Dict[str, torch.Tensor]:
        """
        Returns predictions dict:
            relevance: [B, 40, 1]
            novelty: [B, 40, 1]
            returns: [B, 3]
            risk: [B, 3]
        """
        B, N, D = projected_events.shape

        # 1. Relevance prediction per event slot
        relevance_logits = self.relevance_head(projected_events)  # [B, 40, 1]

        # 2. Novelty prediction: event token concatenated with fused summary
        fused_expanded = fused_representation.unsqueeze(1).expand(-1, N, -1)  # [B, 40, 128]
        novelty_inputs = torch.cat([projected_events, fused_expanded], dim=-1)  # [B, 40, 256]
        novelty_logits = self.novelty_head(novelty_inputs)  # [B, 40, 1]

        # 3 & 4. Stock-Date Return and Risk predictions: fused + context = 256
        stock_date_rep = torch.cat([fused_representation, context_token], dim=-1)  # [B, 256]
        pred_returns = self.return_head(stock_date_rep)  # [B, 3]
        pred_risk = self.risk_head(stock_date_rep)        # [B, 3]

        return {
            "relevance_logits": relevance_logits,
            "novelty_logits": novelty_logits,
            "pred_returns": pred_returns,
            "pred_risk": pred_risk,
        }


class MultiTaskLoss(nn.Module):
    """
    Computes multi-task loss with masked BCE for event tasks,
    Huber loss for return and risk, and learned uncertainty weighting.
    """
    def __init__(self, use_uncertainty_weighting: bool = True):
        super().__init__()
        self.use_uncertainty_weighting = use_uncertainty_weighting

        # 4 task log-variances for Kendall et al. uncertainty weighting
        # [relevance, novelty, return, risk]
        self.log_vars = nn.Parameter(torch.zeros(4))

        self.huber_return = nn.HuberLoss(delta=0.02)
        self.huber_risk = nn.HuberLoss(delta=0.02)

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        relevance_labels: torch.Tensor,  # [B, 40, 1]
        novelty_labels: torch.Tensor,    # [B, 40, 1]
        return_targets: torch.Tensor,    # [B, 3]
        risk_targets: torch.Tensor,      # [B, 3]
        event_mask: torch.Tensor         # [B, 40]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Masked BCE for Relevance
        mask_expanded = event_mask.unsqueeze(-1)  # [B, 40, 1]
        rel_logits = predictions["relevance_logits"]
        loss_rel = F.binary_cross_entropy_with_logits(
            rel_logits, relevance_labels, reduction="none"
        )
        loss_rel = (loss_rel * mask_expanded).sum() / (mask_expanded.sum() + 1e-8)

        # Masked BCE for Novelty
        nov_logits = predictions["novelty_logits"]
        loss_nov = F.binary_cross_entropy_with_logits(
            nov_logits, novelty_labels, reduction="none"
        )
        loss_nov = (loss_nov * mask_expanded).sum() / (mask_expanded.sum() + 1e-8)

        # Huber Loss for Multi-Horizon Returns and Risk
        loss_ret = self.huber_return(predictions["pred_returns"], return_targets)
        loss_risk = self.huber_risk(predictions["pred_risk"], risk_targets)

        task_losses = [loss_rel, loss_nov, loss_ret, loss_risk]

        if self.use_uncertainty_weighting:
            total_loss = torch.tensor(0.0, device=loss_rel.device)
            for i, loss in enumerate(task_losses):
                precision = torch.exp(-self.log_vars[i])
                total_loss = total_loss + 0.5 * precision * loss + 0.5 * self.log_vars[i]
        else:
            # Fixed normalized task weights: [0.2, 0.2, 0.3, 0.3]
            weights = [0.2, 0.2, 0.3, 0.3]
            total_loss = sum(w * l for w, l in zip(weights, task_losses))

        loss_dict = {
            "loss_total": float(total_loss.item()),
            "loss_relevance": float(loss_rel.item()),
            "loss_novelty": float(loss_nov.item()),
            "loss_return": float(loss_ret.item()),
            "loss_risk": float(loss_risk.item()),
        }

        return total_loss, loss_dict

"""
Multi-Task Supervised Heads V8.0
1. Event-Stock Relevance Head (128 -> 64 -> 1)
2. Information Novelty Head (256 -> 64 -> 1) operating on [event, prior_memory]
3. Multi-Horizon Excess Return Head (256 -> 128 -> 3) from next-open
4. Multi-Horizon Future Risk Head (256 -> 128 -> 3) from next-open
Supports 4 independent task masks and 3-level novelty targets {0.0, 0.5, 1.0}.
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
        # Compares current event projection (128) against prior active memory summary (128)
        self.novelty_head = nn.Sequential(
            nn.Linear(event_dim * 2, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # Outputs continuous novelty score in [0.0, 1.0] matching {0.0, 0.5, 1.0}
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
        prior_memory: torch.Tensor,           # [B, 128] (active event memory context)
        fused_representation: torch.Tensor,   # [B, 128]
        context_token: torch.Tensor           # [B, 128]
    ) -> Dict[str, torch.Tensor]:
        """
        Returns predictions dict:
            relevance_logits: [B, 40, 1]
            novelty_scores: [B, 40, 1]
            pred_returns: [B, 3]
            pred_risk: [B, 3]
        """
        B, N, D = projected_events.shape

        # 1. Relevance prediction per event slot
        relevance_logits = self.relevance_head(projected_events)  # [B, 40, 1]

        # 2. Novelty prediction: event token compared with prior active memory context
        memory_expanded = prior_memory.unsqueeze(1).expand(-1, N, -1)  # [B, 40, 128]
        novelty_inputs = torch.cat([projected_events, memory_expanded], dim=-1)  # [B, 40, 256]
        novelty_scores = self.novelty_head(novelty_inputs)  # [B, 40, 1]

        # 3 & 4. Stock-Date Return and Risk predictions: fused + context = 256
        stock_date_rep = torch.cat([fused_representation, context_token], dim=-1)  # [B, 256]
        pred_returns = self.return_head(stock_date_rep)  # [B, 3]
        pred_risk = self.risk_head(stock_date_rep)        # [B, 3]

        return {
            "relevance_logits": relevance_logits,
            "novelty_scores": novelty_scores,
            "novelty_logits": novelty_scores,
            "pred_returns": pred_returns,
            "pred_risk": pred_risk,
        }


class MultiTaskLoss(nn.Module):
    """
    Computes multi-task loss with 4 independent validity masks:
    1. Masked BCE for Relevance
    2. Masked Huber Loss for Novelty {0.0, 0.5, 1.0}
    3. Masked Huber Loss for Multi-Horizon Returns
    4. Masked Huber Loss for Multi-Horizon Risk
    """
    def __init__(self, use_uncertainty_weighting: bool = True, novelty_loss_type: str = "huber"):
        super().__init__()
        self.use_uncertainty_weighting = use_uncertainty_weighting
        self.novelty_loss_type = novelty_loss_type

        # 4 task log-variances for Kendall et al. uncertainty weighting
        self.log_vars = nn.Parameter(torch.zeros(4))

        self.huber_novelty = nn.HuberLoss(delta=0.1, reduction="none")
        self.huber_return = nn.HuberLoss(delta=0.02, reduction="none")
        self.huber_risk = nn.HuberLoss(delta=0.02, reduction="none")

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        relevance_labels: torch.Tensor,                    # [B, 40, 1]
        novelty_labels: torch.Tensor,                      # [B, 40, 1]
        return_targets: torch.Tensor,                      # [B, 3]
        risk_targets: torch.Tensor,                        # [B, 3]
        relevance_mask: Optional[torch.Tensor] = None,     # [B, 40, 1]
        novelty_mask: Optional[torch.Tensor] = None,       # [B, 40, 1]
        return_mask: Optional[torch.Tensor] = None,        # [B, 3]
        risk_mask: Optional[torch.Tensor] = None,          # [B, 3]
        event_mask: Optional[torch.Tensor] = None,         # [B, 40]
        fallback_event_mask: Optional[torch.Tensor] = None # [B, 40]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        B = relevance_labels.shape[0]

        if fallback_event_mask is None and event_mask is not None:
            fallback_event_mask = event_mask

        # Resolve Relevance Mask
        if relevance_mask is None:
            relevance_mask = fallback_event_mask.unsqueeze(-1) if fallback_event_mask is not None else torch.ones_like(relevance_labels, dtype=torch.bool)
        rel_logits = predictions["relevance_logits"]
        loss_rel_raw = F.binary_cross_entropy_with_logits(rel_logits, relevance_labels, reduction="none")
        loss_rel = (loss_rel_raw * relevance_mask).sum() / (relevance_mask.sum() + 1e-8)

        # Resolve Novelty Mask
        if novelty_mask is None:
            novelty_mask = fallback_event_mask.unsqueeze(-1) if fallback_event_mask is not None else torch.ones_like(novelty_labels, dtype=torch.bool)
        nov_scores = predictions["novelty_scores"]

        if self.novelty_loss_type == "bce":
            loss_nov_raw = F.binary_cross_entropy(nov_scores, (novelty_labels > 0.25).float(), reduction="none")
        else:
            # Masked Huber Loss for continuous/ordinal Novelty in {0.0, 0.5, 1.0} per V8.0
            loss_nov_raw = self.huber_novelty(nov_scores, novelty_labels)
        loss_nov = (loss_nov_raw * novelty_mask).sum() / (novelty_mask.sum() + 1e-8)

        # Resolve Return Mask
        if return_mask is None:
            return_mask = torch.ones_like(return_targets, dtype=torch.bool)
        loss_ret_raw = self.huber_return(predictions["pred_returns"], return_targets)
        loss_ret = (loss_ret_raw * return_mask).sum() / (return_mask.sum() + 1e-8)

        # Resolve Risk Mask
        if risk_mask is None:
            risk_mask = torch.ones_like(risk_targets, dtype=torch.bool)
        loss_risk_raw = self.huber_risk(predictions["pred_risk"], risk_targets)
        loss_risk = (loss_risk_raw * risk_mask).sum() / (risk_mask.sum() + 1e-8)

        task_losses = [loss_rel, loss_nov, loss_ret, loss_risk]

        if self.use_uncertainty_weighting:
            total_loss = torch.tensor(0.0, device=loss_rel.device)
            for i, loss in enumerate(task_losses):
                precision = torch.exp(-self.log_vars[i])
                total_loss = total_loss + 0.5 * precision * loss + 0.5 * self.log_vars[i]
        else:
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

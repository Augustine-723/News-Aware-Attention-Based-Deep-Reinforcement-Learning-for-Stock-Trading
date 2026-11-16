"""
Stage 1 Training Pipeline: Multi-Task Representation Pretraining V8.0
Trains Market Encoders, Temporal Attention, Fusion, and Supervised Multi-Task Heads.
"""

from typing import Dict, List
import torch
import torch.optim as optim

from stock_rl.config.settings import TrainingScheduleConfig
from stock_rl.data.synthetic_data import MultimodalBatch, generate_synthetic_batch
from stock_rl.models.full_model import StockRLFullModel
from stock_rl.models.multi_task_heads import MultiTaskLoss


def train_stage1_epoch(
    model: StockRLFullModel,
    loss_module: MultiTaskLoss,
    optimizer: optim.Optimizer,
    num_batches: int = 20,
    batch_size: int = 64,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """Runs a single epoch of Stage 1 representation pretraining."""
    model.train()
    loss_module.train()

    epoch_metrics: Dict[str, float] = {
        "loss_total": 0.0,
        "loss_relevance": 0.0,
        "loss_novelty": 0.0,
        "loss_return": 0.0,
        "loss_risk": 0.0,
    }

    for b in range(num_batches):
        batch = generate_synthetic_batch(batch_size=batch_size, seed=100 + b).to(device)

        _, predictions, _ = model(batch)

        loss, loss_dict = loss_module(
            predictions=predictions,
            relevance_labels=batch.relevance_labels,
            novelty_labels=batch.novelty_labels,
            return_targets=batch.return_targets,
            risk_targets=batch.risk_targets,
            event_mask=batch.event_mask,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        for k, v in loss_dict.items():
            epoch_metrics[k] += v / num_batches

    return epoch_metrics


def run_stage1_pretraining(
    model: StockRLFullModel,
    epochs: int = 5,
    num_batches: int = 10,
    lr: float = 3e-4,
    device: torch.device = torch.device("cpu"),
) -> List[Dict[str, float]]:
    """Executes full Stage 1 training schedule."""
    loss_module = MultiTaskLoss(use_uncertainty_weighting=True).to(device)
    all_params = list(model.parameters()) + list(loss_module.parameters())
    optimizer = optim.AdamW(all_params, lr=lr, weight_decay=1e-4)

    history = []
    for ep in range(epochs):
        metrics = train_stage1_epoch(
            model=model,
            loss_module=loss_module,
            optimizer=optimizer,
            num_batches=num_batches,
            device=device,
        )
        history.append(metrics)

    return history

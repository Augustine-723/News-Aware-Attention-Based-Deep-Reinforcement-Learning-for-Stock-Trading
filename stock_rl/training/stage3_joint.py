"""
Stage 3 Training Pipeline: Joint End-to-End Fine-Tuning V8.0
Unfreezes Temporal Attention, Cross-Modal Fusion, and Event Projection with auxiliary multi-task loss.
Strictly inherits Stage 2's trained Agent, Target Network, Replay Buffer, and epsilon schedule.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from stock_rl.config.settings import TrainingScheduleConfig
from stock_rl.env.stock_trading_env import PointInTimeStockEnv
from stock_rl.models.full_model import StockRLFullModel
from stock_rl.models.multi_task_heads import MultiTaskLoss
from stock_rl.rl.agent import DuelingDoubleDQNAgent
from stock_rl.rl.replay_buffer import MultimodalReplayBuffer


def run_stage3_joint_finetuning(
    model: StockRLFullModel,
    env: PointInTimeStockEnv,
    episodes: int = 5,
    aux_weight: float = 0.2,
    batch_size: int = 32,
    lr: float = 5e-5,
    agent: Optional[DuelingDoubleDQNAgent] = None,
    buffer: Optional[MultimodalReplayBuffer] = None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[DuelingDoubleDQNAgent, MultimodalReplayBuffer, List[Dict[str, float]]]:
    """
    Executes Stage 3: selective unfreezing of Temporal Attention, Fusion,
    and Event Projection with joint RL + auxiliary multi-task loss.
    Inherits agent and buffer from Stage 2 to prevent training state reset.
    """
    # Selectively unfreeze Temporal Attention, Fusion, Event Projection, Heads, and DQN
    model.unfreeze_for_stage3()

    if agent is None:
        agent = DuelingDoubleDQNAgent(model=model, device=device)
    if buffer is None:
        buffer = MultimodalReplayBuffer(capacity=50_000)

    # Reconfigure optimizer for Stage 3 while migrating existing momentum states
    agent.reconfigure_optimizer_for_stage3(lr=lr)

    loss_module = MultiTaskLoss(use_uncertainty_weighting=False, novelty_loss_type="huber").to(device)
    episode_history: List[Dict[str, float]] = []

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        joint_losses = []

        while not done:
            action = agent.select_action(obs, evaluate=False)
            agent.step_epsilon()

            next_obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated
            total_reward += reward

            # Push transition with separate terminated and truncated flags
            buffer.push(obs, action, reward, next_obs, terminated=terminated, truncated=truncated)
            obs = next_obs

            if len(buffer) >= 64:
                b_obs, b_act, b_rew, b_next_obs, b_term, b_trunc = buffer.sample(batch_size)
                b_obs = b_obs.to(device)
                b_next_obs = b_next_obs.to(device)
                b_act = b_act.to(device)
                b_rew = b_rew.to(device)
                b_term = b_term.to(device)

                model.train()
                # 1. Forward online model
                q_online_current, preds, _ = model(b_obs)
                q_action_values = q_online_current.gather(1, b_act.unsqueeze(1)).squeeze(1)

                # 2. Target Q calculation with strict terminated-only bootstrap masking
                with torch.no_grad():
                    q_online_next, _, _ = model(b_next_obs)
                    best_next_actions = torch.argmax(q_online_next, dim=1, keepdim=True)

                    q_target_next, _, _ = agent.target_model(b_next_obs)
                    target_next_q = q_target_next.gather(1, best_next_actions).squeeze(1)
                    # (1.0 - b_term.float()): Time-limit truncated states continue bootstrapping
                    expected_q = b_rew + (1.0 - b_term.float()) * (agent.config.gamma * target_next_q)

                rl_loss = agent.loss_fn(q_action_values, expected_q)

                # 3. Auxiliary multi-task loss with 4 independent validity masks
                if b_obs.return_targets is not None:
                    aux_loss, _ = loss_module(
                        predictions=preds,
                        relevance_labels=b_obs.relevance_labels,
                        novelty_labels=b_obs.novelty_labels,
                        return_targets=b_obs.return_targets,
                        risk_targets=b_obs.risk_targets,
                        relevance_mask=b_obs.relevance_mask,
                        novelty_mask=b_obs.novelty_mask,
                        return_mask=b_obs.return_mask,
                        risk_mask=b_obs.risk_mask,
                        fallback_event_mask=b_obs.event_mask,
                    )
                else:
                    aux_loss = torch.tensor(0.0, device=device)

                total_loss = rl_loss + aux_weight * aux_loss

                agent.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                agent.optimizer.step()

                agent._soft_update_target(agent.config.target_tau)
                joint_losses.append(float(total_loss.item()))

        episode_history.append({
            "episode": ep + 1,
            "total_reward": total_reward,
            "final_equity": float(step_info["portfolio_equity"]),
            "avg_joint_loss": float(np.mean(joint_losses)) if joint_losses else 0.0,
            "epsilon": agent.epsilon,
        })

    return agent, buffer, episode_history

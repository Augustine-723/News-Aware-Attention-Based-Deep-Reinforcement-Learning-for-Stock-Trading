"""
Stage 2 Training Pipeline: Frozen-Encoder Dueling Double DQN V8.0
Freezes representation encoders and trains Q-value backbone on environment transitions.
"""

from typing import List, Dict
import numpy as np
import torch

from stock_rl.config.settings import RLConfig
from stock_rl.data.synthetic_data import MultimodalBatch
from stock_rl.env.stock_trading_env import PointInTimeStockEnv
from stock_rl.models.full_model import StockRLFullModel
from stock_rl.rl.agent import DuelingDoubleDQNAgent
from stock_rl.rl.replay_buffer import MultimodalReplayBuffer


def run_stage2_rl_training(
    model: StockRLFullModel,
    env: PointInTimeStockEnv,
    episodes: int = 10,
    warmup_steps: int = 128,
    train_steps_per_env_step: int = 1,
    batch_size: int = 32,
    device: torch.device = torch.device("cpu"),
) -> List[Dict[str, float]]:
    """
    Executes Stage 2: representation encoders frozen, Dueling DDQN trained on environment.
    """
    # Freeze representation encoders
    model.freeze_encoders()

    agent = DuelingDoubleDQNAgent(model=model, device=device)
    buffer = MultimodalReplayBuffer(capacity=50_000)

    episode_history: List[Dict[str, float]] = []

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        losses = []

        while not done:
            action = agent.select_action(obs, evaluate=False)
            agent.step_epsilon()

            next_obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated
            total_reward += reward

            # Push transition to replay buffer
            buffer.push(obs, action, reward, next_obs, done)
            obs = next_obs

            # Train if buffer exceeds warmup
            if len(buffer) >= warmup_steps:
                for _ in range(train_steps_per_env_step):
                    b_obs, b_act, b_rew, b_next_obs, b_done = buffer.sample(batch_size)
                    loss = agent.train_step(b_obs, b_act, b_rew, b_next_obs, b_done)
                    losses.append(loss)

        avg_loss = float(np.mean(losses)) if losses else 0.0
        final_equity = float(step_info["portfolio_equity"])
        episode_history.append({
            "episode": ep + 1,
            "total_reward": total_reward,
            "final_equity": final_equity,
            "avg_loss": avg_loss,
            "epsilon": agent.epsilon,
        })

    return episode_history

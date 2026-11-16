"""
Unit Tests for Replay Buffer, Double DQN Target Computation, and 3-Stage Training Schedule
"""

import pytest
import torch
from stock_rl.data.synthetic_data import generate_synthetic_batch
from stock_rl.models.full_model import StockRLFullModel
from stock_rl.rl.replay_buffer import MultimodalReplayBuffer
from stock_rl.rl.agent import DuelingDoubleDQNAgent
from stock_rl.env.stock_trading_env import PointInTimeStockEnv
from stock_rl.training.stage1_pretrain import run_stage1_pretraining
from stock_rl.training.stage2_rl import run_stage2_rl_training
from stock_rl.training.stage3_joint import run_stage3_joint_finetuning


def test_replay_buffer_and_double_dqn_update():
    buffer = MultimodalReplayBuffer(capacity=100)
    model = StockRLFullModel()
    agent = DuelingDoubleDQNAgent(model=model)

    obs = generate_synthetic_batch(batch_size=1, seed=1)
    next_obs = generate_synthetic_batch(batch_size=1, seed=2)

    for _ in range(10):
        buffer.push(obs, action=1, reward=0.01, next_obs=next_obs, done=False)

    assert len(buffer) == 10

    b_obs, b_act, b_rew, b_next_obs, b_done = buffer.sample(batch_size=4)
    loss = agent.train_step(b_obs, b_act, b_rew, b_next_obs, b_done)

    assert not torch.isnan(torch.tensor(loss))
    assert loss >= 0.0


def test_three_stage_pipeline_execution():
    model = StockRLFullModel()
    env = PointInTimeStockEnv()

    # Stage 1: Pretraining (1 epoch, 2 batches)
    s1_history = run_stage1_pretraining(model=model, epochs=1, num_batches=2)
    assert len(s1_history) == 1
    assert "loss_total" in s1_history[0]

    # Stage 2: Frozen RL (1 episode)
    s2_history = run_stage2_rl_training(model=model, env=env, episodes=1, warmup_steps=4, batch_size=4)
    assert len(s2_history) == 1
    assert "total_reward" in s2_history[0]

    # Stage 3: Joint Fine-Tuning (1 episode)
    s3_history = run_stage3_joint_finetuning(model=model, env=env, episodes=1, batch_size=4)
    assert len(s3_history) == 1
    assert "avg_joint_loss" in s3_history[0]

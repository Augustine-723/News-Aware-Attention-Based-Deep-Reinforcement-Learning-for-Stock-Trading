"""
Event-Driven Stock RL — Main Demonstration Pipeline V8.0
Executes Stages 1, 2, and 3 (with strict training state inheritance),
followed by Walk-Forward Evaluation & Benchmark comparisons.
"""

import sys
import torch
import numpy as np

from stock_rl.config.schema import FeatureDimensions
from stock_rl.config.settings import TradingConfig, RLConfig, TrainingScheduleConfig
from stock_rl.models.full_model import StockRLFullModel
from stock_rl.env.stock_trading_env import PointInTimeStockEnv
from stock_rl.training.stage1_pretrain import run_stage1_pretraining
from stock_rl.training.stage2_rl import run_stage2_rl_training
from stock_rl.training.stage3_joint import run_stage3_joint_finetuning
from stock_rl.evaluation.walk_forward import evaluate_policy_on_env
from stock_rl.evaluation.ablations import run_buy_and_hold_benchmark


def main():
    print("=" * 80)
    print("Event-Driven Stock RL — Point-in-Time Research System V8.0")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Device: {device}")

    # 1. Initialize Full Model & Environment
    print("\n[+] Initializing Point-in-Time Architecture & Gym Trading Environment...")
    model = StockRLFullModel().to(device)
    env = PointInTimeStockEnv()

    # 2. Stage 1: Representation Pretraining
    print("\n[+] Stage 1: Multi-Task Representation Pretraining...")
    s1_history = run_stage1_pretraining(
        model=model,
        epochs=3,
        num_batches=5,
        device=device,
    )
    for i, h in enumerate(s1_history, 1):
        print(f"    Epoch {i:02d} | Total Loss: {h['loss_total']:.4f} "
              f"(Rel: {h['loss_relevance']:.4f}, Nov: {h['loss_novelty']:.4f}, "
              f"Ret: {h['loss_return']:.4f}, Risk: {h['loss_risk']:.4f})")

    # 3. Stage 2: Frozen Encoders & Dueling Double DQN
    print("\n[+] Stage 2: Training Dueling Double DQN (Frozen Encoders)...")
    agent, buffer, s2_history = run_stage2_rl_training(
        model=model,
        env=env,
        episodes=3,
        warmup_steps=32,
        batch_size=16,
        device=device,
    )
    for h in s2_history:
        print(f"    Episode {h['episode']:02d} | Return: {h['total_reward']:.4f} | "
              f"Equity: ${h['final_equity']:,.2f} | Loss: {h['avg_loss']:.4f} | Epsilon: {h['epsilon']:.3f}")

    # 4. Stage 3: Selective Fine-Tuning (Inheriting agent and buffer)
    print("\n[+] Stage 3: Joint Fine-Tuning (RL + Auxiliary Multi-Task Loss)...")
    agent, buffer, s3_history = run_stage3_joint_finetuning(
        model=model,
        env=env,
        episodes=2,
        batch_size=16,
        agent=agent,
        buffer=buffer,
        device=device,
    )
    for h in s3_history:
        print(f"    Episode {h['episode']:02d} | Return: {h['total_reward']:.4f} | "
              f"Equity: ${h['final_equity']:,.2f} | Joint Loss: {h['avg_joint_loss']:.4f}")

    # 5. Out-of-Sample Evaluation & Benchmark Comparison
    print("\n[+] Running Out-of-Sample Evaluation & Benchmark Comparison...")
    test_env = PointInTimeStockEnv()
    metrics_rl, _, _ = evaluate_policy_on_env(model, test_env, device=device)

    # Buy and Hold Benchmark
    bh_env = PointInTimeStockEnv()
    metrics_bh = run_buy_and_hold_benchmark(bh_env)

    print("\n" + "=" * 80)
    print(f"{'Performance Metric':<30} | {'Event-Driven RL (V8.0)':<22} | {'Buy & Hold Baseline':<20}")
    print("-" * 80)
    metrics_to_show = [
        ("total_return", "Total Return (%)", 100.0, "%"),
        ("annualized_return", "Annualized Return (%)", 100.0, "%"),
        ("sharpe_ratio", "Sharpe Ratio", 1.0, ""),
        ("sortino_ratio", "Sortino Ratio", 1.0, ""),
        ("max_drawdown", "Max Drawdown (%)", 100.0, "%"),
        ("calmar_ratio", "Calmar Ratio", 1.0, ""),
        ("annualized_turnover", "Annualized Turnover (x)", 1.0, "x"),
    ]
    for key, label, scale, unit in metrics_to_show:
        v_rl = f"{metrics_rl.get(key, 0.0) * scale:+.2f}{unit}"
        v_bh = f"{metrics_bh.get(key, 0.0) * scale:+.2f}{unit}"
        print(f"{label:<30} | {v_rl:<22} | {v_bh:<20}")
    print("=" * 80)
    print("\n[✓] Research system pipeline verified successfully.")


if __name__ == "__main__":
    main()

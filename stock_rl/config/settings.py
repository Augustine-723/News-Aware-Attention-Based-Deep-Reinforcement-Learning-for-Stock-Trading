"""
Event-Driven Stock RL — System Settings & Hyperparameters V8.0
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class TradingConfig:
    # Market & Exchange
    market: str = "TWSE"
    timezone: str = "Asia/Taipei"
    decision_time_str: str = "20:00"  # Daily decision at 20:00 Asia/Taipei
    execution_time_str: str = "09:00" # Next trading day open

    # Frictions and Execution Limits
    twse_base_commission_rate: float = 0.001425  # 0.1425%
    broker_discount: float = 0.60                # 40% discount -> 0.0855%
    twse_securities_tax_rate: float = 0.0030     # 0.3% on sell only
    base_slippage_rate: float = 0.0005           # 5 bps slippage proxy
    daily_price_limit_pct: float = 0.10          # TWSE +-10% daily price limit

    # Universe
    universe_size: int = 100                     # Historical dynamic 100-stock selection


@dataclass(frozen=True)
class RLConfig:
    gamma: float = 0.99
    batch_size: int = 64
    replay_capacity: int = 100_000
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    target_update_freq: int = 1000               # Steps between target network sync
    target_tau: float = 0.005                    # Soft update parameter
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay_steps: int = 50_000
    grad_norm_clip: float = 10.0


@dataclass(frozen=True)
class ModelConfig:
    market_hidden_dim: int = 128
    market_num_layers: int = 2
    event_projection_hidden: int = 256
    event_projection_out: int = 128
    num_attention_heads: int = 4
    attention_dropout: float = 0.1
    fusion_dim: int = 128
    rl_state_dim: int = 132
    dqn_shared_hidden: Tuple[int, int] = (128, 64)
    dqn_stream_hidden: int = 32


@dataclass(frozen=True)
class TrainingScheduleConfig:
    # Split dates
    train_start: str = "2021-11-01"
    train_end: str = "2024-10-31"
    val_start: str = "2024-11-01"
    val_end: str = "2025-10-31"
    test_start: str = "2025-11-01"
    test_end: str = "2026-11-01"
    boundary_purge_days: int = 5  # Purge 5 trading days at split boundaries for multi-horizon safety

    # Stage epochs
    stage1_epochs: int = 20
    stage2_episodes: int = 100
    stage3_episodes: int = 50
    auxiliary_loss_weight: float = 0.2

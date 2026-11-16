# Event-Driven Stock RL — Point-in-Time Research System V8.0

A point-in-time, event-driven, multimodal deep reinforcement learning research framework for Taiwan Stock Exchange (TWSE) listed common stocks, built in compliance with Specification V8.0 and V7.0.

---

## 1. System Architecture

```
stock_rl/
├── config/
│   ├── schema.py           # Feature definitions, authoritative column order, tensor shapes
│   └── settings.py         # Trading costs (TWSE fees/taxes), hyperparameters, dates
├── data/
│   ├── pit_engine.py       # Point-in-time guard, multi-timestamp tracking, boundary purging
│   ├── universe.py         # Historical dynamic 100-stock selection (survivorship-free)
│   ├── synthetic_data.py   # Ground-truth multimodal generator with causal event structure
│   └── tensor_validator.py # Strict tensor dimensions, NaN/Inf and range contract validator
├── models/
│   ├── market_encoder.py   # GRU encoders for Price, Technical, Institutional, and Macro
│   ├── event_module.py     # Event Projection (160->256->128), Age encoding, NRE token, Temporal Attention
│   ├── cross_modal.py      # Cross-Modal Attention fusing market tokens & event token into 128-d
│   ├── multi_task_heads.py # Relevance, Novelty, 3-Horizon Return & Risk heads + Uncertainty Loss
│   ├── dueling_dqn.py      # 132-d RL state, Dueling Double DQN (shared backbone, value & advantage streams)
│   └── full_model.py       # End-to-end unified model coordinating encoders, heads, and DQN
├── env/
│   ├── execution.py        # TWSE simulation (open execution, commission, 0.3% tax, slippage, +-10% limits)
│   └── stock_trading_env.py# Point-in-Time Gymnasium trading environment (20:00 decision cycle, equity log return)
├── rl/
│   ├── replay_buffer.py    # Experience replay buffer with raw multimodal tensor storage
│   └── agent.py            # Dueling Double DQN agent with consistent target networks
├── training/
│   ├── stage1_pretrain.py  # Stage 1: Multi-task representation pretraining
│   ├── stage2_rl.py        # Stage 2: Frozen-encoder Dueling Double DQN training
│   └── stage3_joint.py     # Stage 3: End-to-end joint fine-tuning (RL + auxiliary loss)
└── evaluation/
    ├── metrics.py          # Sharpe, Sortino, Calmar, Max Drawdown, Turnover, cost-adjusted returns
    ├── walk_forward.py     # Purged boundary walk-forward validator & multi-seed aggregator
    └── ablations.py        # Market-only, Sentiment-only, Event-removal, Perturbation ablations
```

---

## 2. Specification Tensor Contracts

| Component | Dimensions | Tensor Shape |
|---|---:|---|
| Price Sequence | 12 | `[B, 60, 12]` |
| Technical Sequence | 16 | `[B, 60, 16]` |
| Institutional Flow | 8 | `[B, 60, 8]` |
| Macro Indicators | 12 | `[B, 60, 12]` |
| Stock Context | 16 | `[B, 16]` |
| Recent Event Embeddings | 128 | `[B, 32, 128]` |
| Active Event Embeddings | 128 | `[B, 8, 128]` |
| Event Metadata | 24 | `[B, 40, 24]` |
| Event–Stock Relation | 8 | `[B, 40, 8]` |
| Event Mask | 1 | `[B, 40]` (bool) |
| Position State | 4 | `[B, 4]` |
| **Multimodal Fusion** | **128** | `[B, 128]` |
| **RL State Vector** | **132** | `[B, 132]` |
| **Q-Value Output** | **2** | `[B, 2]` ($\{Q(\text{Cash}), Q(\text{Long})\}) |

---

## 3. Running Tests & Demonstration

```bash
# Run the complete test suite (18 tests passing)
pytest tests/ -v

# Run the 3-Stage Training & Benchmark Demonstration
python3 main.py
```

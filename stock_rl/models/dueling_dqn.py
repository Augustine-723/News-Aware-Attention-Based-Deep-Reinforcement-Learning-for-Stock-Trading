"""
Dueling Double DQN Network V8.0
RL state: 132-d (128-d Multimodal Fusion + 4-d Position State)
Backbone: 132 -> 128 -> 64
Value stream: 64 -> 32 -> 1
Advantage stream: 64 -> 32 -> 2
Output: [B, 2] for Q(Cash) and Q(Long)
"""

import torch
import torch.nn as nn
from stock_rl.config.schema import (
    RL_STATE_DIM,
    NUM_ACTIONS,
    POSITION_STATE_DIM,
    MULTIMODAL_FUSION_DIM,
)


class DuelingDQN(nn.Module):
    """
    Dueling DQN architecture with separate Value and Advantage streams.
    Enforces identifiability via mean-subtracted advantage:
        Q(s, a) = V(s) + (A(s, a) - (1 / |A|) * sum_a' A(s, a'))
    """
    def __init__(
        self,
        state_dim: int = RL_STATE_DIM,  # 132
        num_actions: int = NUM_ACTIONS, # 2
        shared_hidden1: int = 128,
        shared_hidden2: int = 64,
        stream_hidden: int = 32,
    ):
        super().__init__()
        # Shared Q Backbone: 132 -> 128 -> 64
        self.shared_backbone = nn.Sequential(
            nn.Linear(state_dim, shared_hidden1),
            nn.LayerNorm(shared_hidden1),
            nn.GELU(),
            nn.Linear(shared_hidden1, shared_hidden2),
            nn.LayerNorm(shared_hidden2),
            nn.GELU(),
        )

        # Value Stream: 64 -> 32 -> 1
        self.value_stream = nn.Sequential(
            nn.Linear(shared_hidden2, stream_hidden),
            nn.GELU(),
            nn.Linear(stream_hidden, 1),
        )

        # Advantage Stream: 64 -> 32 -> 2
        self.advantage_stream = nn.Sequential(
            nn.Linear(shared_hidden2, stream_hidden),
            nn.GELU(),
            nn.Linear(stream_hidden, num_actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Input: state [B, 132]
        Output: Q values [B, 2]
        """
        features = self.shared_backbone(state)  # [B, 64]

        values = self.value_stream(features)         # [B, 1]
        advantages = self.advantage_stream(features) # [B, 2]

        # Dueling formula: Q = V + (A - mean(A))
        mean_advantage = advantages.mean(dim=-1, keepdim=True)
        q_values = values + (advantages - mean_advantage)  # [B, 2]
        return q_values

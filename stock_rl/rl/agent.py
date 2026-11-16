"""
Dueling Double DQN Agent V8.0
Implements target network consistency, Double DQN update rules, and epsilon decay.
"""

import copy
import random
from typing import Tuple, Dict
import torch
import torch.nn as nn
import torch.optim as optim

from stock_rl.config.settings import RLConfig
from stock_rl.data.synthetic_data import MultimodalBatch
from stock_rl.models.full_model import StockRLFullModel


class DuelingDoubleDQNAgent:
    """
    Agent coordinating action selection and Double DQN updates with consistent target networks.
    """
    def __init__(
        self,
        model: StockRLFullModel,
        config: RLConfig = RLConfig(),
        device: torch.device = torch.device("cpu"),
    ):
        self.config = config
        self.device = device
        self.online_model = model.to(device)
        self.target_model = copy.deepcopy(model).to(device)
        self.target_model.eval()

        # Optimizer: AdamW with weight decay
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.online_model.parameters()),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.loss_fn = nn.HuberLoss(delta=1.0)

        self.epsilon = config.epsilon_start
        self.epsilon_min = config.epsilon_min
        self.epsilon_decay = (config.epsilon_start - config.epsilon_min) / config.epsilon_decay_steps
        self.total_steps = 0

    def select_action(self, obs: MultimodalBatch, evaluate: bool = False) -> int:
        """
        Epsilon-greedy action selection.
        Returns: 0 (Cash) or 1 (Long).
        """
        if not evaluate and random.random() < self.epsilon:
            return random.choice([0, 1])

        self.online_model.eval()
        with torch.no_grad():
            batch_dev = obs.to(self.device)
            q_values, _, _ = self.online_model(batch_dev)
            action = int(torch.argmax(q_values[0]).item())

        return action

    def step_epsilon(self):
        """Linearly decays exploration rate epsilon."""
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)
        self.total_steps += 1

    def train_step(
        self,
        batch_obs: MultimodalBatch,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        batch_next_obs: MultimodalBatch,
        dones: torch.Tensor,
    ) -> float:
        """
        Executes a Double DQN learning step.
        """
        self.online_model.train()
        batch_obs = batch_obs.to(self.device)
        batch_next_obs = batch_next_obs.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)

        # 1. Compute current Q values Q_online(s, a)
        q_online_current, _, _ = self.online_model(batch_obs)
        q_action_values = q_online_current.gather(1, actions.unsqueeze(1)).squeeze(1)

        # 2. Double DQN Target:
        # a* = argmax_a Q_online(s', a)
        # target_val = Q_target(s', a*)
        with torch.no_grad():
            q_online_next, _, _ = self.online_model(batch_next_obs)
            best_next_actions = torch.argmax(q_online_next, dim=1, keepdim=True)

            q_target_next, _, _ = self.target_model(batch_next_obs)
            target_next_q = q_target_next.gather(1, best_next_actions).squeeze(1)

            expected_q = rewards + (1.0 - dones) * (self.config.gamma * target_next_q)

        # 3. Compute loss and optimize
        loss = self.loss_fn(q_action_values, expected_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_model.parameters(), self.config.grad_norm_clip)
        self.optimizer.step()

        # 4. Soft update target network: theta_target = tau * theta_online + (1 - tau) * theta_target
        self._soft_update_target(self.config.target_tau)

        return float(loss.item())

    def _soft_update_target(self, tau: float):
        for target_param, online_param in zip(self.target_model.parameters(), self.online_model.parameters()):
            target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)

    def hard_update_target(self):
        """Copies weights from online to target network directly."""
        self.target_model.load_state_dict(self.online_model.state_dict())

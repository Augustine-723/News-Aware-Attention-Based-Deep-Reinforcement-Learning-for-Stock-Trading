"""
Dueling Double DQN Agent V8.0
Implements target network consistency, strict Terminated vs Truncated bootstrapping,
and stage state preservation/migration across training phases.
"""

import copy
import random
from typing import Tuple, Dict, Optional, Any
import torch
import torch.nn as nn
import torch.optim as optim

from stock_rl.config.settings import RLConfig
from stock_rl.config.schema import CURRENT_ACTIVE_SCHEMA_VERSION, validate_schema_version
from stock_rl.data.synthetic_data import MultimodalBatch
from stock_rl.models.full_model import StockRLFullModel


class DuelingDoubleDQNAgent:
    """
    Agent coordinating action selection and Double DQN updates with consistent target networks.
    Enforces schema version consistency across checkpoints and training updates.
    """
    def __init__(
        self,
        model: StockRLFullModel,
        config: RLConfig = RLConfig(),
        device: torch.device = torch.device("cpu"),
        schema_version: Optional[str] = None,
    ):
        self.config = config
        self.device = device
        self.schema_version = schema_version or CURRENT_ACTIVE_SCHEMA_VERSION
        validate_schema_version(self.schema_version)

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
        terminated: torch.Tensor,
    ) -> float:
        """
        Executes a Double DQN learning step.
        Bootstraps ONLY when terminated is False (1.0 - terminated).
        If an episode was truncated due to time limits, terminated is 0, preserving Bellman bootstrap.
        """
        self.online_model.train()
        batch_obs = batch_obs.to(self.device)
        batch_next_obs = batch_next_obs.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        terminated = terminated.to(self.device)

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

            # Strict V8.0 Bootstrap Rule: Only zero out bootstrap on actual termination (e.g. delisting)
            term_mask = terminated.float() if isinstance(terminated, torch.Tensor) else float(terminated)
            expected_q = rewards + (1.0 - term_mask) * (self.config.gamma * target_next_q)

        # 3. Compute loss and optimize
        loss = self.loss_fn(q_action_values, expected_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_model.parameters(), self.config.grad_norm_clip)
        self.optimizer.step()

        # 4. Soft update target network: theta_target = tau * theta_online + (1 - tau) * theta_target
        self._soft_update_target(self.config.target_tau)

        return float(loss.item())

    def reconfigure_optimizer_for_stage3(
        self,
        lr: float = 5e-5,
        weight_decay: float = 1e-4,
        inherit_momentum: bool = True
    ) -> Dict[str, Any]:
        """
        Reconfigures optimizer for Stage 3 joint fine-tuning.

        If inherit_momentum is True:
            Migrates first (exp_avg) and second (exp_avg_sq) moment buffers and step counts
            for parameters already trained in Stage 2 (DQN backbone). Newly unfrozen parameters
            (Temporal Attention, Fusion, Event Projection) initialize with clean zero momentum.
        If inherit_momentum is False:
            Explicitly re-initializes all optimizer states from scratch.

        Returns:
            migration_summary: Dict with count of migrated vs newly initialized parameters.
        """
        migrated_count = 0
        newly_initialized_count = 0

        # Extract current optimizer momentum states by parameter object reference
        param_to_state = {}
        if self.optimizer is not None:
            for p in self.online_model.parameters():
                if p in self.optimizer.state:
                    param_to_state[p] = self.optimizer.state[p]

        # Gather currently active trainable parameters (now including unfrozen encoders)
        trainable_params = list(filter(lambda p: p.requires_grad, self.online_model.parameters()))

        # Instantiate Stage 3 AdamW optimizer
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=lr,
            weight_decay=weight_decay,
        )

        if inherit_momentum and param_to_state:
            for p in trainable_params:
                if p in param_to_state and len(param_to_state[p]) > 0:
                    old_p_state = param_to_state[p]
                    # Deep copy momentum tensors into the new optimizer state
                    new_p_state = {}
                    for k, v in old_p_state.items():
                        if isinstance(v, torch.Tensor):
                            new_p_state[k] = v.clone().detach()
                        else:
                            new_p_state[k] = copy.deepcopy(v)
                    self.optimizer.state[p] = new_p_state
                    migrated_count += 1
                else:
                    newly_initialized_count += 1
        else:
            newly_initialized_count = len(trainable_params)

        return {
            "inherit_momentum": inherit_momentum,
            "migrated_parameters": migrated_count,
            "newly_initialized_parameters": newly_initialized_count,
            "total_trainable_parameters": len(trainable_params),
        }

    def _soft_update_target(self, tau: float):
        for target_param, online_param in zip(self.target_model.parameters(), self.online_model.parameters()):
            target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)

    def hard_update_target(self):
        """Copies weights from online to target network directly."""
        self.target_model.load_state_dict(self.online_model.state_dict())

    def save_checkpoint(
        self,
        filepath: str,
        additional_metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Saves agent state dictionary with explicit schema version metadata.
        """
        checkpoint = {
            "online_model_state_dict": self.online_model.state_dict(),
            "target_model_state_dict": self.target_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer is not None else None,
            "epsilon": self.epsilon,
            "total_steps": self.total_steps,
            "schema_version": self.schema_version,
            "metadata": additional_metadata or {},
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(
        self,
        filepath: str,
        expected_schema_version: Optional[str] = None,
        strict_schema_check: bool = True,
    ) -> Dict[str, Any]:
        """
        Loads agent state dictionary with strict schema version verification.
        Raises ValueError if checkpoint schema does not match expected schema.
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        saved_schema = checkpoint.get("schema_version", None)
        expected_schema = expected_schema_version or self.schema_version

        if strict_schema_check and saved_schema != expected_schema:
            raise ValueError(
                f"Checkpoint schema violation: checkpoint was saved with schema_version '{saved_schema}', "
                f"but agent/runtime requires '{expected_schema}'. Silent mixing of different schemas is prohibited."
            )

        self.online_model.load_state_dict(checkpoint["online_model_state_dict"])
        self.target_model.load_state_dict(checkpoint["target_model_state_dict"])
        if self.optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon)
        self.total_steps = checkpoint.get("total_steps", self.total_steps)
        return checkpoint

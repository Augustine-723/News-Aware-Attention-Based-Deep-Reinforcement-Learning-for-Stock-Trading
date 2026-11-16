"""
Multimodal Experience Replay Buffer V8.0
Stores complete transition tuples with raw multimodal observation tensors and explicitly
distinguishes terminated vs truncated flags for correct Double DQN bootstrapping.
"""

from collections import deque
import random
from typing import List, Tuple, NamedTuple, Optional
import torch
from stock_rl.config.schema import CURRENT_ACTIVE_SCHEMA_VERSION, validate_schema_version
from stock_rl.data.synthetic_data import MultimodalBatch


class Transition(NamedTuple):
    obs: MultimodalBatch
    action: int
    reward: float
    next_obs: MultimodalBatch
    terminated: bool
    truncated: bool


class MultimodalReplayBuffer:
    """
    Fixed-capacity ring buffer storing transition tuples.
    Collates raw tensors into single batched MultimodalBatch objects upon sampling.
    Guarantees strict schema version consistency and rejects mixed schemas.
    """
    def __init__(self, capacity: int = 100_000, schema_version: Optional[str] = None):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.schema_version = schema_version or CURRENT_ACTIVE_SCHEMA_VERSION
        validate_schema_version(self.schema_version)

    def push(
        self,
        obs: MultimodalBatch,
        action: int,
        reward: float,
        next_obs: MultimodalBatch,
        terminated: bool,
        truncated: bool = False
    ):
        obs_schema = getattr(obs, "schema_version", None)
        next_obs_schema = getattr(next_obs, "schema_version", None)

        if obs_schema is not None and obs_schema != self.schema_version:
            raise ValueError(
                f"Replay Buffer schema violation: buffer configured for '{self.schema_version}', "
                f"but received obs with '{obs_schema}'. Silent mixing of different schemas is prohibited."
            )
        if next_obs_schema is not None and next_obs_schema != self.schema_version:
            raise ValueError(
                f"Replay Buffer schema violation: buffer configured for '{self.schema_version}', "
                f"but received next_obs with '{next_obs_schema}'. Silent mixing of different schemas is prohibited."
            )

        self.buffer.append(Transition(obs.clone(), action, reward, next_obs.clone(), bool(terminated), bool(truncated)))

    def __len__(self) -> int:
        return len(self.buffer)

    def sample(self, batch_size: int) -> Tuple[MultimodalBatch, torch.Tensor, torch.Tensor, MultimodalBatch, torch.Tensor, torch.Tensor]:
        """
        Samples a mini-batch of transitions.
        Returns:
            (batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_terminated, batch_truncated)
        """
        samples: List[Transition] = random.sample(self.buffer, batch_size)

        actions = torch.tensor([s.action for s in samples], dtype=torch.int64)
        rewards = torch.tensor([s.reward for s in samples], dtype=torch.float32)
        terminated = torch.tensor([float(s.terminated) for s in samples], dtype=torch.float32)
        truncated = torch.tensor([float(s.truncated) for s in samples], dtype=torch.float32)

        # Collate observations into batched MultimodalBatch with explicit schema_version
        batch_obs = self._collate_batches([s.obs for s in samples], schema_version=self.schema_version)
        batch_next_obs = self._collate_batches([s.next_obs for s in samples], schema_version=self.schema_version)

        return batch_obs, actions, rewards, batch_next_obs, terminated, truncated

    @staticmethod
    def _collate_batches(batch_list: List[MultimodalBatch], schema_version: Optional[str] = None) -> MultimodalBatch:
        """Concatenates a list of single-sample batches along dim 0."""
        target_schema = schema_version or getattr(batch_list[0], "schema_version", CURRENT_ACTIVE_SCHEMA_VERSION)
        return MultimodalBatch(
            price=torch.cat([b.price for b in batch_list], dim=0),
            technical=torch.cat([b.technical for b in batch_list], dim=0),
            institutional=torch.cat([b.institutional for b in batch_list], dim=0),
            macro=torch.cat([b.macro for b in batch_list], dim=0),
            stock_context=torch.cat([b.stock_context for b in batch_list], dim=0),
            recent_event_emb=torch.cat([b.recent_event_emb for b in batch_list], dim=0),
            active_event_emb=torch.cat([b.active_event_emb for b in batch_list], dim=0),
            event_metadata=torch.cat([b.event_metadata for b in batch_list], dim=0),
            event_stock_rel=torch.cat([b.event_stock_rel for b in batch_list], dim=0),
            event_mask=torch.cat([b.event_mask for b in batch_list], dim=0),
            position=torch.cat([b.position for b in batch_list], dim=0),
            relevance_labels=torch.cat([b.relevance_labels for b in batch_list], dim=0) if batch_list[0].relevance_labels is not None else None,
            novelty_labels=torch.cat([b.novelty_labels for b in batch_list], dim=0) if batch_list[0].novelty_labels is not None else None,
            return_targets=torch.cat([b.return_targets for b in batch_list], dim=0) if batch_list[0].return_targets is not None else None,
            risk_targets=torch.cat([b.risk_targets for b in batch_list], dim=0) if batch_list[0].risk_targets is not None else None,
            relevance_mask=torch.cat([b.relevance_mask for b in batch_list], dim=0) if batch_list[0].relevance_mask is not None else None,
            novelty_mask=torch.cat([b.novelty_mask for b in batch_list], dim=0) if batch_list[0].novelty_mask is not None else None,
            return_mask=torch.cat([b.return_mask for b in batch_list], dim=0) if batch_list[0].return_mask is not None else None,
            risk_mask=torch.cat([b.risk_mask for b in batch_list], dim=0) if batch_list[0].risk_mask is not None else None,
            schema_version=target_schema,
        )

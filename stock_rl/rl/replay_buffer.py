"""
Multimodal Experience Replay Buffer V8.0
Stores complete transition tuples with raw multimodal observation tensors to maintain
encoder-update compatibility across Stage 2 and Stage 3 training.
"""

from collections import deque
import random
from typing import List, Tuple, NamedTuple
import torch
from stock_rl.data.synthetic_data import MultimodalBatch


class Transition(NamedTuple):
    obs: MultimodalBatch
    action: int
    reward: float
    next_obs: MultimodalBatch
    done: bool


class MultimodalReplayBuffer:
    """
    Fixed-capacity ring buffer storing transition tuples.
    Collates raw tensors into single batched MultimodalBatch objects upon sampling.
    """
    def __init__(self, capacity: int = 100_000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        obs: MultimodalBatch,
        action: int,
        reward: float,
        next_obs: MultimodalBatch,
        done: bool
    ):
        self.buffer.append(Transition(obs, action, reward, next_obs, done))

    def __len__(self) -> int:
        return len(self.buffer)

    def sample(self, batch_size: int) -> Tuple[MultimodalBatch, torch.Tensor, torch.Tensor, MultimodalBatch, torch.Tensor]:
        """
        Samples a mini-batch of transitions.
        Returns:
            (batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones)
        """
        samples: List[Transition] = random.sample(self.buffer, batch_size)

        actions = torch.tensor([s.action for s in samples], dtype=torch.int64)
        rewards = torch.tensor([s.reward for s in samples], dtype=torch.float32)
        dones = torch.tensor([float(s.done) for s in samples], dtype=torch.float32)

        # Collate observations into batched MultimodalBatch
        batch_obs = self._collate_batches([s.obs for s in samples])
        batch_next_obs = self._collate_batches([s.next_obs for s in samples])

        return batch_obs, actions, rewards, batch_next_obs, dones

    @staticmethod
    def _collate_batches(batch_list: List[MultimodalBatch]) -> MultimodalBatch:
        """Concatenates a list of single-sample batches along dim 0."""
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
        )

"""Experience Replay Buffer for DQN."""

from __future__ import annotations

import random
from collections import deque
from typing import NamedTuple

import numpy as np
import torch


class Transition(NamedTuple):
    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    done: bool


class ReplayBuffer:
    """Fixed-size circular buffer storing environment transitions.

    Args:
        capacity: Maximum number of transitions to store.
    """

    def __init__(self, capacity: int) -> None:
        self._buffer: deque[Transition] = deque(maxlen=capacity)

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Add a transition to the buffer."""
        self._buffer.append(Transition(obs, action, reward, next_obs, done))

    def sample(
        self, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Randomly sample a mini-batch of transitions.

        Returns:
            Five tensors: (obs, actions, rewards, next_obs, dones), each of
            shape ``(batch_size, ...)``.
        """
        batch = random.sample(self._buffer, batch_size)
        obs = torch.tensor(
            np.array([t.obs for t in batch]), dtype=torch.float32, device=device
        )
        actions = torch.tensor(
            [t.action for t in batch], dtype=torch.long, device=device
        )
        rewards = torch.tensor(
            [t.reward for t in batch], dtype=torch.float32, device=device
        )
        next_obs = torch.tensor(
            np.array([t.next_obs for t in batch]), dtype=torch.float32, device=device
        )
        dones = torch.tensor(
            [t.done for t in batch], dtype=torch.float32, device=device
        )
        return obs, actions, rewards, next_obs, dones

    def __len__(self) -> int:
        return len(self._buffer)

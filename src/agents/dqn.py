"""Deep Q-Network (DQN) agent.

Reference: Mnih et al. (2015) "Human-level control through deep reinforcement
learning", Nature 518, 529–533.
"""

from __future__ import annotations

import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.networks.q_network import QNetwork
from src.utils.replay_buffer import ReplayBuffer


class DQNAgent:
    """DQN agent with experience replay and a target network.

    Args:
        obs_dim: Observation space dimensionality.
        action_dim: Number of discrete actions.
        lr: Learning rate for the Adam optimiser.
        gamma: Discount factor.
        epsilon_start: Initial ε for ε-greedy exploration.
        epsilon_end: Minimum ε.
        epsilon_decay: Multiplicative decay applied after every step.
        batch_size: Mini-batch size for gradient updates.
        buffer_capacity: Maximum replay buffer size.
        target_update_freq: Steps between hard target-network synchronisations.
        hidden_sizes: Hidden layer sizes for the Q-network.
        device: Torch device string (e.g. ``"cpu"`` or ``"cuda"``).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        buffer_capacity: int = 10_000,
        target_update_freq: int = 200,
        hidden_sizes: tuple[int, ...] = (128, 128),
        device: str = "cpu",
    ) -> None:
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = torch.device(device)

        self.online_net = QNetwork(obs_dim, action_dim, hidden_sizes).to(self.device)
        self.target_net = copy.deepcopy(self.online_net)
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self._step_count = 0

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray) -> int:
        """ε-greedy action selection.

        Args:
            obs: Current observation (1-D numpy array).

        Returns:
            Chosen action index.
        """
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online_net(obs_t)
        return int(q_values.argmax(dim=1).item())

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def store_transition(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Push a transition to the replay buffer."""
        self.replay_buffer.push(obs, action, reward, next_obs, done)

    def update(self) -> float | None:
        """Sample a mini-batch and perform one gradient-descent step.

        Returns:
            Scalar loss value, or ``None`` if the buffer is too small.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        obs, actions, rewards, next_obs, dones = self.replay_buffer.sample(
            self.batch_size, self.device
        )

        # Current Q-values for taken actions
        q_values = self.online_net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)

        # TD target using the frozen target network
        with torch.no_grad():
            next_q_values = self.target_net(next_obs).max(dim=1).values
            targets = rewards + self.gamma * next_q_values * (1.0 - dones)

        loss = self.loss_fn(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self._step_count += 1
        if self._step_count % self.target_update_freq == 0:
            self._sync_target()

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return loss.item()

    def _sync_target(self) -> None:
        """Hard-copy online network weights to the target network."""
        self.target_net.load_state_dict(self.online_net.state_dict())

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save online network weights to *path*."""
        torch.save(self.online_net.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path: str) -> None:
        """Load online (and target) network weights from *path*."""
        state = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(state)
        self._sync_target()
        print(f"Model loaded from {path}")

"""
d3qn_per_agent.py
=================
Dueling Double Deep Q-Network with Prioritised Experience Replay (D3QN-PER).

Key additions over vanilla D3QN:
  • Prioritised Replay: transitions are sampled proportional to their TD-error
    (priority = |r + γ·max Q(s', a') - Q(s, a)|).  This focuses learning on
    "surprising" or hard-to-predict transitions, speeding up convergence.

  • Sum-Tree data structure: enables O(log N) priority updates and sampling.

  • Importance-sampling weights: corrects the bias introduced by non-uniform
    sampling; weights are annealed from initial beta → 1.0.

Usage
-----
    from d3qn_per_agent import D3QNPERAgent
    agent = D3QNPERAgent(obs_dim=72, n_actions=5, **hyperparams)
    action = agent.select_action(obs)
    agent.store(obs, action, reward, next_obs, done)
    loss = agent.learn()
"""

from __future__ import annotations

import os
import random
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Sum-Tree for Prioritised Sampling
# ═══════════════════════════════════════════════════════════════════════════════

class SumTree:
    """
    Sum-Tree data structure for efficient prioritised sampling.

    A binary tree where each parent node is the sum of its children.
    The leaf nodes store transition priorities.

    Operations:
      • push(priority)     : O(log N) to insert with a given priority.
      • sample(batch_size) : O(batch_size · log N) to draw priorities.
      • update(idx, priority) : O(log N) to update a single priority.

    Memory: O(N) for N transitions (slightly more than a flat array).
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        # Tree has 2*capacity nodes: capacity leaves + capacity-1 internal nodes.
        self.tree = np.zeros(2 * capacity - 1)
        self.data_ptr = 0
        self.filled = 0

    def push(self, priority: float) -> int:
        """
        Add a transition with the given priority.  Returns its leaf index.
        """
        leaf_idx = self.capacity - 1 + self.data_ptr
        self.data_ptr = (self.data_ptr + 1) % self.capacity
        self.filled = min(self.filled + 1, self.capacity)
        self._set_priority(leaf_idx, priority)
        return leaf_idx

    def update(self, leaf_idx: int, priority: float) -> None:
        """Update the priority of a leaf and propagate changes upward."""
        self._set_priority(leaf_idx, priority)

    def _set_priority(self, leaf_idx: int, priority: float) -> None:
        """Internal: set leaf priority and update all ancestors."""
        delta = priority - self.tree[leaf_idx]
        self.tree[leaf_idx] = priority
        parent_idx = (leaf_idx - 1) // 2
        while parent_idx >= 0:
            self.tree[parent_idx] += delta
            parent_idx = (parent_idx - 1) // 2

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample batch_size leaf indices proportional to their priorities.

        Returns:
          indices  : shape (batch_size,)
          priorities : shape (batch_size,) — the sampled priorities
        """
        root_sum = self.tree[0]
        if root_sum <= 0:
            # No priorities set; fall back to uniform.
            return (
                np.random.randint(0, self.filled, size=batch_size),
                np.ones(batch_size),
            )

        indices = []
        priorities = []
        segment = root_sum / batch_size

        for i in range(batch_size):
            # Sample uniformly within segment [i*segment, (i+1)*segment).
            target = random.uniform(segment * i, segment * (i + 1))
            leaf_idx = self._find_leaf(target)
            indices.append(leaf_idx - (self.capacity - 1))  # Convert to data index.
            priorities.append(self.tree[leaf_idx])

        return np.array(indices), np.array(priorities)

    def _find_leaf(self, target: float) -> int:
        """Find the leaf node whose cumulative sum includes target."""
        idx = 0
        while idx < self.capacity - 1:
            left_child = 2 * idx + 1
            right_child = 2 * idx + 2
            if self.tree[left_child] >= target:
                idx = left_child
            else:
                target -= self.tree[left_child]
                idx = right_child
        return idx

    def max_priority(self) -> float:
        """Return the maximum priority in the tree."""
        return np.max(self.tree[self.capacity - 1 :])

    def get_filled(self) -> int:
        """Return the number of transitions stored."""
        return self.filled


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Prioritised Replay Buffer
# ═══════════════════════════════════════════════════════════════════════════════

class PrioritisedReplayBuffer:
    """
    Experience replay buffer with prioritised sampling.

    Each transition (obs, action, reward, next_obs, done) is assigned a priority
    proportional to its TD-error magnitude.  Transitions with higher TD-error are
    sampled more often, focusing learning on harder-to-predict states.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions to store.
    obs_dim : int
        Observation dimension.
    device : torch.device
        Device for tensor operations.
    alpha : float
        Prioritisation exponent. alpha=0 → uniform replay. alpha=1 → full PER.
        Typical: 0.6 (allows both high- and low-priority samples).
    beta : float
        Initial importance-sampling exponent.  beta=0 → no correction.
        beta=1 → full IS correction.  Usually start at 0.4, anneal to 1.0.
    beta_anneal_steps : int
        Steps to anneal beta from initial value to 1.0.
    epsilon : float
        Small constant added to priorities to ensure all transitions have
        non-zero probability.  Typical: 1e-6.
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        device: torch.device,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_anneal_steps: int = 100_000,
        epsilon: float = 1e-6,
    ):
        self.capacity = capacity
        self.device = device
        self.obs_dim = obs_dim
        self.alpha = alpha
        self.beta = beta
        self.beta_start = beta
        self.beta_anneal_steps = beta_anneal_steps
        self.epsilon = epsilon

        # Pre-allocated storage.
        self.obs      = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions  = np.zeros(capacity, dtype=np.int64)
        self.rewards  = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones    = np.zeros(capacity, dtype=np.float32)

        # Sum-tree for priorities.
        self.sum_tree = SumTree(capacity)

        self.step_count = 0  # For beta annealing.

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """
        Store a transition with maximum priority (so new transitions are
        sampled quickly at least once).
        """
        idx = self.sum_tree.data_ptr
        self.obs[idx] = obs
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_obs[idx] = next_obs
        self.dones[idx] = float(done)

        # Assign max priority to encourage sampling new transitions.
        max_p = self.sum_tree.max_priority()
        if max_p <= 0:
            max_p = 1.0
        self.sum_tree.push(max_p)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Sample a prioritised mini-batch.

        Returns:
          obs, actions, rewards, next_obs, dones : tensors
          indices : leaf indices in sum-tree (needed for priority updates)
          weights : importance-sampling weights (for correcting bias)
        """
        # Sample indices based on priorities.
        indices, priorities = self.sum_tree.sample(batch_size)

        # Clip to valid range (in case buffer is not yet full).
        indices = np.clip(indices, 0, self.sum_tree.filled - 1)

        # Compute importance-sampling weights.
        # w_i = (1 / (N · P_i))^β, normalised by max weight for stability.
        N = self.sum_tree.filled
        ps = priorities / (self.sum_tree.tree[0] + self.epsilon)
        weights = (N * ps) ** (-self.beta)
        weights /= weights.max()  # Normalise for stability.
        weights = torch.FloatTensor(weights).to(self.device)

        # Convert batch indices to leaf indices for later update.
        leaf_indices = indices + (self.capacity - 1)

        # Return sampled transitions.
        return (
            torch.FloatTensor(self.obs[indices]).to(self.device),
            torch.LongTensor(self.actions[indices]).to(self.device),
            torch.FloatTensor(self.rewards[indices]).to(self.device),
            torch.FloatTensor(self.next_obs[indices]).to(self.device),
            torch.FloatTensor(self.dones[indices]).to(self.device),
            leaf_indices,
            weights,
        )

    def update_priorities(
        self, leaf_indices: np.ndarray, td_errors: np.ndarray
    ) -> None:
        """
        Update priorities based on TD-errors.

        Parameters
        ----------
        leaf_indices : np.ndarray of int
            Sum-tree leaf indices (returned by sample()).
        td_errors : np.ndarray of float
            Absolute TD-errors |r + γ·max Q(s', a') - Q(s, a)|.
        """
        new_priorities = (np.abs(td_errors) + self.epsilon) ** self.alpha
        for leaf_idx, priority in zip(leaf_indices, new_priorities):
            self.sum_tree.update(leaf_idx, priority)

    def anneal_beta(self) -> None:
        """Anneal beta towards 1.0 over beta_anneal_steps."""
        self.beta = min(
            1.0,
            self.beta_start + (1.0 - self.beta_start) * self.step_count / self.beta_anneal_steps,
        )
        self.step_count += 1

    def __len__(self) -> int:
        return self.sum_tree.get_filled()


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Dueling Network Architecture
# ═══════════════════════════════════════════════════════════════════════════════

class DuelingDQN(nn.Module):
    """Dueling DQN network with separate value and advantage heads."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_dim: int = 256,
        n_layers: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions

        # Shared feature trunk.
        layers = []
        in_dim = obs_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            in_dim = hidden_dim
        self.shared = nn.Sequential(*layers)

        # Value head.
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Advantage head.
        self.advantage_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_actions),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """He initialisation for ReLU networks."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return Q-values for all actions."""
        features = self.shared(x)
        value = self.value_head(features)
        advantage = self.advantage_head(features)
        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  D3QN-PER Agent
# ═══════════════════════════════════════════════════════════════════════════════

class D3QNPERAgent:
    """
    Dueling Double DQN agent with Prioritised Experience Replay.

    Combines D3QN algorithmic improvements with PER's sample efficiency.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        # ── Network ──────────────────────────────────────────────────────
        hidden_dim: int = 256,
        n_layers: int = 3,
        dropout: float = 0.0,
        # ── Optimisation ─────────────────────────────────────────────────
        lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        # ── Replay ───────────────────────────────────────────────────────
        buffer_capacity: int = 100_000,
        batch_size: int = 64,
        learn_every: int = 4,
        target_update_freq: int = 1,
        # ── PER parameters ───────────────────────────────────────────────
        per_alpha: float = 0.6,
        per_beta: float = 0.4,
        per_beta_anneal_steps: int = 100_000,
        # ── Exploration ───────────────────────────────────────────────────
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay_steps: int = 50_000,
        stuck_turn_prob: float = 0.7,
        stuck_epsilon_boost: float = 0.25,
        avoid_forward_when_stuck: bool = True,
        # ── Misc ──────────────────────────────────────────────────────────
        grad_clip: float = 10.0,
        device: str = "auto",
        seed: Optional[int] = None,
    ):
        # ── Seeding ───────────────────────────────────────────────────────
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        # ── Device ────────────────────────────────────────────────────────
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.learn_every = learn_every
        self.target_update_freq = target_update_freq
        self.grad_clip = grad_clip

        # ── Networks ──────────────────────────────────────────────────────
        net_kwargs = dict(
            obs_dim=obs_dim,
            n_actions=n_actions,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
        )
        self.online_net = DuelingDQN(**net_kwargs).to(self.device)
        self.target_net = DuelingDQN(**net_kwargs).to(self.device)
        self._hard_update()
        self.target_net.eval()

        # ── Optimiser ─────────────────────────────────────────────────────
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)

        # ── Prioritised replay buffer ─────────────────────────────────────
        self.buffer = PrioritisedReplayBuffer(
            capacity=buffer_capacity,
            obs_dim=obs_dim,
            device=self.device,
            alpha=per_alpha,
            beta=per_beta,
            beta_anneal_steps=per_beta_anneal_steps,
        )

        # ── Exploration schedule ──────────────────────────────────────────
        self.eps = eps_start
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = (eps_start - eps_end) / max(1, eps_decay_steps)
        self.stuck_turn_prob = float(np.clip(stuck_turn_prob, 0.0, 1.0))
        self.stuck_epsilon_boost = float(np.clip(stuck_epsilon_boost, 0.0, 1.0))
        self.avoid_forward_when_stuck = avoid_forward_when_stuck

        # ── Counters ──────────────────────────────────────────────────────
        self.total_steps = 0
        self.learn_steps = 0

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def select_action(self, obs: np.ndarray) -> int:
        """ε-greedy action selection with annealing."""
        is_stuck = bool(obs[-1] > 0.5)
        effective_eps = max(self.eps, self.stuck_epsilon_boost) if is_stuck else self.eps

        if is_stuck and self.avoid_forward_when_stuck and random.random() < self.stuck_turn_prob:
            action = random.choice([0, 1, 3, 4])
        elif random.random() < effective_eps:
            action = random.randrange(self.n_actions)
            if is_stuck and self.avoid_forward_when_stuck and action == 2:
                action = random.choice([0, 1, 3, 4])
        else:
            action = self._greedy_action(obs, avoid_forward_if_stuck=is_stuck and self.avoid_forward_when_stuck)

        self.eps = max(self.eps_end, self.eps - self.eps_decay)
        self.total_steps += 1
        return action

    def select_greedy_action(self, obs: np.ndarray) -> int:
        """Pure greedy action (for evaluation)."""
        is_stuck = bool(obs[-1] > 0.5)
        return self._greedy_action(obs, avoid_forward_if_stuck=is_stuck and self.avoid_forward_when_stuck)

    def store(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition in the prioritised replay buffer."""
        self.buffer.push(obs, action, reward, next_obs, done)

    def learn(self) -> Optional[float]:
        """
        Sample a prioritised mini-batch and perform one gradient update.

        Uses importance-sampling weights to correct for the bias introduced
        by non-uniform sampling.
        """
        if len(self.buffer) < self.batch_size:
            return None

        (
            obs,
            actions,
            rewards,
            next_obs,
            dones,
            leaf_indices,
            is_weights,
        ) = self.buffer.sample(self.batch_size)

        # ── Double DQN target ────────────────────────────────────────────
        with torch.no_grad():
            next_actions = self.online_net(next_obs).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_obs).gather(1, next_actions).squeeze(1)
            target_q = rewards + self.gamma * next_q * (1.0 - dones)

        # ── Online Q values ───────────────────────────────────────────────
        current_q = self.online_net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)

        # ── TD-error and importance-sampled loss ──────────────────────────
        td_error = current_q - target_q
        loss = (is_weights * F.smooth_l1_loss(current_q, target_q, reduction="none")).mean()

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.online_net.parameters(), self.grad_clip)
        self.optimizer.step()

        self.learn_steps += 1

        # ── Update priorities based on TD-errors ──────────────────────────
        self.buffer.update_priorities(leaf_indices, td_error.detach().cpu().numpy())

        # ── Anneal beta (importance-sampling correction) ──────────────────
        self.buffer.anneal_beta()

        # ── Soft target update ────────────────────────────────────────────
        if self.learn_steps % self.target_update_freq == 0:
            self._soft_update()

        return loss.item()

    def maybe_learn(self) -> Optional[float]:
        """Throttle learning to once every `learn_every` environment steps."""
        if self.total_steps % self.learn_every == 0:
            return self.learn()
        return None

    # ──────────────────────────────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save checkpoint."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save(
            {
                "online_state_dict": self.online_net.state_dict(),
                "target_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "total_steps": self.total_steps,
                "learn_steps": self.learn_steps,
                "eps": self.eps,
                "buffer_beta": self.buffer.beta,
            },
            path,
        )
        print(f"[D3QN-PER] Checkpoint saved → {path}")

    def load(self, path: str) -> None:
        """Load checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt["online_state_dict"])
        self.target_net.load_state_dict(ckpt["target_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.total_steps = ckpt.get("total_steps", 0)
        self.learn_steps = ckpt.get("learn_steps", 0)
        self.eps = ckpt.get("eps", self.eps_end)
        self.buffer.beta = ckpt.get("buffer_beta", self.buffer.beta)
        print(f"[D3QN-PER] Checkpoint loaded ← {path}")

    # ──────────────────────────────────────────────────────────────────────
    # Private
    # ──────────────────────────────────────────────────────────────────────

    def _greedy_action(self, obs: np.ndarray, avoid_forward_if_stuck: bool = False) -> int:
        """Return argmax Q-value action."""
        self.online_net.eval()
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            q_values = self.online_net(obs_t)
        self.online_net.train()

        if avoid_forward_if_stuck:
            q_values = q_values.clone()
            q_values[:, 2] = -1e9

        return int(q_values.argmax(dim=1).item())

    def _soft_update(self) -> None:
        """Polyak averaging."""
        for p_target, p_online in zip(
            self.target_net.parameters(), self.online_net.parameters()
        ):
            p_target.data.copy_(
                self.tau * p_online.data + (1.0 - self.tau) * p_target.data
            )

    def _hard_update(self) -> None:
        """Full copy."""
        self.target_net.load_state_dict(self.online_net.state_dict())

    # ──────────────────────────────────────────────────────────────────────
    # Diagnostics
    # ──────────────────────────────────────────────────────────────────────

    def get_q_values(self, obs: np.ndarray) -> np.ndarray:
        """Return Q-values for debugging."""
        self.online_net.eval()
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            q = self.online_net(obs_t).squeeze(0).cpu().numpy()
        self.online_net.train()
        return q

"""
q_tile_agent.py
===============
Q-learning agent with tile coding for the OBELIX environment.

Tile coding groups the 18 binary observation bits into meaningful feature
tiles, reducing the effective state space from 2^18 = 262,144 to a
manageable set of tile combinations that can be learned from.

Tile groups for OBELIX 18-bit obs:
    bits 0-7   : front sonar sector (8 bits)
    bits 8-11  : left sonar sector  (4 bits)
    bits 12-15 : right sonar sector (4 bits)
    bit  16    : IR sensor (box detected)
    bit  17    : stuck flag

Each tile group is hashed to an integer index. The full state key is a
tuple of tile indices — this is the dict key in the Q-table.

Public API (mirrors D3QNPERAgent for drop-in use in train file):
    agent = QTileAgent(obs_dim=18, n_actions=5, **kwargs)
    action = agent.select_action(obs)
    action = agent.select_greedy_action(obs)
    agent.store(obs, action, reward, next_obs, done)   # triggers learn()
    loss   = agent.maybe_learn()                       # returns TD error
    agent.save("path/checkpoint.pkl")
    agent.load("path/checkpoint.pkl")
"""

from __future__ import annotations

import os
import pickle
import random
from collections import defaultdict
from typing import Optional

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Tile coding
# ═══════════════════════════════════════════════════════════════════════════

# Define which bits belong to each tile group.
# Each group is converted to an integer (binary → int) to form the state key.
TILE_GROUPS = [
    list(range(0, 8)),    # front sonar   (8 bits → 0..255)
    list(range(8, 12)),   # left sonar    (4 bits → 0..15)
    list(range(12, 16)),  # right sonar   (4 bits → 0..15)
    [16],                 # IR sensor     (1 bit  → 0..1)
    [17],                 # stuck flag    (1 bit  → 0..1)
]

# Max possible tile combinations: 256 × 16 × 16 × 2 × 2 = 262,144
# In practice only a small fraction are visited.


def obs_to_state_key(obs: np.ndarray) -> tuple:
    """
    Convert a raw 18-bit observation to a tuple of tile indices.

    Each tile group's bits are packed into an integer.
    The tuple is used directly as the Q-table dictionary key.

    Parameters
    ----------
    obs : np.ndarray, shape (18,) or (72,) with frame stacking
        Raw OBELIX observation. If frame-stacked, only the most recent
        18-bit frame (last slice) is used for tile coding.

    Returns
    -------
    tuple of ints, one per tile group
    """
    # Use only the most recent frame if stacked
    raw = obs[-18:].astype(int)

    key = []
    for group in TILE_GROUPS:
        bits = [int(raw[i]) for i in group]
        # Pack bits into integer: bit[0] is MSB
        val = 0
        for b in bits:
            val = (val << 1) | b
        key.append(val)

    return tuple(key)


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Q-Table
# ═══════════════════════════════════════════════════════════════════════════

class QTable:
    """
    Dictionary-based Q-table with optimistic initialisation.

    Unvisited (state, action) pairs return `initial_value` (optimistic),
    which encourages exploration of new states.
    """

    def __init__(self, n_actions: int, initial_value: float = 0.0):
        self.n_actions     = n_actions
        self.initial_value = initial_value
        # table[state_key] = np.array of shape (n_actions,)
        self._table: dict = {}

    def get(self, state_key: tuple) -> np.ndarray:
        if state_key not in self._table:
            self._table[state_key] = np.full(
                self.n_actions, self.initial_value, dtype=np.float64
            )
        return self._table[state_key]

    def update(self, state_key: tuple, action: int, value: float) -> None:
        self.get(state_key)[action] = value

    def max_q(self, state_key: tuple) -> float:
        return float(np.max(self.get(state_key)))

    def argmax_q(self, state_key: tuple) -> int:
        return int(np.argmax(self.get(state_key)))

    def __len__(self) -> int:
        return len(self._table)


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Q-Tile Agent
# ═══════════════════════════════════════════════════════════════════════════

class QTileAgent:
    """
    Tabular Q-learning agent with tile-coded state representation.

    Parameters
    ----------
    obs_dim : int
        Raw observation dimension (18 for OBELIX, or 72 with frame stacking).
        Only the last 18 bits are used for tile coding.
    n_actions : int
        Number of discrete actions (5 for OBELIX).
    lr : float
        Learning rate alpha. Typical: 0.1–0.3 for tabular Q-learning.
    gamma : float
        Discount factor. Typical: 0.99.
    eps_start : float
        Initial epsilon for epsilon-greedy exploration.
    eps_end : float
        Minimum epsilon.
    eps_decay_steps : int
        Number of steps to decay epsilon from start to end.
    initial_q : float
        Optimistic initial Q-value. Positive values (e.g. 1.0) encourage
        exploration of unvisited states.
    seed : int | None
        Random seed.
    """

    def __init__(
        self,
        obs_dim: int = 18,
        n_actions: int = 5,
        lr: float = 0.2,
        gamma: float = 0.99,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay_steps: int = 50_000,
        initial_q: float = 0.0,
        seed: Optional[int] = None,
    ):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.obs_dim        = obs_dim
        self.n_actions      = n_actions
        self.lr             = lr
        self.gamma          = gamma
        self.eps            = eps_start
        self.eps_start      = eps_start
        self.eps_end        = eps_end
        self.eps_decay      = (eps_start - eps_end) / max(1, eps_decay_steps)

        self.q_table        = QTable(n_actions, initial_value=initial_q)

        # Pending transition for learn step
        self._pending: Optional[tuple] = None

        # Counters
        self.total_steps    = 0
        self.learn_steps    = 0

    # ── Action selection ──────────────────────────────────────────────────

    def select_action(self, obs: np.ndarray) -> int:
        """Epsilon-greedy action selection."""
        if random.random() < self.eps:
            action = random.randrange(self.n_actions)
        else:
            state_key = obs_to_state_key(obs)
            action    = self.q_table.argmax_q(state_key)

        # Decay epsilon
        self.eps = max(self.eps_end, self.eps - self.eps_decay)
        self.total_steps += 1
        return action

    def select_greedy_action(self, obs: np.ndarray) -> int:
        """Pure greedy action (for evaluation)."""
        state_key = obs_to_state_key(obs)
        return self.q_table.argmax_q(state_key)

    # ── Experience storage ────────────────────────────────────────────────

    def store(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition. Q-learning updates immediately on each step."""
        self._pending = (obs, action, reward, next_obs, done)

    # ── Learning ──────────────────────────────────────────────────────────

    def maybe_learn(self) -> Optional[float]:
        """Perform a Q-learning update on the pending transition."""
        if self._pending is None:
            return None
        return self.learn()

    def learn(self) -> Optional[float]:
        """
        Tabular Q-learning update:
            Q(s,a) ← Q(s,a) + α · [r + γ · max_a' Q(s',a') − Q(s,a)]
        """
        if self._pending is None:
            return None

        obs, action, reward, next_obs, done = self._pending
        self._pending = None

        state_key      = obs_to_state_key(obs)
        next_state_key = obs_to_state_key(next_obs)

        current_q = self.q_table.get(state_key)[action]

        if done:
            target_q = reward
        else:
            target_q = reward + self.gamma * self.q_table.max_q(next_state_key)

        td_error = target_q - current_q
        new_q    = current_q + self.lr * td_error

        self.q_table.update(state_key, action, new_q)
        self.learn_steps += 1

        return float(abs(td_error))

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save Q-table and training state to a pickle file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        payload = {
            "q_table":     self.q_table._table,
            "eps":         self.eps,
            "total_steps": self.total_steps,
            "learn_steps": self.learn_steps,
            "lr":          self.lr,
            "gamma":       self.gamma,
            "n_actions":   self.n_actions,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        print(f"[QTile] Checkpoint saved → {path}  "
              f"(states={len(self.q_table)}, steps={self.total_steps})")

    def load(self, path: str) -> None:
        """Load Q-table and training state from a pickle file."""
        with open(path, "rb") as f:
            payload = pickle.load(f)
        self.q_table._table = payload["q_table"]
        self.eps             = payload["eps"]
        self.total_steps     = payload["total_steps"]
        self.learn_steps     = payload["learn_steps"]
        self.lr              = payload.get("lr", self.lr)
        self.gamma           = payload.get("gamma", self.gamma)
        print(f"[QTile] Checkpoint loaded ← {path}  "
              f"(states={len(self.q_table)}, steps={self.total_steps})")

    # ── Diagnostics ───────────────────────────────────────────────────────

    def get_q_values(self, obs: np.ndarray) -> np.ndarray:
        state_key = obs_to_state_key(obs)
        return self.q_table.get(state_key).copy()

    @property
    def n_states_visited(self) -> int:
        return len(self.q_table)

"""REINFORCE (Monte-Carlo Policy Gradient) agent.

Reference: Williams, R. J. (1992) "Simple statistical gradient-following
algorithms for connectionist reinforcement learning", Machine Learning 8,
229–256.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.optim as optim

from src.networks.policy_network import PolicyNetwork


class REINFORCEAgent:
    """REINFORCE agent with discounted return baseline (mean subtraction).

    Args:
        obs_dim: Observation space dimensionality.
        action_dim: Number of discrete actions.
        lr: Learning rate for the Adam optimiser.
        gamma: Discount factor.
        hidden_sizes: Hidden layer sizes for the policy network.
        device: Torch device string (e.g. ``"cpu"`` or ``"cuda"``).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        hidden_sizes: tuple[int, ...] = (128, 128),
        device: str = "cpu",
    ) -> None:
        self.gamma = gamma
        self.device = torch.device(device)

        self.policy = PolicyNetwork(obs_dim, action_dim, hidden_sizes).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Episode storage (reset at the start of each episode)
        self._log_probs: list[torch.Tensor] = []
        self._rewards: list[float] = []

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray) -> int:
        """Sample an action from the policy and store the log-probability.

        Args:
            obs: Current observation (1-D numpy array).

        Returns:
            Chosen action index.
        """
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        action, log_prob = self.policy.act(obs_t)
        self._log_probs.append(log_prob)
        return action

    def store_reward(self, reward: float) -> None:
        """Append the reward received at the current step."""
        self._rewards.append(reward)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def update(self) -> float:
        """Compute the policy-gradient loss and perform one optimiser step.

        Should be called **once per episode** after the episode terminates.

        Returns:
            Scalar policy-gradient loss (positive value).
        """
        # Compute discounted returns G_t for each time step
        returns: list[float] = []
        g = 0.0
        for r in reversed(self._rewards):
            g = r + self.gamma * g
            returns.insert(0, g)

        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        # Normalise returns for variance reduction
        if returns_t.std() > 1e-8:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        # Policy-gradient loss: -E[log π(a|s) · G_t]
        log_probs_t = torch.stack(self._log_probs)
        loss = -(log_probs_t * returns_t).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Reset episode storage
        self._log_probs = []
        self._rewards = []

        return loss.item()

    def reset_episode(self) -> None:
        """Discard stored episode data without updating the policy."""
        self._log_probs = []
        self._rewards = []

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save policy network weights to *path*."""
        torch.save(self.policy.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path: str) -> None:
        """Load policy network weights from *path*."""
        state = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(state)
        print(f"Model loaded from {path}")

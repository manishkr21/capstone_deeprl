"""Q-Network for Deep Q-Network (DQN) agent."""

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """Fully-connected Q-value network.

    Maps an observation vector to Q-values for every discrete action.

    Args:
        obs_dim: Dimensionality of the observation space.
        action_dim: Number of discrete actions.
        hidden_sizes: Sizes of hidden layers (default: [128, 128]).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return Q-values for each action given an observation batch."""
        return self.net(x)

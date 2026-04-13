"""Policy Network for REINFORCE (policy gradient) agent."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    """Fully-connected stochastic policy network.

    Outputs a categorical probability distribution over discrete actions.

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
        """Return action log-probabilities given an observation batch."""
        logits = self.net(x)
        return F.log_softmax(logits, dim=-1)

    def act(self, obs: torch.Tensor) -> tuple[int, torch.Tensor]:
        """Sample an action and return (action, log_prob).

        Args:
            obs: 1-D observation tensor (single environment step).

        Returns:
            Tuple of (action index, log probability of chosen action).
        """
        log_probs = self.forward(obs.unsqueeze(0)).squeeze(0)  # shape: (action_dim,)
        probs = log_probs.exp()
        action = torch.multinomial(probs, num_samples=1).item()
        return int(action), log_probs[action]

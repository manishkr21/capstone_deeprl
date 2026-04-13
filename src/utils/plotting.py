"""Plotting utilities for training reward curves."""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")  # non-interactive backend – saves figures without a display


def plot_rewards(
    rewards: list[float],
    title: str = "Training Rewards",
    window: int = 50,
    save_path: str | None = None,
) -> None:
    """Plot episode rewards with a rolling-average smoothing overlay.

    Always renders to a file when *save_path* is given.  When *save_path* is
    ``None`` the figure is saved to ``"rewards.png"`` in the current directory
    (the Agg backend does not support interactive display).

    Args:
        rewards: Per-episode total rewards.
        title: Plot title.
        window: Window size for the rolling average.
        save_path: Destination PNG path.  Defaults to ``"rewards.png"`` when
            ``None``.
    """
    episodes = np.arange(1, len(rewards) + 1)
    rolling_mean = (
        np.convolve(rewards, np.ones(window) / window, mode="valid")
        if len(rewards) >= window
        else np.array(rewards)
    )
    rolling_episodes = episodes[window - 1 :] if len(rewards) >= window else episodes

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, rewards, alpha=0.3, color="steelblue", label="Episode reward")
    ax.plot(
        rolling_episodes,
        rolling_mean,
        color="steelblue",
        linewidth=2,
        label=f"Rolling avg (w={window})",
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")
    else:
        default_path = "rewards.png"
        fig.savefig(default_path, dpi=150)
        print(f"Plot saved to {default_path}")
    plt.close(fig)

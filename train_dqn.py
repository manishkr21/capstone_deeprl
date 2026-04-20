"""Train a DQN agent on a Gymnasium environment.

Usage
-----
    python train_dqn.py                          # CartPole-v1, default params
    python train_dqn.py --env LunarLander-v2
    python train_dqn.py --help
"""

from __future__ import annotations

import argparse
import os

import gymnasium as gym
import numpy as np

from src.agents.dqn import DQNAgent
from src.utils.plotting import plot_rewards


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a DQN agent on a Gymnasium environment."
    )
    parser.add_argument("--env", default="CartPole-v1", help="Gymnasium environment ID")
    parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.01)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-capacity", type=int, default=10_000)
    parser.add_argument("--target-update-freq", type=int, default=200)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--device", default="cpu", help="Torch device (cpu/cuda)")
    parser.add_argument("--save-model", default="dqn_model.pt", help="Path to save final model")
    parser.add_argument("--plot", default="dqn_rewards.png", help="Path to save reward plot")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--solve-threshold", type=float, default=None,
                        help="Stop early when rolling mean reward exceeds this value")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    env = gym.make(args.env)
    np.random.seed(args.seed)

    obs_dim: int = int(np.prod(env.observation_space.shape))
    action_dim: int = env.action_space.n  # type: ignore[attr-defined]

    agent = DQNAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        lr=args.lr,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        batch_size=args.batch_size,
        buffer_capacity=args.buffer_capacity,
        target_update_freq=args.target_update_freq,
        hidden_sizes=tuple(args.hidden_sizes),
        device=args.device,
    )

    episode_rewards: list[float] = []
    print(f"Training DQN on {args.env} for {args.episodes} episodes …")

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset(seed=args.seed + ep)
        total_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.store_transition(obs, action, float(reward), next_obs, done)
            agent.update()
            obs = next_obs
            total_reward += float(reward)

        episode_rewards.append(total_reward)

        if ep % 50 == 0 or ep == 1:
            window = min(50, ep)
            mean_r = float(np.mean(episode_rewards[-window:]))
            print(
                f"  Episode {ep:4d}/{args.episodes} | "
                f"reward {total_reward:8.2f} | "
                f"mean({window}) {mean_r:8.2f} | "
                f"ε {agent.epsilon:.3f}"
            )

        if args.solve_threshold is not None and len(episode_rewards) >= 100:
            if float(np.mean(episode_rewards[-100:])) >= args.solve_threshold:
                print(f"Solved at episode {ep} (mean reward ≥ {args.solve_threshold})!")
                break

    env.close()

    # Persist artefacts
    os.makedirs(os.path.dirname(args.save_model) or ".", exist_ok=True)
    agent.save(args.save_model)
    plot_rewards(episode_rewards, title=f"DQN – {args.env}", save_path=args.plot)
    print("Training complete.")


if __name__ == "__main__":
    main()

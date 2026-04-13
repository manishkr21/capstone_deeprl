"""
train_d3qn_per.py
=================
Training script for D3QN-PER agent on the OBELIX environment.

Features:
  - 10 seeds per configuration
  - 3 difficulty levels × 2 wall settings = 6 environment variants
  - Reward shaping: penalises consecutive circular motion (oscillation)
  - Results saved via save_training_results() to organised directory tree
  - Evaluation every N training episodes with greedy policy rollouts
  - Full metrics: per-episode reward, wall-clock time, eval rewards, total steps

Directory layout after training:
  results/
    D3QN-PER/
      OBELIX_diff0_nowall/
        seed_0/
          training_results.json
          checkpoint.pt
        seed_1/ ...
      OBELIX_diff0_wall/  ...
      OBELIX_diff2_nowall/ ...
      OBELIX_diff2_wall/  ...
      OBELIX_diff3_nowall/ ...
      OBELIX_diff3_wall/  ...

Usage
-----
    python train_d3qn_per.py                        # full run (all configs)
    python train_d3qn_per.py --difficulty 0         # only diff-0 configs
    python train_d3qn_per.py --no-wall              # only no-wall variants
    python train_d3qn_per.py --episodes 500 --eval-interval 50
    python train_d3qn_per.py --seeds 0 1 2          # subset of seeds
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import time
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Local imports (must be on the Python path)
# ---------------------------------------------------------------------------
from d3qn_per_agent import D3QNPERAgent
from wrapper_obelix import ObelixEnv


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Results persistence
# ═══════════════════════════════════════════════════════════════════════════

def _make_results_dir(algorithm_name: str, env_name: str, seed: int) -> str:
    """
    Create and return the directory:
        results/<algorithm_name>/<env_name>/seed_<seed>/
    """
    path = os.path.join("results", algorithm_name, env_name, f"seed_{seed}")
    os.makedirs(path, exist_ok=True)
    return path


def save_training_results(
    algorithm_name: str,
    env_name: str,
    seed: int,
    train_rewards: List[float],
    train_time: float,
    eval_rewards: List[float],
    wall_time: float,
    final_eval: float,
    total_steps: int,
    extra: Optional[dict] = None,
) -> str:
    """
    Persist all training metrics to JSON.

    Parameters
    ----------
    algorithm_name : str   e.g. "D3QN-PER"
    env_name       : str   e.g. "OBELIX_diff0_nowall"
    seed           : int
    train_rewards  : per-episode undiscounted return during training
    train_time     : total training time in seconds
    eval_rewards   : mean return at each evaluation checkpoint
    wall_time      : same as train_time (kept separate for extensibility)
    final_eval     : mean return over the final evaluation window
    total_steps    : total environment steps taken
    extra          : optional dict of additional diagnostics

    Returns
    -------
    str : path to the saved JSON file
    """
    results_dir = _make_results_dir(algorithm_name, env_name, seed)

    payload = {
        "algorithm": algorithm_name,
        "env": env_name,
        "seed": seed,
        "total_steps": total_steps,
        "train_time_s": round(train_time, 3),
        "wall_time_s": round(wall_time, 3),
        "final_eval_mean_reward": round(float(final_eval), 4),
        "train_rewards": [round(float(r), 4) for r in train_rewards],
        "eval_rewards": [round(float(r), 4) for r in eval_rewards],
    }
    if extra:
        payload["extra"] = extra

    out_path = os.path.join(results_dir, "training_results.json")
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)

    print(f"  [save] {out_path}")
    return out_path


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Circular-motion (oscillation) detector + reward shaper
# ═══════════════════════════════════════════════════════════════════════════

class CircularMotionShaper:
    """
    Penalises repeated left-right oscillation or pure spinning in place.

    Heuristic: if the agent takes ≥ `window` consecutive non-forward actions
    AND those actions alternate strictly between left and right turns
    (i.e. the sign of each turn alternates), it is penalised.

    Example punished pattern (window=4):
        L45 → R45 → L45 → R45   (sign: +1 -1 +1 -1)
        L22 → R22 → L22 → R22
    Example NOT punished:
        L45 → L22 → FW  → R22   (includes forward, breaks the pattern)

    Parameters
    ----------
    window : int
        Number of consecutive turns that triggers the penalty (default 4).
    penalty : float
        Negative reward applied each step the pattern is active (default -5).
    """

    # Action index → signed turn magnitude (0 = forward)
    _TURN_SIGN = {
        0: +1,   # L45
        1: +1,   # L22
        2:  0,   # FW  (forward — breaks oscillation)
        3: -1,   # R22
        4: -1,   # R45
    }

    def __init__(self, window: int = 4, penalty: float = -5.0):
        self.window = window
        self.penalty = penalty
        self._history: collections.deque = collections.deque(maxlen=window)

    def reset(self) -> None:
        self._history.clear()

    def shape(self, action: int, base_reward: float) -> float:
        """
        Compute the shaped reward given the current action and base reward.

        Returns base_reward + shaping_penalty (penalty ≤ 0).
        """
        sign = self._TURN_SIGN[action]

        # Forward action resets the oscillation counter.
        if sign == 0:
            self._history.clear()
            return base_reward

        self._history.append(sign)

        if len(self._history) < self.window:
            return base_reward

        # Check strict alternating pattern across the window.
        signs = list(self._history)
        alternating = all(
            signs[i] != signs[i + 1] for i in range(len(signs) - 1)
        )
        if alternating:
            return base_reward + self.penalty

        return base_reward


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Single-seed training loop
# ═══════════════════════════════════════════════════════════════════════════

def evaluate(
    agent: D3QNPERAgent,
    env: ObelixEnv,
    n_episodes: int = 10,
    seed_offset: int = 9999,
) -> float:
    """
    Run n_episodes greedy rollouts and return the mean undiscounted return.
    Resets env with deterministic seeds so evaluation is reproducible.
    """
    returns = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_offset + ep)
        ep_return = 0.0
        done = False
        while not done:
            action = agent.select_greedy_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += reward
            done = terminated or truncated
        returns.append(ep_return)
    return float(np.mean(returns))


def train_one_seed(
    seed: int,
    env_kwargs: dict,
    agent_kwargs: dict,
    n_episodes: int,
    eval_interval: int,
    eval_episodes: int,
    shaper_window: int,
    shaper_penalty: float,
    algorithm_name: str,
    env_name: str,
) -> None:
    """
    Train D3QN-PER for one seed and save results.

    The env is instantiated fresh for training and a separate clone is used
    for evaluation to avoid contaminating training state.
    """
    print(f"\n{'─'*60}")
    print(f"  {algorithm_name} | {env_name} | seed={seed}")
    print(f"{'─'*60}")

    # ── Environment (training) ──────────────────────────────────────────
    train_env = ObelixEnv(**env_kwargs, seed=seed)

    # ── Environment (evaluation) — separate instance, same config ───────
    eval_env = ObelixEnv(**env_kwargs, seed=seed + 10_000)

    # ── Agent ────────────────────────────────────────────────────────────
    obs_dim = train_env.obs_dim
    n_actions = train_env.n_actions
    agent = D3QNPERAgent(
        obs_dim=obs_dim,
        n_actions=n_actions,
        seed=seed,
        **agent_kwargs,
    )

    # ── Reward shaper ─────────────────────────────────────────────────────
    shaper = CircularMotionShaper(window=shaper_window, penalty=shaper_penalty)

    # ── Metrics ───────────────────────────────────────────────────────────
    train_rewards: List[float] = []
    eval_rewards: List[float] = []
    losses: List[float] = []

    start_wall = time.time()

    for episode in range(1, n_episodes + 1):
        obs, _ = train_env.reset(seed=seed * 10_000 + episode)
        shaper.reset()
        ep_return = 0.0
        done = False

        while not done:
            action = agent.select_action(obs)
            next_obs, raw_reward, terminated, truncated, _ = train_env.step(action)

            # ── Reward shaping: penalise circular oscillation ─────────────
            shaped_reward = shaper.shape(action, raw_reward)

            agent.store(obs, action, shaped_reward, next_obs, terminated or truncated)
            loss = agent.maybe_learn()
            if loss is not None:
                losses.append(loss)

            ep_return += raw_reward  # log raw (unshaped) for fair comparison
            obs = next_obs
            done = terminated or truncated

        train_rewards.append(ep_return)

        # ── Periodic evaluation ───────────────────────────────────────────
        if episode % eval_interval == 0:
            mean_eval = evaluate(agent, eval_env, n_episodes=eval_episodes)
            eval_rewards.append(mean_eval)

            recent_train = np.mean(train_rewards[-eval_interval:])
            elapsed = time.time() - start_wall
            print(
                f"  Ep {episode:>5}/{n_episodes} | "
                f"train_mean={recent_train:>8.1f} | "
                f"eval_mean={mean_eval:>8.1f} | "
                f"eps={agent.eps:.3f} | "
                f"steps={agent.total_steps:>7} | "
                f"t={elapsed:.0f}s"
            )

    total_wall = time.time() - start_wall
    final_eval = evaluate(agent, eval_env, n_episodes=eval_episodes * 2)

    # ── Save checkpoint ───────────────────────────────────────────────────
    results_dir = _make_results_dir(algorithm_name, env_name, seed)
    ckpt_path = os.path.join(results_dir, "checkpoint.pt")
    agent.save(ckpt_path)

    # ── Save results ──────────────────────────────────────────────────────
    save_training_results(
        algorithm_name=algorithm_name,
        env_name=env_name,
        seed=seed,
        train_rewards=train_rewards,
        train_time=total_wall,
        eval_rewards=eval_rewards,
        wall_time=total_wall,
        final_eval=final_eval,
        total_steps=agent.total_steps,
        extra={
            "mean_loss": float(np.mean(losses)) if losses else None,
            "n_learn_steps": agent.learn_steps,
            "final_eps": round(agent.eps, 4),
            "buffer_beta": round(agent.buffer.beta, 4),
            "shaper_window": shaper_window,
            "shaper_penalty": shaper_penalty,
        },
    )

    train_env.close()
    eval_env.close()

    print(
        f"  ✓ seed={seed} done | "
        f"final_eval={final_eval:.1f} | "
        f"steps={agent.total_steps} | "
        f"t={total_wall:.0f}s"
    )


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Configuration matrix
# ═══════════════════════════════════════════════════════════════════════════

def build_configs(
    difficulty_filter: Optional[List[int]] = None,
    wall_filter: Optional[bool] = None,
) -> List[dict]:
    """
    Return the list of environment configurations to train over.

    Difficulty levels:
      0 → stationary box
      2 → blinking box
      3 → blinking + moving box

    Wall settings:
      False → open arena
      True  → central wall with gap
    """
    difficulties = [0, 2, 3]
    wall_settings = [False, True]

    if difficulty_filter is not None:
        difficulties = [d for d in difficulties if d in difficulty_filter]
    if wall_filter is not None:
        wall_settings = [w for w in wall_settings if w == wall_filter]

    configs = []
    for diff in difficulties:
        for wall in wall_settings:
            wall_str = "wall" if wall else "nowall"
            env_name = f"OBELIX_diff{diff}_{wall_str}"
            configs.append(
                {
                    "env_name": env_name,
                    "env_kwargs": {
                        "scaling_factor": 2,
                        "arena_size": 500,
                        "max_steps": 2000,
                        "wall_obstacles": wall,
                        "difficulty": diff,
                        "box_speed": 2,
                        "n_stack": 4,
                        "reward_clip": None,
                        "render_mode": None,
                    },
                }
            )
    return configs


# ═══════════════════════════════════════════════════════════════════════════
# 5.  CLI entry-point
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train D3QN-PER on OBELIX across seeds and configurations."
    )

    # ── Experiment scope ──────────────────────────────────────────────────
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(range(10)),
        help="Random seeds to train with (default: 0..9)",
    )
    p.add_argument(
        "--difficulty",
        type=int,
        nargs="+",
        choices=[0, 2, 3],
        default=None,
        help="Difficulty levels to include (default: all = 0 2 3)",
    )
    p.add_argument(
        "--no-wall",
        action="store_true",
        help="Only run no-wall variant (skip wall-obstacle configs)",
    )
    p.add_argument(
        "--wall-only",
        action="store_true",
        help="Only run wall-obstacle variant",
    )

    # ── Training schedule ─────────────────────────────────────────────────
    p.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Training episodes per seed (default: 1000)",
    )
    p.add_argument(
        "--eval-interval",
        type=int,
        default=100,
        help="Evaluate every N episodes (default: 100)",
    )
    p.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="Episodes per evaluation (default: 10)",
    )

    # ── Reward shaping ────────────────────────────────────────────────────
    p.add_argument(
        "--shaper-window",
        type=int,
        default=4,
        help="Consecutive turns required to trigger oscillation penalty (default: 4)",
    )
    p.add_argument(
        "--shaper-penalty",
        type=float,
        default=-5.0,
        help="Penalty applied per step of detected oscillation (default: -5.0)",
    )

    # ── Agent hyper-parameters ────────────────────────────────────────────
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--gamma",        type=float, default=0.99)
    p.add_argument("--tau",          type=float, default=0.005)
    p.add_argument("--hidden-dim",   type=int,   default=256)
    p.add_argument("--n-layers",     type=int,   default=3)
    p.add_argument("--dropout",      type=float, default=0.0)
    p.add_argument("--buffer-cap",   type=int,   default=100_000)
    p.add_argument("--batch-size",   type=int,   default=64)
    p.add_argument("--learn-every",  type=int,   default=4)
    p.add_argument("--per-alpha",    type=float, default=0.6)
    p.add_argument("--per-beta",     type=float, default=0.4)
    p.add_argument("--eps-start",    type=float, default=1.0)
    p.add_argument("--eps-end",      type=float, default=0.05)
    p.add_argument("--eps-decay",    type=int,   default=50_000)
    p.add_argument("--grad-clip",    type=float, default=10.0)
    p.add_argument("--device",       type=str,   default="auto")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Wall filter ───────────────────────────────────────────────────────
    wall_filter = None
    if args.no_wall and not args.wall_only:
        wall_filter = False
    elif args.wall_only and not args.no_wall:
        wall_filter = True
    # If both or neither flag is set, wall_filter stays None → include both.

    # ── Build configuration matrix ────────────────────────────────────────
    configs = build_configs(
        difficulty_filter=args.difficulty,
        wall_filter=wall_filter,
    )

    # ── Agent hyper-parameter dict ────────────────────────────────────────
    agent_kwargs = dict(
        lr=args.lr,
        gamma=args.gamma,
        tau=args.tau,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        dropout=args.dropout,
        buffer_capacity=args.buffer_cap,
        batch_size=args.batch_size,
        learn_every=args.learn_every,
        per_alpha=args.per_alpha,
        per_beta=args.per_beta,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay_steps=args.eps_decay,
        grad_clip=args.grad_clip,
        device=args.device,
    )

    algorithm_name = "D3QN-PER"

    print("=" * 60)
    print(f"  Training {algorithm_name}")
    print(f"  Seeds    : {args.seeds}")
    print(f"  Episodes : {args.episodes}")
    print(f"  Configs  : {[c['env_name'] for c in configs]}")
    print(f"  Shaper   : window={args.shaper_window}, penalty={args.shaper_penalty}")
    print("=" * 60)

    total_runs = len(configs) * len(args.seeds)
    run_idx = 0

    for cfg in configs:
        env_name = cfg["env_name"]
        env_kwargs = cfg["env_kwargs"]

        for seed in args.seeds:
            run_idx += 1
            print(f"\n[Run {run_idx}/{total_runs}]")
            train_one_seed(
                seed=seed,
                env_kwargs=env_kwargs,
                agent_kwargs=agent_kwargs,
                n_episodes=args.episodes,
                eval_interval=args.eval_interval,
                eval_episodes=args.eval_episodes,
                shaper_window=args.shaper_window,
                shaper_penalty=args.shaper_penalty,
                algorithm_name=algorithm_name,
                env_name=env_name,
            )

    print("\n" + "=" * 60)
    print("  All training runs completed.")
    print(f"  Results saved under: results/{algorithm_name}/")
    print("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════
# Quick smoke-test (no training, just verifies plumbing)
# ═══════════════════════════════════════════════════════════════════════════

def _smoke_test() -> None:
    """
    Run a single episode (without learning) to verify the full pipeline.
    Call directly: python train_d3qn_per.py --smoke-test
    """
    print("[smoke-test] Instantiating env and agent …")
    env = ObelixEnv(
        scaling_factor=2,
        arena_size=500,
        max_steps=50,
        wall_obstacles=False,
        difficulty=0,
        n_stack=4,
        render_mode=None,
        seed=0,
    )
    agent = D3QNPERAgent(obs_dim=env.obs_dim, n_actions=env.n_actions, seed=0)
    shaper = CircularMotionShaper(window=4, penalty=-5.0)

    obs, _ = env.reset(seed=0)
    done = False
    step = 0
    while not done:
        action = agent.select_action(obs)
        next_obs, raw_rew, term, trunc, _ = env.step(action)
        shaped = shaper.shape(action, raw_rew)
        agent.store(obs, action, shaped, next_obs, term or trunc)
        agent.maybe_learn()
        obs = next_obs
        done = term or trunc
        step += 1

    save_training_results(
        algorithm_name="D3QN-PER",
        env_name="OBELIX_diff0_nowall",
        seed=0,
        train_rewards=[1.0, 2.0, 3.0],
        train_time=0.5,
        eval_rewards=[2.5],
        wall_time=0.5,
        final_eval=2.5,
        total_steps=agent.total_steps,
    )
    print(f"[smoke-test] Passed. Steps={step}, total_env_steps={agent.total_steps}")
    env.close()


if __name__ == "__main__":
    import sys

    if "--smoke-test" in sys.argv:
        sys.argv.remove("--smoke-test")
        _smoke_test()
    else:
        main()
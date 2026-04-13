"""
train_qtile.py
==============
Training script for Q-learning with tile coding on the OBELIX environment.

Features:
  - Multiple seeds and difficulty levels
  - Oscillation shaper (same as D3QN-PER, gated after warmup)
  - Dense progress reward hook (if box position available)
  - Evaluation every N episodes with greedy rollouts
  - Results saved to results/QTile/<env_name>/seed_<N>/
  - Checkpoint saved as .pkl (Q-table is a dict, not a neural net)

Usage
-----
    # Full run (all configs, 10 seeds)
    python train_qtile.py

    # Quick test — one config, 3 seeds, 200 episodes
    python train_qtile.py --difficulty 0 --no-wall --seeds 0 1 2 --episodes 200

    # Timed test — stop after N minutes
    python train_qtile.py --minutes 10 --difficulty 0 --no-wall --seeds 0

    # Compare with D3QN-PER on same config
    python train_qtile.py --difficulty 0 --no-wall --seeds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import time
from typing import List, Optional

import numpy as np

from q_tile_agent import QTileAgent
from wrapper_obelix import ObelixEnv


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Results saving (mirrors train_d3qn_per.py structure)
# ═══════════════════════════════════════════════════════════════════════════

def save_training_results(
    algorithm_name: str,
    env_name: str,
    seed: int,
    train_rewards: List[float],
    train_time: float,
    eval_rewards: List[float],
    final_eval: float,
    total_steps: int,
    extra: Optional[dict] = None,
) -> str:
    path = os.path.join("results", algorithm_name, env_name, f"seed_{seed}")
    os.makedirs(path, exist_ok=True)

    payload = {
        "algorithm":            algorithm_name,
        "env":                  env_name,
        "seed":                 seed,
        "total_steps":          total_steps,
        "train_time_s":         round(train_time, 3),
        "final_eval_mean_reward": round(float(final_eval), 4),
        "train_rewards":        [round(float(r), 4) for r in train_rewards],
        "eval_rewards":         [round(float(r), 4) for r in eval_rewards],
    }
    if extra:
        payload["extra"] = extra

    out_path = os.path.join(path, "training_results.json")
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"  [save] {out_path}")
    return out_path


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Oscillation shaper (same as D3QN-PER)
# ═══════════════════════════════════════════════════════════════════════════

class CircularMotionShaper:
    _TURN_SIGN = {0: +1, 1: +1, 2: 0, 3: -1, 4: -1}

    def __init__(self, window: int = 4, penalty: float = -5.0):
        self.window  = window
        self.penalty = penalty
        self._history: collections.deque = collections.deque(maxlen=window)

    def reset(self):
        self._history.clear()

    def shape(self, action: int, base_reward: float) -> float:
        sign = self._TURN_SIGN[action]
        if sign == 0:
            self._history.clear()
            return base_reward
        self._history.append(sign)
        if len(self._history) < self.window:
            return base_reward
        signs = list(self._history)
        alternating = all(signs[i] != signs[i + 1] for i in range(len(signs) - 1))
        return base_reward + self.penalty if alternating else base_reward


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Dense progress reward
# ═══════════════════════════════════════════════════════════════════════════

_dist_warned = False

def get_box_dist(env: ObelixEnv) -> Optional[float]:
    inner = env._env
    for ax, ay, bx, by in [
        ("agent_x", "agent_y", "box_x",   "box_y"),
        ("robot_x", "robot_y", "box_x",   "box_y"),
        ("agent_pos", "agent_pos", "box_pos", "box_pos"),
    ]:
        try:
            if ax == "agent_pos":
                ap = getattr(inner, "agent_pos")
                bp = getattr(inner, "box_pos")
                return float(np.sqrt((ap[0]-bp[0])**2 + (ap[1]-bp[1])**2))
            else:
                ax_ = getattr(inner, ax)
                ay_ = getattr(inner, ay)
                bx_ = getattr(inner, bx)
                by_ = getattr(inner, by)
                return float(np.sqrt((ax_-bx_)**2 + (ay_-by_)**2))
        except AttributeError:
            continue
    return None


def progress_bonus(env, prev_dist, scale=0.5):
    global _dist_warned
    curr = get_box_dist(env)
    if curr is None:
        if not _dist_warned:
            print("[WARN] Box position not accessible — dense reward disabled.")
            _dist_warned = True
        return 0.0, None
    if prev_dist is None:
        return 0.0, curr
    return scale * (prev_dist - curr), curr


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate(agent: QTileAgent, env: ObelixEnv, n_episodes: int = 10) -> float:
    returns = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=9000 + ep)
        ep_ret = 0.0
        done   = False
        while not done:
            action = agent.select_greedy_action(obs)
            obs, r, term, trunc, _ = env.step(action)
            ep_ret += r
            done = term or trunc
        returns.append(ep_ret)
    return float(np.mean(returns))


def random_baseline(env: ObelixEnv, n_episodes: int = 10) -> float:
    returns = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=8000 + ep)
        ep_ret = 0.0
        done   = False
        while not done:
            action = env.action_space.sample()
            obs, r, term, trunc, _ = env.step(action)
            ep_ret += r
            done = term or trunc
        returns.append(ep_ret)
    return float(np.mean(returns))


# ═══════════════════════════════════════════════════════════════════════════
# 5.  Single-seed training loop
# ═══════════════════════════════════════════════════════════════════════════

def train_one_seed(
    seed: int,
    env_kwargs: dict,
    agent_kwargs: dict,
    n_episodes: int,
    eval_interval: int,
    eval_episodes: int,
    shaper_window: int,
    shaper_penalty: float,
    shaper_gate: int,
    algorithm_name: str,
    env_name: str,
    deadline: Optional[float] = None,   # wall-clock deadline (time.time())
) -> None:

    print(f"\n{'─'*58}")
    print(f"  {algorithm_name} | {env_name} | seed={seed}")
    print(f"{'─'*58}")

    train_env = ObelixEnv(**env_kwargs, seed=seed)
    eval_env  = ObelixEnv(**{**env_kwargs, "seed": seed + 10_000})

    # Compute eps decay steps from expected total steps
    expected_steps   = env_kwargs["max_steps"] * n_episodes
    eps_decay_steps  = max(10_000, int(expected_steps * 0.4))

    agent = QTileAgent(
        obs_dim=train_env.obs_dim,
        n_actions=train_env.n_actions,
        seed=seed,
        eps_decay_steps=eps_decay_steps,
        **agent_kwargs,
    )

    shaper   = CircularMotionShaper(window=shaper_window, penalty=shaper_penalty)
    baseline = random_baseline(eval_env, n_episodes=5)
    print(f"  Random baseline: {baseline:.1f}")

    train_rewards: List[float] = []
    eval_rewards:  List[float] = []
    td_errors:     List[float] = []
    start_wall = time.time()

    for episode in range(1, n_episodes + 1):

        # Respect time deadline if set
        if deadline and time.time() > deadline:
            print(f"  [time limit] stopping at episode {episode-1}")
            break

        obs, _   = train_env.reset(seed=seed * 10_000 + episode)
        shaper.reset()
        prev_dist = get_box_dist(train_env)
        ep_return = 0.0
        done      = False

        while not done:
            action = agent.select_action(obs)
            next_obs, raw_reward, terminated, truncated, _ = train_env.step(action)

            # Dense progress reward
            bonus, prev_dist = progress_bonus(train_env, prev_dist)
            raw_reward += bonus

            # Oscillation shaper (gated)
            if agent.total_steps > shaper_gate:
                shaped_reward = shaper.shape(action, raw_reward)
            else:
                shaped_reward = raw_reward

            agent.store(obs, action, shaped_reward, next_obs, terminated or truncated)
            td = agent.maybe_learn()
            if td is not None:
                td_errors.append(td)

            ep_return += raw_reward
            obs  = next_obs
            done = terminated or truncated

        train_rewards.append(ep_return)

        # Progress print
        if episode % eval_interval == 0:
            mean_eval  = evaluate(agent, eval_env, n_episodes=eval_episodes)
            eval_rewards.append(mean_eval)
            elapsed    = time.time() - start_wall
            mean_train = np.mean(train_rewards[-eval_interval:])
            mean_td    = np.mean(td_errors[-1000:]) if td_errors else 0.0
            print(
                f"  Ep {episode:>5}/{n_episodes} | "
                f"train={mean_train:>8.1f} | "
                f"eval={mean_eval:>8.1f} | "
                f"eps={agent.eps:.3f} | "
                f"states={agent.n_states_visited:>6} | "
                f"td={mean_td:.3f} | "
                f"t={elapsed:.0f}s"
            )

    total_wall = time.time() - start_wall
    final_eval = evaluate(agent, eval_env, n_episodes=eval_episodes * 2)

    # Save checkpoint
    results_dir = os.path.join("results", algorithm_name, env_name, f"seed_{seed}")
    os.makedirs(results_dir, exist_ok=True)
    ckpt_path = os.path.join(results_dir, "checkpoint.pkl")
    agent.save(ckpt_path)

    save_training_results(
        algorithm_name=algorithm_name,
        env_name=env_name,
        seed=seed,
        train_rewards=train_rewards,
        train_time=total_wall,
        eval_rewards=eval_rewards,
        final_eval=final_eval,
        total_steps=agent.total_steps,
        extra={
            "n_states_visited": agent.n_states_visited,
            "mean_td_error":    float(np.mean(td_errors)) if td_errors else None,
            "final_eps":        round(agent.eps, 4),
            "baseline":         round(baseline, 2),
            "shaper_gate":      shaper_gate,
        },
    )

    train_env.close()
    eval_env.close()

    print(
        f"  ✓ seed={seed} | final_eval={final_eval:.1f} | "
        f"states={agent.n_states_visited} | steps={agent.total_steps} | "
        f"t={total_wall:.0f}s"
    )


# ═══════════════════════════════════════════════════════════════════════════
# 6.  Configuration matrix
# ═══════════════════════════════════════════════════════════════════════════

def build_configs(
    difficulty_filter: Optional[List[int]] = None,
    wall_filter: Optional[bool] = None,
) -> List[dict]:
    difficulties  = [0, 2, 3]
    wall_settings = [False, True]
    if difficulty_filter:
        difficulties  = [d for d in difficulties if d in difficulty_filter]
    if wall_filter is not None:
        wall_settings = [w for w in wall_settings if w == wall_filter]

    configs = []
    for diff in difficulties:
        for wall in wall_settings:
            wall_str = "wall" if wall else "nowall"
            configs.append({
                "env_name": f"OBELIX_diff{diff}_{wall_str}",
                "env_kwargs": {
                    "scaling_factor": 2,
                    "arena_size":     500,
                    "max_steps":      2000,
                    "wall_obstacles": wall,
                    "difficulty":     diff,
                    "box_speed":      2,
                    "n_stack":        1,          # tile coding uses raw 18-dim obs
                    "reward_clip":    (-10, 10),
                    "render_mode":    None,
                },
            })
    return configs


# ═══════════════════════════════════════════════════════════════════════════
# 7.  CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Q-learning + tile coding on OBELIX.")

    p.add_argument("--seeds",      type=int, nargs="+", default=list(range(5)))
    p.add_argument("--difficulty", type=int, nargs="+", choices=[0,2,3], default=None)
    p.add_argument("--no-wall",    action="store_true")
    p.add_argument("--wall-only",  action="store_true")
    p.add_argument("--episodes",   type=int,   default=100)
    p.add_argument("--minutes",    type=float, default=None,
                   help="Wall-clock time limit per seed in minutes (optional)")
    p.add_argument("--eval-interval", type=int,   default=100)
    p.add_argument("--eval-episodes", type=int,   default=10)

    # Agent hyperparameters
    p.add_argument("--lr",          type=float, default=0.2)
    p.add_argument("--gamma",       type=float, default=0.99)
    p.add_argument("--eps-start",   type=float, default=1.0)
    p.add_argument("--eps-end",     type=float, default=0.05)
    p.add_argument("--initial-q",   type=float, default=0.0,
                   help="Optimistic initial Q-value (try 1.0 to boost exploration)")

    # Shaper
    p.add_argument("--shaper-window",  type=int,   default=4)
    p.add_argument("--shaper-penalty", type=float, default=-5.0)
    p.add_argument("--shaper-gate",    type=int,   default=20_000,
                   help="Steps before oscillation penalty activates")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    wall_filter = None
    if args.no_wall and not args.wall_only:
        wall_filter = False
    elif args.wall_only and not args.no_wall:
        wall_filter = True

    configs = build_configs(
        difficulty_filter=args.difficulty,
        wall_filter=wall_filter,
    )

    agent_kwargs = dict(
        lr        = args.lr,
        gamma     = args.gamma,
        eps_start = args.eps_start,
        eps_end   = args.eps_end,
        initial_q = args.initial_q,
    )

    algorithm_name = "QTile"

    print("=" * 58)
    print(f"  Training {algorithm_name}")
    print(f"  Seeds    : {args.seeds}")
    print(f"  Episodes : {args.episodes}")
    print(f"  Configs  : {[c['env_name'] for c in configs]}")
    print("=" * 58)

    # total_runs = len(configs) * len(args.seeds)
    total_runs = 20
    run_idx    = 0

    for cfg in configs:
        for seed in args.seeds:
            run_idx += 1
            print(f"\n[Run {run_idx}/{total_runs}]")

            deadline = None
            if args.minutes:
                deadline = time.time() + args.minutes * 60

            train_one_seed(
                seed=seed,
                env_kwargs=cfg["env_kwargs"],
                agent_kwargs=agent_kwargs,
                n_episodes=args.episodes,
                eval_interval=args.eval_interval,
                eval_episodes=args.eval_episodes,
                shaper_window=args.shaper_window,
                shaper_penalty=args.shaper_penalty,
                shaper_gate=args.shaper_gate,
                algorithm_name=algorithm_name,
                env_name=cfg["env_name"],
                deadline=deadline,
            )

    print("\n" + "=" * 58)
    print(f"  All runs complete. Results in: results/{algorithm_name}/")
    print("=" * 58)


if __name__ == "__main__":
    main()

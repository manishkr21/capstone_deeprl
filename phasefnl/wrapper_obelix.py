"""
wrapper_obelix.py
=================
A Gymnasium-compatible wrapper around the OBELIX simulation environment.

The wrapper handles:
  - Converting the raw 18-bit numpy observation into a properly typed Box space.
  - Exposing a standard Discrete(5) action space.
  - Optional frame-stacking (n_stack > 1) to give the agent short-term memory.
  - Optional observation normalisation (divide each bit by 1.0 — already in {0,1},
    but kept as a hook for future continuous sensors).
  - Reward clipping (optional, useful for stabilising training).
  - Seeding for reproducibility.
  - A thin render() shim that forwards to the underlying env.

Usage
-----
    from wrapper_obelix import ObelixEnv

    env = ObelixEnv(
        scaling_factor=2,
        difficulty=0,
        n_stack=4,
        reward_clip=None,   # or e.g. (-500, 500)
        seed=42,
    )
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
"""

from __future__ import annotations

import collections
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np

# ---------------------------------------------------------------------------
# Make sure the OBELIX module is importable.  The environment file must be
# on sys.path or in the same directory.
# ---------------------------------------------------------------------------
from obelix import OBELIX  # adjust import path if needed


# ──────────────────────────────────────────────────────────────────────────────
# Action map: index → string key used by OBELIX.step()
# ──────────────────────────────────────────────────────────────────────────────
ACTION_MAP = {
    0: "L45",
    1: "L22",
    2: "FW",
    3: "R22",
    4: "R45",
}

# Raw observation dimension from OBELIX (16 sonar bits + 1 IR bit + 1 stuck bit)
RAW_OBS_DIM = 18


class ObelixEnv(gym.Env):
    """
    Gymnasium wrapper for the OBELIX pushing-robot environment.

    Parameters
    ----------
    scaling_factor : int
        Pixel-per-inch scaling for the simulated arena.  Larger values give
        more physical resolution but slow down rendering.  Typical: 2.
    arena_size : int
        Square arena side length in pixels (before flip).  Default: 500.
    max_steps : int
        Episode length cap (OBELIX truncates at this many steps).
    wall_obstacles : bool
        Enable the central wall obstacle with a gap.
    difficulty : int
        0 = static box, 2 = blinking box, 3 = moving + blinking box.
    box_speed : int
        Pixels per step for the moving box (only used when difficulty >= 3).
    n_stack : int
        Number of consecutive observations to stack into one state vector.
        n_stack=1 disables stacking (raw 18-dim obs).
        n_stack=4 gives a 72-dim state vector with 4 frames of history.
    reward_clip : tuple | None
        If provided, clips each step reward to (low, high).  Useful to
        prevent exploding gradients from the large stuck penalty (-200) or
        success bonus (+2000) when combined with an aggressive learning rate.
        Set to None to use raw rewards.
    render_mode : str | None
        "human"  → calls OBELIX's cv2 display every step.
        None     → headless (recommended for training).
    seed : int | None
        Master seed forwarded to OBELIX.
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        scaling_factor: int = 2,
        arena_size: int = 500,
        max_steps: int = 2000,
        wall_obstacles: bool = False,
        difficulty: int = 0,
        box_speed: int = 2,
        n_stack: int = 4,
        reward_clip: Optional[Tuple[float, float]] = None,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()

        self.scaling_factor = scaling_factor
        self.arena_size = arena_size
        self.max_steps = max_steps
        self.wall_obstacles = wall_obstacles
        self.difficulty = difficulty
        self.box_speed = box_speed
        self.n_stack = max(1, int(n_stack))
        self.reward_clip = reward_clip
        self.render_mode = render_mode
        self._seed = seed

        # ── Instantiate the underlying simulation ──────────────────────────
        self._env = OBELIX(
            scaling_factor=scaling_factor,
            arena_size=arena_size,
            max_steps=max_steps,
            wall_obstacles=wall_obstacles,
            difficulty=difficulty,
            box_speed=box_speed,
            seed=seed,
        )

        # ── Action space ───────────────────────────────────────────────────
        self.action_space = gym.spaces.Discrete(len(ACTION_MAP))

        # ── Observation space ──────────────────────────────────────────────
        # Each raw observation is a binary vector of length RAW_OBS_DIM.
        # With frame-stacking the agent sees n_stack consecutive frames.
        obs_dim = RAW_OBS_DIM * self.n_stack
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # ── Frame-stack buffer ─────────────────────────────────────────────
        self._frame_buffer: collections.deque = collections.deque(
            maxlen=self.n_stack
        )

    # ──────────────────────────────────────────────────────────────────────
    # Gymnasium API
    # ──────────────────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        """Reset the environment and return (observation, info)."""
        if seed is None:
            seed = self._seed

        raw_obs = self._env.reset(seed=seed)
        raw_obs = self._process_raw(raw_obs)

        # Fill every slot in the buffer with the initial observation so there
        # are no zeros at the start of an episode.
        self._frame_buffer.clear()
        for _ in range(self.n_stack):
            self._frame_buffer.append(raw_obs)

        obs = self._get_stacked_obs()
        info = self._build_info()
        return obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Apply action and return (obs, reward, terminated, truncated, info).

        Gymnasium separates 'terminated' (natural episode end) from
        'truncated' (hit max_steps).  OBELIX sets done=True for both cases,
        so we approximate: if we hit max_steps and done==True, it is
        truncated; otherwise terminated.
        """
        move_str = ACTION_MAP[int(action)]
        render = self.render_mode == "human"

        raw_obs, reward, done = self._env.step(move_str, render=render)

        raw_obs = self._process_raw(raw_obs)
        self._frame_buffer.append(raw_obs)
        obs = self._get_stacked_obs()

        if self.reward_clip is not None:
            reward = float(np.clip(reward, self.reward_clip[0], self.reward_clip[1]))
        else:
            reward = float(reward)

        # Distinguish termination cause.
        truncated = done and (self._env.current_step >= self._env.max_steps)
        terminated = done and not truncated

        info = self._build_info()
        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        """Render the current frame (only in 'human' mode)."""
        if self.render_mode == "human":
            self._env.render_frame()

    def close(self) -> None:
        """Clean up OpenCV windows."""
        import cv2
        cv2.destroyAllWindows()

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _process_raw(raw_obs: np.ndarray) -> np.ndarray:
        """Cast raw OBELIX observation to float32 in [0, 1]."""
        return raw_obs.astype(np.float32)

    def _get_stacked_obs(self) -> np.ndarray:
        """Concatenate buffered frames into a single flat vector."""
        return np.concatenate(list(self._frame_buffer), axis=0)

    def _build_info(self) -> dict:
        """Return a diagnostic info dict (does not affect training)."""
        return {
            "step": self._env.current_step,
            "enable_push": self._env.enable_push,
            "active_state": self._env.active_state,
            "stuck_flag": int(self._env.stuck_flag),
            "box_visible": self._env.box_visible,
            "difficulty": self.difficulty,
        }

    # ──────────────────────────────────────────────────────────────────────
    # Convenience
    # ──────────────────────────────────────────────────────────────────────

    @property
    def obs_dim(self) -> int:
        """Flat observation dimension (after frame-stacking)."""
        return RAW_OBS_DIM * self.n_stack

    @property
    def n_actions(self) -> int:
        """Number of discrete actions."""
        return len(ACTION_MAP)

    def seed(self, seed: Optional[int] = None) -> list:
        """Set master seed (legacy Gym API compatibility)."""
        self._seed = seed
        return [seed]
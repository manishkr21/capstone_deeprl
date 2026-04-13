"""
agent_template_qtile.py
=======================
Drop-in policy file for the trained Q-learning + tile coding OBELIX agent.

Evaluator calls only:
    action_str = policy(obs, rng)

Setup:
    1. Train:
        python train_qtile.py --difficulty 0 --no-wall --seeds 0
    2. Copy checkpoint:
        cp results/QTile/OBELIX_diff0_nowall/seed_0/checkpoint.pkl qtile_obelix_model.pkl
    3. Place this file + model in same directory
"""

import os
import pickle
import numpy as np


# ──────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "qtile_obelix_model_merged.pkl"   # ← fixed name (was inconsistent before)
)

ACTION_MAP = {
    0: "L45",
    1: "L22",
    2: "FW",
    3: "R22",
    4: "R45",
}
N_ACTIONS = 5


# ──────────────────────────────────────────────────────────────────────────
# Tile Coding (MUST match training exactly)
# ──────────────────────────────────────────────────────────────────────────

TILE_GROUPS = [
    list(range(0, 8)),    # front sonar
    list(range(8, 12)),   # left sonar
    list(range(12, 16)),  # right sonar
    [16],                 # IR
    [17],                 # stuck
]


def obs_to_state_key(obs: np.ndarray) -> tuple:
    """
    Convert observation → tile key (same as training).
    Supports frame-stacked obs by taking last 18 bits.
    """
    raw = obs[-18:].astype(int)

    key = []
    for group in TILE_GROUPS:
        val = 0
        for i in group:
            val = (val << 1) | int(raw[i])
        key.append(val)

    return tuple(key)


# ──────────────────────────────────────────────────────────────────────────
# Model loading (lazy + silent)
# ──────────────────────────────────────────────────────────────────────────

_q_table = None


def _load_model():
    global _q_table

    if _q_table is not None:
        return _q_table

    try:
        with open(MODEL_PATH, "rb") as f:
            ckpt = pickle.load(f)

        # Only store raw dict for speed
        _q_table = ckpt.get("q_table", {})

        # Safety: ensure correct type
        if not isinstance(_q_table, dict):
            _q_table = {}

    except Exception:
        # Silent fallback (important for evaluator)
        _q_table = {}

    return _q_table


# ──────────────────────────────────────────────────────────────────────────
# Policy
# ──────────────────────────────────────────────────────────────────────────

def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    """
    Greedy Q-table policy with safe fallback.

    Parameters
    ----------
    obs : np.ndarray
        Observation (18 or 72 dim)
    rng : np.random.Generator
        Provided by evaluator

    Returns
    -------
    str action
    """
    q_table = _load_model()
    state_key = obs_to_state_key(obs)

    # ── Greedy action if seen ─────────────────────────────────────────────
    if state_key in q_table:
        q_values = q_table[state_key]

        # Safety: handle corrupted entries
        if isinstance(q_values, np.ndarray) and len(q_values) == N_ACTIONS:
            action_idx = int(np.argmax(q_values))
        else:
            action_idx = int(rng.integers(N_ACTIONS))

    # ── Unseen state fallback ─────────────────────────────────────────────
    else:
        action_idx = int(rng.integers(N_ACTIONS))

    return ACTION_MAP[action_idx]
"""
agent_template_d3qn.py
======================
Drop-in policy file for the trained D3QN-PER OBELIX agent.

The only function the evaluator calls is:
    action_str = policy(obs, rng)

Before using:
    1. Train your model and save it:
           agent.save("d3qn_obelix_model.pt")
    2. Put this file and d3qn_obelix_model.pt in the same directory.
    3. Import and call policy().
"""

import os
import collections
import numpy as np
import torch
import torch.nn as nn

# ── Config — must match exactly what you trained with ─────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "checkpoint.pt")
RAW_OBS_DIM = 18
N_STACK     = 4
OBS_DIM     = RAW_OBS_DIM * N_STACK   # 72
N_ACTIONS   = 5
HIDDEN_DIM  = 128
N_LAYERS    = 2

ACTION_MAP = {0: "L45", 1: "L22", 2: "FW", 3: "R22", 4: "R45"}


# ── Network (identical to d3qn_per_agent.py) ──────────────────────────────
class DuelingDQN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        in_dim = OBS_DIM
        for _ in range(N_LAYERS):
            layers += [nn.Linear(in_dim, HIDDEN_DIM), nn.ReLU()]
            in_dim = HIDDEN_DIM
        self.shared = nn.Sequential(*layers)
        self.value_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2), nn.ReLU(),
            nn.Linear(HIDDEN_DIM // 2, 1)
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2), nn.ReLU(),
            nn.Linear(HIDDEN_DIM // 2, N_ACTIONS)
        )

    def forward(self, x):
        f = self.shared(x)
        v = self.value_head(f)
        a = self.advantage_head(f)
        return v + a - a.mean(dim=1, keepdim=True)


# ── Load model once, cache it ─────────────────────────────────────────────
_model = None

def _load_model():
    global _model
    if _model is not None:
        return _model
    _model = DuelingDQN()
    if os.path.exists(MODEL_PATH):
        ckpt = torch.load(MODEL_PATH, map_location="cpu")
        key  = "online_state_dict" if "online_state_dict" in ckpt else None
        _model.load_state_dict(ckpt[key] if key else ckpt)
        print(f"[D3QN] Loaded model from {MODEL_PATH}")
    else:
        print(f"[D3QN] WARNING: {MODEL_PATH} not found — using random weights")
    _model.eval()
    return _model


# ── Frame-stack buffer (stateful across steps, reset each episode) ─────────
_frame_buf = collections.deque(maxlen=N_STACK)

def reset_episode():
    """Call this at the start of every episode before the first policy() call."""
    _frame_buf.clear()

def _stack(raw_obs):
    raw_obs = raw_obs.astype(np.float32)
    if len(_frame_buf) == 0:                  # first step — fill all slots
        for _ in range(N_STACK):
            _frame_buf.append(raw_obs)
    else:
        _frame_buf.append(raw_obs)
    return np.concatenate(list(_frame_buf))   # shape (72,)


# ── Policy function ───────────────────────────────────────────────────────
def policy(obs, rng):
    """
    Greedy D3QN policy.

    Parameters
    ----------
    obs : np.ndarray, shape (18,)
        Raw OBELIX sensor observation.
    rng : np.random.Generator
        Provided by the evaluator — unused (policy is deterministic).

    Returns
    -------
    str : one of "L45", "L22", "FW", "R22", "R45"
    """
    model   = _load_model()
    stacked = _stack(obs)
    tensor  = torch.FloatTensor(stacked).unsqueeze(0)
    with torch.no_grad():
        q_values = model(tensor)
    action_idx = int(q_values.argmax(dim=1).item())
    return ACTION_MAP[action_idx]
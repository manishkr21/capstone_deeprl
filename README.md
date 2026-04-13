# CS780 Capstone: Deep Reinforcement Learning

Course project for **CS780 – Deep Reinforcement Learning**, offered at **IIT Kanpur** by **Prof. Ashutosh Modi**.

---

## Overview

This repository implements two foundational Deep RL algorithms and evaluates them on classic control tasks using [OpenAI Gymnasium](https://gymnasium.farama.org/).

| Algorithm | Type | Environment |
|-----------|------|-------------|
| DQN (Deep Q-Network) | Value-based | CartPole-v1, LunarLander-v2 |
| REINFORCE (Policy Gradient) | Policy-based | CartPole-v1, LunarLander-v2 |

---

## Project Structure

```
capstone_deeprl/
├── src/
│   ├── agents/
│   │   ├── dqn.py          # Deep Q-Network agent
│   │   └── reinforce.py    # REINFORCE (Monte-Carlo Policy Gradient) agent
│   ├── networks/
│   │   ├── q_network.py    # Q-value neural network
│   │   └── policy_network.py  # Policy neural network
│   └── utils/
│       ├── replay_buffer.py   # Experience replay buffer for DQN
│       └── plotting.py        # Training curve utilities
├── train_dqn.py            # Entry point – train DQN
├── train_reinforce.py      # Entry point – train REINFORCE
└── requirements.txt
```

---

## Setup

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Training

### DQN

```bash
# CartPole-v1 (default)
python train_dqn.py

# LunarLander-v2
python train_dqn.py --env LunarLander-v2 --episodes 1000

# All options
python train_dqn.py --help
```

### REINFORCE

```bash
# CartPole-v1 (default)
python train_reinforce.py

# LunarLander-v2
python train_reinforce.py --env LunarLander-v2 --episodes 2000

# All options
python train_reinforce.py --help
```

Both scripts save training reward curves as PNG files and print episode returns to the console.

---

## Algorithms

### DQN (Mnih et al., 2015)

Key components:
- **Experience Replay** – random mini-batch sampling from a circular replay buffer breaks temporal correlations.
- **Target Network** – a periodically synced copy of the online network stabilises the TD target.
- **ε-greedy Exploration** – epsilon decays from 1.0 → 0.01 over training.

### REINFORCE (Williams, 1992)

Key components:
- **Monte-Carlo Returns** – full-episode returns with a discounted baseline (mean-subtracted).
- **Policy Gradient** – directly optimises the log-probability of actions weighted by the discounted return.

---

## References

1. Mnih, V. et al. (2015). *Human-level control through deep reinforcement learning*. Nature.
2. Williams, R. J. (1992). *Simple statistical gradient-following algorithms for connectionist reinforcement learning*. Machine Learning.
3. Sutton, R. S. & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.).
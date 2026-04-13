import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ppo_agent import PPOAgent
from obelix_gym_wrapper import OBELIXGymWrapper


class PPOConfig:
    """PPO Training Configuration"""

    # ── Episode settings ──────────────────────────────────────────────
    num_episodes = 1000
    max_steps    = 1000

    # ── PPO hyperparameters ───────────────────────────────────────────
    learning_rate = 0.0001
    gamma         = 0.99
    gae_lambda    = 0.95
    clip_ratio    = 0.25
    value_coef    = 0.75
    entropy_coef  = 0.05

    # ── Training settings ─────────────────────────────────────────────
    # Steps collected before each PPO update (replaces per-episode training)
    rollout_length = 2048
    num_epochs     = 12
    batch_size     = 512

    # ── Network ───────────────────────────────────────────────────────
    hidden_size = 256

    # # ── Reward handling ───────────────────────────────────────────────
    # normalize_rewards    = True
    # reward_normalize_min = 200    # samples needed before normalising
    # reward_history_len   = 2000   # window for running mean/std
    # reward_clip_norm     = 5.0   # clip after normalisation
    # reward_clip_raw      = 10  # clip raw reward before normalisation

    # ── Reward handling ───────────────────────────────────────────────
    normalize_rewards    = True
    reward_normalize_min = 500
    reward_history_len   = 1000

    reward_scale         = 0.1   # replaces /100
    reward_clip_raw      = 3.0
    reward_clip_norm     = 2.5

    step_penalty         = 0.0005   # smaller penalty
    terminal_bonus       = 0.3


    # ── Learning rate schedule ────────────────────────────────────────
    lr_schedule    = False  # use ExponentialLR scheduler
    lr_decay_rate  = 0.9995   # applied every episode

    # ── Logging / checkpointing ───────────────────────────────────────
    log_interval        = 10
    checkpoint_interval = 100


# def process_reward(reward, all_rewards, config):
#     """
#     1. Clip the raw reward to prevent extreme outlier values.
#     2. Normalise using a running window mean/std once enough samples exist.
#     3. Clip the normalised reward to [-reward_clip_norm, +reward_clip_norm].

#     Returns (original_clipped, normalised_reward) so the training loop can
#     track original values for logging while storing normalised values.
#     """
#     clipped = float(np.clip(reward, -config.reward_clip_raw, config.reward_clip_raw))

#     if config.normalize_rewards and len(all_rewards) >= config.reward_normalize_min:
#         window       = all_rewards[-config.reward_history_len:]
#         reward_mean  = float(np.mean(window))
#         reward_std   = max(float(np.std(window)), 1e-6)
#         normalised   = (clipped - reward_mean) / reward_std
#         normalised   = float(np.clip(normalised, -config.reward_clip_norm, config.reward_clip_norm))
#     else:
#         normalised = clipped

#     return clipped, normalised

def process_reward(reward, all_rewards, config, step=None, done=False):
    """
    Stable PPO reward processing:
    1. Scale reward
    2. Clip raw reward
    3. Optional step penalty
    4. Optional terminal shaping
    5. Normalize using running stats
    """

    # ── 1. SCALE ───────────────────────────────────────────────
    scaled = reward * config.reward_scale

    # ── 2. CLIP RAW ────────────────────────────────────────────
    clipped = float(np.clip(
        scaled,
        -config.reward_clip_raw,
        config.reward_clip_raw
    ))
    # clipped -= 1.0
    # ── 3. STEP PENALTY ────────────────────────────────────────
    if config.step_penalty > 0:
        clipped -= config.step_penalty

    # ── 4. TERMINAL SHAPING ────────────────────────────────────
    if done:
        if clipped > 0:
            clipped += config.terminal_bonus
        else:
            clipped -= config.terminal_bonus

    # ⚠️ IMPORTANT: store processed reward (not raw)
    all_rewards.append(clipped)

    # ── 5. NORMALIZATION ───────────────────────────────────────
    if config.normalize_rewards and len(all_rewards) >= config.reward_normalize_min:

        window = all_rewards[-config.reward_history_len:]

        mean = float(np.mean(window))
        std  = float(np.std(window))

        if std < 1e-6:
            std = 1.0

        normalized = (clipped - mean) / std

        normalized = float(np.clip(
            normalized,
            -config.reward_clip_norm,
            config.reward_clip_norm
        ))
    else:
        normalized = clipped

    return clipped, normalized


def get_next_value(agent, observation):
    """Critic estimate of a state without gradient tracking."""
    with torch.no_grad():
        obs_t = torch.FloatTensor(observation).unsqueeze(0).to(agent.device)
        _, value = agent.network(obs_t)
    return value.item()


def train_ppo():
    config = PPOConfig()

    env = OBELIXGymWrapper(
        scaling_factor=5,
        arena_size=500,
        max_steps=config.max_steps,
        wall_obstacles=False,
        difficulty=0,
        render_mode=None
    )

    agent = PPOAgent(
        action_space=list(range(5)),
        observation_space=18,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_ratio=config.clip_ratio,
        value_coef=config.value_coef,
        entropy_coef=config.entropy_coef,
        hidden_size=config.hidden_size,
        kl_threshold=0.05  # Optional: early stopping if KL divergence exceeds this (not implemented in this code but can be added in agent.train()
    )

    # LR scheduler — step once per episode so lr_decay_rate is per-episode
    scheduler = None
    if config.lr_schedule:
        from torch.optim.lr_scheduler import ExponentialLR
        scheduler = ExponentialLR(agent.optimizer, gamma=config.lr_decay_rate)

    # ── Metrics ───────────────────────────────────────────────────────
    episode_rewards  = deque(maxlen=100)
    episode_lengths  = deque(maxlen=100)
    episode_losses   = []
    all_rewards      = []   # running history for normalisation
    best_reward      = float('-inf')

    # ── Rollout accumulation state ────────────────────────────────────
    # Collect rollout_length steps across episode boundaries, then train.
    steps_since_update = 0
    observation, _ = env.reset()
    episode_reward  = 0
    episode_step    = 0

    start_time = time.time()

    print("=" * 70)
    print(f"PPO Training  —  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print(f"  Episodes:        {config.num_episodes}")
    print(f"  Rollout length:  {config.rollout_length}")
    print(f"  Learning rate:   {config.learning_rate}")
    print(f"  Clip ratio:      {config.clip_ratio}")
    print(f"  Gamma:           {config.gamma}")
    print("=" * 70 + "\n")

    for episode in range(config.num_episodes):
        # Each episode runs until done or max_steps;
        # PPO update fires whenever steps_since_update >= rollout_length.
        done = False
        episode_reward = 0
        episode_step   = 0

        while not done and episode_step < config.max_steps:
            action, action_idx, log_prob, value = agent.select_action(
                observation, training=True
            )

            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            clipped_reward, norm_reward = process_reward(
                reward, all_rewards, config, step=episode_step, done=done
            )
            agent.store_transition(
                observation, action, action_idx,
                norm_reward,   # agent trains on normalised reward
                value, log_prob, done
            )

            episode_reward  += clipped_reward   # log original (clipped) scale
            observation      = next_observation
            episode_step    += 1
            steps_since_update += 1

            # ── PPO update when rollout is full ────────────────────────
            if steps_since_update >= config.rollout_length:
                next_val = 0.0 if done else get_next_value(agent, observation)
                loss = agent.train(
                    next_val,
                    num_epochs=config.num_epochs,
                    batch_size=config.batch_size
                )
                episode_losses.append(loss)
                steps_since_update = 0

        # ── End of episode: train on any remaining steps ───────────────
        if len(agent.trajectory['observations']) > 0:
            next_val = 0.0 if done else get_next_value(agent, observation)
            loss = agent.train(
                next_val,
                num_epochs=config.num_epochs,
                batch_size=config.batch_size
            )
            episode_losses.append(loss)
            steps_since_update = 0

        # Reset environment for next episode
        observation, _ = env.reset()

        # ── Metrics ────────────────────────────────────────────────────
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_step)

        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save_model('ppo_obelix_model_best.pth')

        # ── LR schedule: step every episode ────────────────────────────
        if scheduler is not None:
            scheduler.step()

        # ── Logging ────────────────────────────────────────────────────
        if (episode + 1) % config.log_interval == 0:
            elapsed  = time.time() - start_time
            avg_rew  = float(np.mean(episode_rewards))
            avg_len  = float(np.mean(episode_lengths))
            avg_loss = float(np.mean(episode_losses[-config.log_interval:])) \
                       if episode_losses else 0.0

            print(
                f"Episode {episode + 1:4d}/{config.num_episodes} | "
                f"Reward: {episode_reward:9.2f} (avg: {avg_rew:9.2f}) | "
                f"Length: {episode_step:4d} (avg: {avg_len:6.1f}) | "
                f"Loss: {avg_loss:8.4f} | "
                f"Best: {best_reward:9.2f} | "
                f"Time: {elapsed:6.1f}s"
            )

        # ── Checkpoint ────────────────────────────────────────────────
        if (episode + 1) % config.checkpoint_interval == 0:
            agent.save_model(f'ppo_obelix_model_ep{episode + 1}.pth')

    # ── Final save & summary ──────────────────────────────────────────
    agent.save_model('ppo_obelix_model.pth')

    print("\n" + "=" * 70)
    print("Training complete")
    print("=" * 70)
    print(f"  Total episodes:      {config.num_episodes}")
    print(f"  Best reward:         {best_reward:.2f}")
    print(f"  Final avg (last 100):{float(np.mean(episode_rewards)):.2f}")
    print(f"  Final avg length:    {float(np.mean(episode_lengths)):.2f}")
    print(f"  Total steps:         {agent.total_steps}")
    print(f"  Total time:          {(time.time() - start_time) / 60:.2f} min")
    print("=" * 70 + "\n")

    plot_training_curves(
        list(episode_rewards),
        episode_losses,
        list(episode_lengths)
    )

    return agent, list(episode_rewards), episode_losses, list(episode_lengths)


def plot_training_curves(rewards, losses, lengths):
    """Plot and save training statistics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    window = 20

    def moving_avg(data, w):
        return np.convolve(data, np.ones(w) / w, mode='valid') if len(data) > w else []

    # Episode rewards
    ax = axes[0, 0]
    ax.plot(rewards, alpha=0.5, label='Episode reward')
    ma = moving_avg(rewards, window)
    if len(ma):
        ax.plot(range(window - 1, len(rewards)), ma, linewidth=2,
                label=f'Moving avg (n={window})')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total reward')
    ax.set_title('Episode rewards over training')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Moving average only (zoomed view)
    ax = axes[0, 1]
    ma = moving_avg(rewards, window)
    if len(ma):
        ax.plot(range(window - 1, len(rewards)), ma, linewidth=2, color='orange')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Moving avg reward')
    ax.set_title(f'Moving average reward (window={window})')
    ax.grid(True, alpha=0.3)

    # Episode lengths
    ax = axes[1, 0]
    ax.plot(lengths, alpha=0.5, label='Episode length')
    ma = moving_avg(lengths, window)
    if len(ma):
        ax.plot(range(window - 1, len(lengths)), ma, linewidth=2,
                label=f'Moving avg (n={window})')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps')
    ax.set_title('Episode lengths over training')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Training loss (log scale)
    ax = axes[1, 1]
    if losses:
        ax.semilogy(losses, alpha=0.5, label='Loss')
        ma = moving_avg(losses, window)
        if len(ma):
            ax.semilogy(range(window - 1, len(losses)), ma, linewidth=2,
                        label=f'Moving avg (n={window})')
    ax.set_xlabel('Update step')
    ax.set_ylabel('Loss (log scale)')
    ax.set_title('Training loss over time')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig('ppo_training_curves.png', dpi=150, bbox_inches='tight')
    print("✓ Training curves saved to ppo_training_curves.png")
    plt.show()


if __name__ == "__main__":
    agent, rewards, losses, lengths = train_ppo()
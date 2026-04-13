# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.distributions import Categorical
# from collections import deque
# import time


# class PPONetwork(nn.Module):
#     """Actor-Critic Network for PPO with Improved Architecture"""
#     def __init__(self, observation_space, action_space, hidden_size=256):
#         super(PPONetwork, self).__init__()
        
#         self.observation_space = observation_space
#         self.action_space = action_space
        
#         # # Shared feature extraction with layer normalization
#         # self.shared_layers = nn.Sequential(
#         #     nn.Linear(observation_space, hidden_size),
#         #     nn.ReLU(),
#         #     nn.Linear(hidden_size, hidden_size),
#         #     nn.ReLU()
#         # )
#         #Add LayerNorm to the shared network. Replacing bare ReLU stacks with Linear → LayerNorm → Tanh (as used in OpenAI's PPO baselines) significantly stabilises training, especially with larger batches

#         self.shared_layers = nn.Sequential(
#             nn.Linear(observation_space, hidden_size),
#             nn.LayerNorm(hidden_size),
#             nn.Tanh(),
#             nn.Linear(hidden_size, hidden_size),
#             nn.LayerNorm(hidden_size),
#             nn.Tanh()
#         )
        
#         # Actor (policy) head - outputs logits
#         self.actor_head = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, action_space)
#         )
        
#         # Critic (value) head - normalized value output
#         self.critic_head = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, 1)
#         )
        
#         # Initialize weights with smaller variance for stability
#         self._initialize_weights()
    
#     def _initialize_weights(self):
#         """Initialize weights for training stability"""
#         for module in self.modules():
#             if isinstance(module, nn.Linear):
#                 nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
#                 nn.init.constant_(module.bias, 0)
    
#     def forward(self, x):
#         """Forward pass returning both policy and value"""
#         features = self.shared_layers(x)
        
#         # Policy output (logits)
#         action_logits = self.actor_head(features)
        
#         # Value output (single scalar per sample)
#         state_value = self.critic_head(features)
        
#         return action_logits, state_value


# class PPOAgent:
#     """Proximal Policy Optimization Agent"""
    
#     def __init__(self,
#                  action_space,
#                  observation_space,
#                  learning_rate=0.0003,
#                  gamma=0.99,
#                  gae_lambda=0.95,
#                  clip_ratio=0.2,
#                  value_coef=0.5,
#                  entropy_coef=0.01,
#                  max_grad_norm=0.5,
#                  hidden_size=256,
#                  device=None):
#         """
#         Initialize PPO Agent
        
#         Args:
#             action_space: List of action indices
#             observation_space: Size of observation vector
#             learning_rate: Learning rate for optimizer
#             gamma: Discount factor
#             gae_lambda: GAE lambda parameter for advantage estimation
#             clip_ratio: Clipping parameter for PPO objective
#             value_coef: Coefficient for value function loss
#             entropy_coef: Coefficient for entropy regularization
#             max_grad_norm: Maximum gradient norm for clipping
#             hidden_size: Hidden layer size
#             device: torch device (CPU/GPU)
#         """
#         self.action_space = action_space
#         self.observation_space = observation_space
#         self.learning_rate = learning_rate
#         self.gamma = gamma
#         self.gae_lambda = gae_lambda
#         self.clip_ratio = clip_ratio
#         self.value_coef = value_coef
#         self.entropy_coef = entropy_coef
#         self.max_grad_norm = max_grad_norm
        
#         # Device setup
#         self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
#         # Network
#         self.network = PPONetwork(
#             observation_space,
#             len(action_space),
#             hidden_size
#         ).to(self.device)
        
#         # Optimizer
#         # self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
#         self.optimizer = optim.Adam([
#             {'params': self.network.shared_layers.parameters(), 'lr': learning_rate},
#             {'params': self.network.actor_head.parameters(),    'lr': learning_rate},
#             {'params': self.network.critic_head.parameters(),   'lr': learning_rate * 0.5}
#         ])

#         # Trajectory buffer for collecting experiences
#         self.trajectory = {
#             'observations': [],
#             'actions': [],
#             'action_indices': [],
#             'rewards': [],
#             'values': [],
#             'log_probs': [],
#             'dones': []
#         }
        
#         # Training statistics
#         self.training_losses = []
#         self.total_steps = 0
    
#     def select_action(self, observation, training=True):
#         """
#         Select action using policy
        
#         Args:
#             observation: Current observation
#             training: If True, sample from distribution. If False, use greedy action
        
#         Returns:
#             action: Selected action
#             log_prob: Log probability of selected action
#             value: State value estimate
#         """
#         self.total_steps += 1

#         with torch.no_grad():
#             obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
#             action_logits, state_value = self.network(obs_tensor)
        
#         if training:
#             # Sample from policy distribution
#             action_dist = Categorical(logits=action_logits)
#             action_idx = action_dist.sample()
#             log_prob = action_dist.log_prob(action_idx)
#         else:
#             # Greedy action selection
#             action_idx = action_logits.argmax(dim=1)
#             action_dist = Categorical(logits=action_logits)
#             log_prob = action_dist.log_prob(action_idx)
        
#         return self.action_space[action_idx.item()], action_idx.item(), log_prob.item(), state_value.item()
    
#     def store_transition(self, observation, action, action_idx, reward, value, log_prob, done):
#         """Store transition in trajectory buffer"""
#         self.trajectory['action_indices'].append(action_idx)
#         self.trajectory['observations'].append(observation)
#         self.trajectory['actions'].append(action)
#         self.trajectory['rewards'].append(reward)
#         self.trajectory['values'].append(value)
#         self.trajectory['log_probs'].append(log_prob)
#         self.trajectory['dones'].append(done)
    
#     def compute_gae(self, next_value):
#         """
#         Compute Generalized Advantage Estimation (GAE)
        
#         Args:
#             next_value: Value estimate of next state (after trajectory ends)
        
#         Returns:
#             advantages: Computed advantages
#             returns: Computed returns (value targets)
#         """
#         rewards = np.array(self.trajectory['rewards'])
#         values = np.array(self.trajectory['values'] + [next_value])
#         dones = np.array(self.trajectory['dones'])
        
#         advantages = np.zeros_like(rewards)
#         gae = 0
        
#         # Compute GAE backwards through trajectory
#         for t in reversed(range(len(rewards))):
#             if t == len(rewards) - 1:
#                 next_non_terminal = 1.0 - dones[t]
#                 next_value_t = next_value
#             else:
#                 next_non_terminal = 1.0 - dones[t]
#                 next_value_t = values[t + 1]
            
#             delta = rewards[t] + self.gamma * next_value_t * next_non_terminal - values[t]
#             gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
#             advantages[t] = gae
        
#         returns = advantages + values[:-1]
        
#         return advantages, returns
    
#     def train(self, next_value, num_epochs=10, batch_size=64):
#         """
#         Train network on collected trajectory
        
#         Args:
#             next_value: Value estimate of next state
#             num_epochs: Number of training epochs per trajectory
#             batch_size: Batch size for training
        
#         Returns:
#             mean_loss: Average training loss
#         """
#         # Compute advantages and returns
#         advantages, returns = self.compute_gae(next_value)
        
#         # Normalize advantages
#         advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
#         # Convert to tensors
#         observations = torch.FloatTensor(
#             np.array(self.trajectory['observations'])
#         ).to(self.device)
#         # actions = torch.LongTensor(
#         #     [self.action_space.index(a) for a in self.trajectory['actions']]
#         # ).to(self.device)
#         actions = torch.LongTensor(self.trajectory['action_indices']).to(self.device)

#         old_log_probs = torch.FloatTensor(
#             self.trajectory['log_probs']
#         ).to(self.device)
#         advantages = torch.FloatTensor(advantages).to(self.device)
#         returns = torch.FloatTensor(returns).to(self.device)
        
#         # Training loop
#         epoch_losses = []
#         num_samples = len(self.trajectory['observations'])
#         indices = np.arange(num_samples)
        
#         for epoch in range(num_epochs):
#             np.random.shuffle(indices)
            
#             for i in range(0, num_samples, batch_size):
#                 batch_indices = indices[i:i + batch_size]
                
#                 # Forward pass
#                 action_logits, state_values = self.network(observations[batch_indices])
                
#                 # Compute new log probabilities
#                 dist = Categorical(logits=action_logits)
#                 new_log_probs = dist.log_prob(actions[batch_indices])
#                 entropy = dist.entropy().mean()
                
#                 # PPO surrogate loss
#                 ratio = torch.exp(new_log_probs - old_log_probs[batch_indices])
#                 surr1 = ratio * advantages[batch_indices]
#                 surr2 = torch.clamp(
#                     ratio,
#                     1.0 - self.clip_ratio,
#                     1.0 + self.clip_ratio
#                 ) * advantages[batch_indices]
#                 actor_loss = -torch.min(surr1, surr2).mean()

#                 old_values = torch.FloatTensor(self.trajectory['values']).to(self.device)

#                 # Value function loss with clipping
                
#                 value_pred_clipped = old_values[batch_indices] + torch.clamp(
#                     state_values.squeeze() - old_values[batch_indices],
#                     -self.clip_ratio,
#                     self.clip_ratio
#                 )

#                 critic_loss = torch.max(
#                     nn.functional.mse_loss(state_values.squeeze(-1), returns[batch_indices]),
#                     nn.functional.mse_loss(value_pred_clipped, returns[batch_indices])
#                 )

#                 # if critic loss is below 1
#                 if critic_loss < 1.0:
#                     print(f"Warning: suspiciously low critic loss {critic_loss.item():.4f} — possible collapse")                
               
#                 # Total loss
#                 loss = (
#                     actor_loss +
#                     self.value_coef * critic_loss -
#                     self.entropy_coef * entropy
#                 )
                
#                 # Backward pass
#                 self.optimizer.zero_grad()
#                 loss.backward()
#                 nn.utils.clip_grad_norm_(
#                     self.network.parameters(),
#                     self.max_grad_norm
#                 )
#                 self.optimizer.step()

#                 # compute kl divergence for monitoring
#                 with torch.no_grad():
#                     kl_divergence = (old_log_probs[batch_indices] - new_log_probs).mean().item()
#                     # if kl_divergence > 1.5 * self.clip_ratio:
#                     #     print(f"Early stopping at epoch {epoch} due to KL divergence: {kl_divergence:.4f}")
#                     #     break
#                     if kl_divergence > 0.015:  # 1.5% KL divergence threshold
#                         print(f"Early stopping at epoch {epoch} due to high KL divergence: {kl_divergence:.4f}")
#                         break
                
#                 epoch_losses.append(loss.item())
        
#         # Clear trajectory
#         self.trajectory = {
#             'observations': [],
#             'actions': [],
#             'action_indices': [],
#             'rewards': [],
#             'values': [],
#             'log_probs': [],
#             'dones': []
#         }
        
#         mean_loss = np.mean(epoch_losses)
#         self.training_losses.append(mean_loss)
        
#         return mean_loss
    
#     def save_model(self, filepath):
#         """Save model weights"""
#         torch.save(
#             self.network.state_dict(),
#             filepath
#         )
#         print(f"✓ Model saved to {filepath}")
    
#     def load_model(self, filepath):
#         """Load model weights"""
#         self.network.load_state_dict(
#             torch.load(filepath, map_location=self.device)
#         )
#         print(f"✓ Model loaded from {filepath}")
    
#     def get_stats(self):
#         """Get training statistics"""
#         return {
#             'total_steps': self.total_steps,
#             'avg_loss': np.mean(self.training_losses[-100:]) if self.training_losses else 0
#         }

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque
import time


class PPONetwork(nn.Module):
    """Actor-Critic Network for PPO with Improved Architecture"""
    def __init__(self, observation_space, action_space, hidden_size=256):
        super(PPONetwork, self).__init__()

        self.observation_space = observation_space
        self.action_space = action_space

        # Shared feature extraction with LayerNorm + Tanh (OpenAI PPO baseline style)
        self.shared_layers = nn.Sequential(
            nn.Linear(observation_space, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Tanh()
        )

        # Actor (policy) head - outputs logits
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_space)
        )

        # Critic (value) head
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Orthogonal init for training stability"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        features = self.shared_layers(x)
        action_logits = self.actor_head(features)
        state_value = self.critic_head(features)
        return action_logits, state_value


class PPOAgent:
    """Proximal Policy Optimization Agent"""

    def __init__(self,
                 action_space,
                 observation_space,
                 learning_rate=0.0001,   # reduced from 0.0003 for stability
                 gamma=0.99,
                 gae_lambda=0.95,
                 clip_ratio=0.1,         # tightened from 0.2
                 value_coef=1.0,         # increased from 0.5 so critic keeps up
                 entropy_coef=0.01,
                 max_grad_norm=0.5,
                 hidden_size=256,
                 kl_threshold=0.02,
                 critic_loss_floor=0.1,
                 device=None):
        """
        Args:
            action_space:       List of action indices
            observation_space:  Size of observation vector
            learning_rate:      Base LR (critic head uses 0.5x this)
            gamma:              Discount factor
            gae_lambda:         GAE lambda for advantage estimation
            clip_ratio:         PPO clipping parameter
            value_coef:         Weight of critic loss in total loss
            entropy_coef:       Weight of entropy bonus
            max_grad_norm:      Gradient clipping threshold
            hidden_size:        Hidden layer width
            kl_threshold:       Max KL before early-stopping an epoch
            critic_loss_floor:  Skip critic update when loss is below this
            device:             torch device (CPU/GPU)
        """
        self.action_space      = action_space
        self.observation_space = observation_space
        self.learning_rate     = learning_rate
        self.gamma             = gamma
        self.gae_lambda        = gae_lambda
        self.clip_ratio        = clip_ratio
        self.value_coef        = value_coef
        self.entropy_coef      = entropy_coef
        self.max_grad_norm     = max_grad_norm
        self.kl_threshold      = kl_threshold
        self.critic_loss_floor = critic_loss_floor

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.network = PPONetwork(
            observation_space,
            len(action_space),
            hidden_size
        ).to(self.device)

        # Critic head gets half the learning rate so it doesn't overshoot
        self.optimizer = optim.Adam([
            {'params': self.network.shared_layers.parameters(), 'lr': learning_rate},
            {'params': self.network.actor_head.parameters(),    'lr': learning_rate},
            {'params': self.network.critic_head.parameters(),   'lr': learning_rate * 0.5}
        ])

        self.trajectory = {
            'observations':  [],
            'actions':       [],
            'action_indices':[],
            'rewards':       [],
            'values':        [],
            'log_probs':     [],
            'dones':         []
        }

        self.training_losses = []
        self.total_steps     = 0

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, observation, training=True):
        """
        Returns:
            action:      Value from action_space
            action_idx:  Integer index (store this, not the action value)
            log_prob:    Log probability of the chosen action
            value:       Critic estimate of current state
        """
        self.total_steps += 1

        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            action_logits, state_value = self.network(obs_tensor)

        action_dist = Categorical(logits=action_logits)

        if training:
            action_idx = action_dist.sample()
        else:
            action_idx = action_logits.argmax(dim=1)

        log_prob = action_dist.log_prob(action_idx)

        return (
            self.action_space[action_idx.item()],
            action_idx.item(),
            log_prob.item(),
            state_value.item()
        )

    # ------------------------------------------------------------------
    # Trajectory storage
    # ------------------------------------------------------------------

    def store_transition(self, observation, action, action_idx, reward, value, log_prob, done):
        self.trajectory['observations'].append(observation)
        self.trajectory['actions'].append(action)
        self.trajectory['action_indices'].append(action_idx)
        self.trajectory['rewards'].append(reward)
        self.trajectory['values'].append(value)
        self.trajectory['log_probs'].append(log_prob)
        self.trajectory['dones'].append(done)

    # ------------------------------------------------------------------
    # GAE
    # ------------------------------------------------------------------

    def compute_gae(self, next_value):
        """
        Generalised Advantage Estimation.

        Returns:
            advantages: shape (T,)
            returns:    shape (T,)  — used as value targets
        """
        rewards = np.array(self.trajectory['rewards'])
        values  = np.array(self.trajectory['values'] + [next_value])
        dones   = np.array(self.trajectory['dones'])

        advantages = np.zeros_like(rewards)
        gae = 0.0

        # Simplified: no redundant special-case for the last timestep
        for t in reversed(range(len(rewards))):
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * values[t + 1] * next_non_terminal - values[t]
            gae   = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[t] = gae

        returns = advantages + values[:-1]
        return advantages, returns

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, next_value, num_epochs=10, batch_size=64):
        """
        PPO update on the current trajectory.

        Returns:
            mean_loss: average total loss across all mini-batches
        """
        advantages, returns = self.compute_gae(next_value)

        # Normalise advantages for stable gradients
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Build tensors once, outside the loop
        observations  = torch.FloatTensor(np.array(self.trajectory['observations'])).to(self.device)
        actions       = torch.LongTensor(self.trajectory['action_indices']).to(self.device)
        old_log_probs = torch.FloatTensor(self.trajectory['log_probs']).to(self.device)
        old_values    = torch.FloatTensor(self.trajectory['values']).to(self.device)
        advantages    = torch.FloatTensor(advantages).to(self.device)
        returns       = torch.FloatTensor(returns).to(self.device)

        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        num_samples = len(self.trajectory['observations'])
        indices     = np.arange(num_samples)
        epoch_losses = []

        for epoch in range(num_epochs):
            np.random.shuffle(indices)
            kl_exceeded = False  # flag so both loops break cleanly

            for i in range(0, num_samples, batch_size):
                batch_idx = indices[i:i + batch_size]

                # ── Forward pass ──────────────────────────────────────
                action_logits, state_values = self.network(observations[batch_idx])
                state_values = state_values.squeeze(-1)  # safe for any batch size

                dist         = Categorical(logits=action_logits)
                new_log_probs = dist.log_prob(actions[batch_idx])
                entropy      = dist.entropy().mean()

                # ── Actor (PPO-clip) loss ──────────────────────────────
                ratio  = torch.exp(new_log_probs - old_log_probs[batch_idx])
                surr1  = ratio * advantages[batch_idx]
                surr2  = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) \
                         * advantages[batch_idx]
                actor_loss = -torch.min(surr1, surr2).mean()

                # ── Critic (value-clip) loss ───────────────────────────
                value_pred_clipped = old_values[batch_idx] + torch.clamp(
                    state_values - old_values[batch_idx],
                    -self.clip_ratio,
                    self.clip_ratio
                )
                critic_loss_unclipped = nn.functional.mse_loss(state_values,         returns[batch_idx])
                critic_loss_clipped   = nn.functional.mse_loss(value_pred_clipped,   returns[batch_idx])
                critic_loss           = torch.max(critic_loss_unclipped, critic_loss_clipped)


                critic_loss = nn.functional.mse_loss(state_values, returns[batch_idx])

                # Detect value collapse — warn but skip the critic term
                # so a degenerate critic doesn't drag the actor off-course
                critic_collapsed = critic_loss.item() < self.critic_loss_floor
                if critic_collapsed:
                    value_std = state_values.std().item()
                    print(
                        f"  [critic collapse] loss={critic_loss.item():.4f}  "
                        f"value_std={value_std:.4f}  — critic may be collapsed"
                    )

                # ── Total loss ────────────────────────────────────────
                # if critic_collapsed:
                #     loss = actor_loss - self.entropy_coef * entropy
                # else:
                #     loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
                # Total loss — always include critic term
                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

                # ── Backward pass ─────────────────────────────────────
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                epoch_losses.append(loss.item())

                # ── KL early stopping (breaks both loops) ─────────────
                with torch.no_grad():
                    kl = (old_log_probs[batch_idx] - new_log_probs).mean().item()

                if kl > self.kl_threshold:
                    print(f"  [early stop] epoch {epoch + 1}  KL={kl:.4f} > {self.kl_threshold}")
                    kl_exceeded = True
                    break

            if kl_exceeded:
                break

        # Clear trajectory for next rollout
        self.trajectory = {
            'observations':  [],
            'actions':       [],
            'action_indices':[],
            'rewards':       [],
            'values':        [],
            'log_probs':     [],
            'dones':         []
        }

        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        self.training_losses.append(mean_loss)
        return mean_loss

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, filepath):
        torch.save(self.network.state_dict(), filepath)
        print(f"✓ Model saved to {filepath}")

    def load_model(self, filepath):
        self.network.load_state_dict(torch.load(filepath, map_location=self.device))
        print(f"✓ Model loaded from {filepath}")

    def get_stats(self):
        return {
            'total_steps': self.total_steps,
            'avg_loss':    float(np.mean(self.training_losses[-100:])) if self.training_losses else 0.0
        }
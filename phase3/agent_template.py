import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical


class PPONetwork(nn.Module):
    """PPO Network with Actor-Critic Architecture"""
    def __init__(self, observation_space, action_space, hidden_size=128):
        super(PPONetwork, self).__init__()
        
        self.observation_space = observation_space
        self.action_space = action_space
        
        self.shared_layers = nn.Sequential(
            nn.Linear(observation_space, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_space),
            nn.Softmax(dim=-1)
        )
        
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x):
        features = self.shared_layers(x)
        action_probs = self.actor_head(features)
        value = self.critic_head(features)
        return action_probs, value

_model_cache = None
_optimizer_cache = None

def _load_model():
    """Load model once and cache it"""
    global _model_cache, _optimizer_cache
    if _model_cache is None:
        model = PPONetwork(observation_space=18, action_space=5, hidden_size=256)
        try:
            model.load_state_dict(torch.load('ppo_obelix_model.pth', map_location='cpu'))
            print("✓ Model loaded successfully")
        except FileNotFoundError:
            print("✗ Model file not found - using random initialization")
        model.eval()
        _model_cache = model
        _optimizer_cache = optim.Adam(model.parameters(), lr=3e-4)
    return _model_cache, _optimizer_cache


def policy(obs, rng):
    """
    PPO policy function for evaluation
    
    Args:
        obs: numpy array of shape (18,) - sensor feedback
        rng: numpy random generator
    
    Returns:
        action: str in ["L45", "L22", "FW", "R22", "R45"]
    """
    model, _ = _load_model()
    
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
    
    with torch.no_grad():
        action_probs, _ = model(obs_tensor)
    
    action_dist = Categorical(action_probs)
    action_idx = action_dist.sample().item()
    
    action_map = {
        0: "L45",
        1: "L22",
        2: "FW",
        3: "R22",
        4: "R45"
    }
    
    return action_map[action_idx]


# def compute_gae(rewards, values, gamma=0.99, lambda_=0.95):
#     """Compute Generalized Advantage Estimation"""
#     advantages = []
#     gae = 0
#     for t in reversed(range(len(rewards))):
#         if t == len(rewards) - 1:
#             next_value = 0
#         else:
#             next_value = values[t + 1]
#         delta = rewards[t] + gamma * next_value - values[t]
#         gae = delta + gamma * lambda_ * gae
#         advantages.insert(0, gae)
#     return torch.tensor(advantages, dtype=torch.float32)


# def ppo_update(batch_obs, batch_actions, batch_rewards, batch_values, clip_ratio=0.2, epochs=3):
#     """PPO update step"""
#     model, optimizer = _load_model()
    
#     obs_tensor = torch.FloatTensor(batch_obs)
#     actions_tensor = torch.LongTensor(batch_actions)
#     returns = torch.FloatTensor([r + v for r, v in zip(batch_rewards, batch_values)])
#     advantages = compute_gae(batch_rewards, batch_values)
#     advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
#     old_action_probs, _ = model(obs_tensor)
#     old_action_probs = old_action_probs.detach()
    
#     for _ in range(epochs):
#         action_probs, values = model(obs_tensor)
#         dist = Categorical(action_probs)
#         new_log_probs = dist.log_prob(actions_tensor)
#         old_log_probs = torch.log(old_action_probs.gather(1, actions_tensor.unsqueeze(1)) + 1e-8).squeeze(1)
        
#         ratio = torch.exp(new_log_probs - old_log_probs)
#         surr1 = ratio * advantages
#         surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
#         actor_loss = -torch.min(surr1, surr2).mean()
#         critic_loss = nn.MSELoss()(values.squeeze(1), returns)
#         loss = actor_loss + 0.5 * critic_loss
        
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
    
#     torch.save(model.state_dict(), 'ppo_obelix_model_best.pth')
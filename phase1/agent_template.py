import torch
import torch.nn as nn
import numpy as np


class DuelingDQNNetwork(nn.Module):
    """Dueling DQN Network Architecture"""
    def __init__(self, observation_space, action_space, hidden_size=128):
        super(DuelingDQNNetwork, self).__init__()
        
        self.observation_space = observation_space
        self.action_space = action_space
        
        self.feature_layers = nn.Sequential(
            nn.Linear(observation_space, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_space)
        )
    
    def forward(self, x):
        features = self.feature_layers(x)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values


# Load model once at module level (caching)
_model_cache = None

def _load_model():
    """Load model once and cache it"""
    global _model_cache
    if _model_cache is None:
        model = DuelingDQNNetwork(observation_space=18, action_space=5, hidden_size=128)
        try:
            model.load_state_dict(torch.load('d3qn_obelix_model.pth', map_location='cpu'))
            print("✓ Model loaded successfully")
        except FileNotFoundError:
            print("✗ Model file not found - using random initialization")
        model.eval()
        _model_cache = model
    return _model_cache


def policy(obs, rng):
    """
    Policy function for evaluation
    
    Args:
        obs: numpy array of shape (18,) - sensor feedback
        rng: numpy random generator
    
    Returns:
        action: str in ["L45", "L22", "FW", "R22", "R45"]
    """
    model = _load_model()
    
    # Convert observation to tensor
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
    
    # Get Q-values
    with torch.no_grad():
        q_values = model(obs_tensor)
    
    # Select greedy action
    action_idx = q_values.argmax(dim=1).item()
    
    action_map = {
        0: "L45",
        1: "L22",
        2: "FW",
        3: "R22",
        4: "R45"
    }
    
    return action_map[action_idx]
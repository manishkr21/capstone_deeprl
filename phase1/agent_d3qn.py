import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class DuelingDQNNetwork(nn.Module):
    """Dueling DQN Network Architecture - Simplified for Stable Training"""
    def __init__(self, observation_space, action_space, hidden_size=128):
        super(DuelingDQNNetwork, self).__init__()
        
        self.observation_space = observation_space
        self.action_space = action_space
        
        # Simpler shared feature extraction layers
        self.feature_layers = nn.Sequential(
            nn.Linear(observation_space, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Value stream (V)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        # Advantage stream (A)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_space)
        )
    
    def forward(self, x):
        """Forward pass through dueling architecture"""
        # Shared feature extraction
        features = self.feature_layers(x)
        
        # Value and advantage streams
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Dueling combination: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values


class ReplayBuffer:
    """Experience Replay Buffer with Reward Normalization"""
    def __init__(self, max_size=10000):
        self.buffer = deque(maxlen=max_size)
        self.rewards = []  # Track for normalization
    
    def store(self, state, action, reward, next_state, done):
        """Store experience in buffer"""
        self.buffer.append((state, action, reward, next_state, done))
        self.rewards.append(reward)
        # Keep only last 1000 rewards for statistics
        if len(self.rewards) > 1000:
            self.rewards.pop(0)
    
    def get_reward_stats(self):
        """Get mean and std of recent rewards"""
        if len(self.rewards) < 2:
            return 0, 1
        return np.mean(self.rewards), max(np.std(self.rewards), 0.01)
    
    def sample(self, batch_size):
        """Sample batch from buffer"""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
        states = np.array([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch])
        next_states = np.array([exp[3] for exp in batch])
        dones = np.array([exp[4] for exp in batch])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class D3QNAgent:
    """Dueling Double Deep Q-Network Agent"""
    def __init__(self, 
                 action_space,
                 observation_space,
                 learning_rate=1e-3,
                 gamma=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 epsilon_decay=0.995,
                 buffer_size=10000,
                 hidden_size=128,
                 device=None):
        
        self.action_space = action_space
        self.observation_space = observation_space
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.hidden_size = hidden_size
        
        # Device setup (GPU or CPU)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.q_network = DuelingDQNNetwork(
            observation_space, len(action_space), hidden_size
        ).to(self.device)
        
        self.target_network = DuelingDQNNetwork(
            observation_space, len(action_space), hidden_size
        ).to(self.device)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Set to evaluation mode
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Training statistics
        self.total_steps = 0
        self.training_losses = []

    def select_action(self, observation, training=True):
        """Select action using epsilon-greedy policy"""
        if training and np.random.rand() < self.epsilon:
            # Exploration: random action
            return random.choice(self.action_space)
        else:
            # Exploitation: greedy action
            with torch.no_grad():
                state = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
                q_values = self.q_network(state)
                action_idx = q_values.argmax(dim=1).item()
                return self.action_space[action_idx]

    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        # Convert action to index if necessary
        if isinstance(action, int):
            action_idx = action
        else:
            action_idx = self.action_space.index(action)
        
        self.replay_buffer.store(state, action_idx, reward, next_state, done)
        self.total_steps += 1

    def train(self, batch_size=32, target_update_freq=1000):
        """Train agent using D3QN algorithm with reward normalization"""
        if len(self.replay_buffer) < batch_size:
            return None
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # Normalize rewards to reduce variance
        reward_mean, reward_std = self.replay_buffer.get_reward_stats()
        rewards_normalized = (rewards - reward_mean) / reward_std
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards_normalized).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Double DQN: Use Q-network to select actions, target network to evaluate
        with torch.no_grad():
            # Select actions using Q-network (Double DQN)
            next_q_values = self.q_network(next_states)
            next_actions = next_q_values.argmax(dim=1)
            
            # Evaluate actions using target network
            next_q_values_target = self.target_network(next_states)
            next_q_values_selected = next_q_values_target.gather(1, next_actions.unsqueeze(1))
            
            # Calculate target Q-values
            target_q_values = rewards.unsqueeze(1) + self.gamma * next_q_values_selected * (1 - dones.unsqueeze(1))
        
        # Calculate current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Calculate loss (MSE)
        loss = self.criterion(current_q_values, target_q_values)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Decay epsilon more gradually
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # Update target network periodically
        if self.total_steps % target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        loss_value = loss.item()
        self.training_losses.append(loss_value)
        
        return loss_value

    def save_model(self, filepath):
        """Save only model weights (not full agent state)"""
        torch.save(self.q_network.state_dict(), filepath)
        print(f"✓ Model weights saved to {filepath}")

    def save_full_checkpoint(self, filepath):
        """Save complete agent checkpoint (for resuming training)"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'total_steps': self.total_steps
        }, filepath)
        print(f"✓ Full checkpoint saved to {filepath}")

    def load_model(self, filepath):
        """Load model weights"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.total_steps = checkpoint['total_steps']
        print(f"Model loaded from {filepath}")

    def get_stats(self):
        """Get training statistics"""
        return {
            'total_steps': self.total_steps,
            'epsilon': self.epsilon,
            'avg_loss': np.mean(self.training_losses[-100:]) if self.training_losses else 0,
            'buffer_size': len(self.replay_buffer)
        }
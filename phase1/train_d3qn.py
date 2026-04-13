import gymnasium as gym
import numpy as np
from obelix_gym_wrapper import OBELIXGymWrapper
from agent_d3qn import D3QNAgent

def train_with_obelix():
    """Train D3QN agent on OBELIX environment"""
    
    # Create wrapped environment
    env = OBELIXGymWrapper(
        scaling_factor=5,
        arena_size=500,
        max_steps=1000,
        wall_obstacles=False,
        difficulty=0,
        box_speed=2,
        render_mode=None  # Set to "human" to visualize
    )
    
    # Initialize D3QN agent
    agent = D3QNAgent(
        action_space=list(range(5)),
        observation_space=18,
        learning_rate=1e-3,  # Higher LR - agent needs to learn faster
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,  # Keep exploration at 5% even late
        epsilon_decay=0.9999,  # CRITICAL: Much slower - stay exploring 200+ episodes
        buffer_size=100000,  # Larger buffer for better experience diversity
        hidden_size=256,  # Smaller network - was overfitting with 512
        device=None  # Automatically use GPU if available

    )
    
    num_episodes = 2500
    batch_size = 128  # Larger batch for more stable updates
    target_update_freq = 500  # Update target network more frequently
    
    print("Training D3QN on OBELIX Environment (Gymnasium)")
    print("=" * 60)
    
    for episode in range(num_episodes):
        observation, info = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Agent selects action
            action = agent.select_action(observation, training=True)
            
            # Execute action
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Remove step penalty - it hurts learning
            # Store experience
            agent.store_experience(observation, action, reward, next_observation, done)
            
            # Train only after buffer has sufficient data
            if len(agent.replay_buffer) > batch_size * 2:
                loss = agent.train(batch_size=batch_size, target_update_freq=target_update_freq)
            
            episode_reward += reward
            observation = next_observation
        
        if (episode + 1) % 10 == 0:
            stats = agent.get_stats()
            print(f"Episode {episode + 1}: Reward={episode_reward:.2f}, "
                  f"Loss={stats['avg_loss']:.4f}, Epsilon={stats['epsilon']:.4f}")
    
    agent.save_model('d3qn_obelix_model.pth')
    print("Training complete!")

if __name__ == "__main__":
    train_with_obelix()
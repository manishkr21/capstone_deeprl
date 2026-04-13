import torch
import numpy as np
from ppo_agent import PPOAgent, PPONetwork
from obelix_gym_wrapper import OBELIXGymWrapper


class PPOEvaluator:
    """Evaluate trained PPO agent"""
    
    def __init__(self, model_path, device=None):
        """
        Initialize evaluator
        
        Args:
            model_path: Path to saved model weights
            device: torch device
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create agent
        self.agent = PPOAgent(
            action_space=list(range(5)),
            observation_space=18,
            hidden_size=256,
            device=self.device
        )
        
        # Load model
        try:
            self.agent.load_model(model_path)
            print(f"✓ Model loaded from {model_path}")
        except FileNotFoundError:
            print(f"✗ Model file not found: {model_path}")
    
    def evaluate(self, num_episodes=10, max_steps=1000, render=False):
        """
        Evaluate agent over multiple episodes
        
        Args:
            num_episodes: Number of evaluation episodes
            max_steps: Maximum steps per episode
            render: Whether to render episodes
        
        Returns:
            results: Dictionary with evaluation metrics
        """
        env = OBELIXGymWrapper(
            scaling_factor=5,
            arena_size=500,
            max_steps=max_steps,
            wall_obstacles=False,
            difficulty=0,
            render_mode="human" if render else None
        )
        
        episode_rewards = []
        episode_lengths = []
        successes = 0
        
        print("=" * 60)
        print(f"Evaluating PPO Agent ({num_episodes} episodes)")
        print("=" * 60)
        
        for episode in range(num_episodes):
            observation, info = env.reset()
            episode_reward = 0
            episode_step = 0
            done = False
            
            while not done and episode_step < max_steps:
                # Select action (greedy)
                action, _, _, _ = self.agent.select_action(observation, training=False)
                
                # Execute action
                next_observation, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                observation = next_observation
                episode_step += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_step)
            
            # Check if successful (box at boundary)
            if info.get('current_step', 0) < max_steps and terminated:
                successes += 1
            
            print(f"Episode {episode + 1:2d}: Reward = {episode_reward:8.2f}, "
                  f"Length = {episode_step:4d}, "
                  f"Success = {'✓' if successes == episode + 1 else '✗'}")
        
        # Compute statistics
        results = {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'success_rate': successes / num_episodes,
            'num_successes': successes
        }
        
        print("\n" + "=" * 60)
        print("Evaluation Results:")
        print("=" * 60)
        print(f"Mean Reward:     {results['mean_reward']:8.2f} ± {results['std_reward']:8.2f}")
        print(f"Mean Length:     {results['mean_length']:8.2f}")
        print(f"Success Rate:    {results['success_rate']*100:6.2f}% ({successes}/{num_episodes})")
        print("=" * 60 + "\n")
        
        env.close()
        return results


def evaluate_policy(obs, rng):
    """Policy function for evaluation (compatible with evaluate.py)"""
    # Load model once (caching)
    if not hasattr(evaluate_policy, '_model'):
        print("Loading PPO model...")
        agent = PPOAgent(
            action_space=list(range(5)),
            observation_space=18,
            hidden_size=256
        )
        try:
            agent.load_model('ppo_obelix_model.pth')
        except FileNotFoundError:
            print("Model file not found, using random initialization")
        evaluate_policy._model = agent
    
    agent = evaluate_policy._model
    
    # Get action
    action, _, _ = agent.select_action(obs, training=False)
    
    return action


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate PPO agent")
    parser.add_argument("--model", type=str, default="ppo_obelix_model.pth",
                        help="Path to model file")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of evaluation episodes")
    parser.add_argument("--max-steps", type=int, default=1000,
                        help="Maximum steps per episode")
    parser.add_argument("--render", action="store_true",
                        help="Render episodes")
    
    args = parser.parse_args()
    
    evaluator = PPOEvaluator(args.model)
    results = evaluator.evaluate(
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        render=args.render
    )

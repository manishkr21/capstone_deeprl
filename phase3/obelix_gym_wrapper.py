import gymnasium as gym
from gymnasium import spaces
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from obelix import OBELIX


class OBELIXGymWrapper(gym.Env):
    """Gymnasium wrapper for OBELIX environment"""
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self,
                 scaling_factor=5,
                 arena_size=500,
                 max_steps=1000,
                 wall_obstacles=False,
                 difficulty=0,
                 box_speed=2,
                 seed=None,
                 render_mode=None):
        """
        Initialize OBELIX Gymnasium wrapper
        
        Args:
            scaling_factor: Scaling factor for robot/box size
            arena_size: Size of arena
            max_steps: Maximum steps per episode
            wall_obstacles: Whether to include obstacles
            difficulty: Difficulty level
            box_speed: Speed of moving box
            seed: Random seed
            render_mode: Rendering mode ('human', 'rgb_array', None)
        """
        
        # Initialize OBELIX
        self.obelix = OBELIX(
            scaling_factor=scaling_factor,
            arena_size=arena_size,
            max_steps=max_steps,
            wall_obstacles=wall_obstacles,
            difficulty=difficulty,
            box_speed=box_speed,
            seed=seed
        )
        
        self.render_mode = render_mode
        
        # Action space (5 discrete actions)
        self.action_space = spaces.Discrete(5)
        self.action_map = {
            0: "L45",
            1: "L22",
            2: "FW",
            3: "R22",
            4: "R45"
        }
        
        # Observation space (18-bit sensor feedback)
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(18,),
            dtype=np.float32
        )
    
    def reset(self, seed=None, options=None):
        """Reset environment"""
        super().reset(seed=seed)
        
        if seed is not None:
            self.obelix.reset(seed=seed)
        else:
            self.obelix.reset()
        
        observation = self.obelix.sensor_feedback.copy().astype(np.float32)
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        """
        Execute one step
        
        Args:
            action: Action index (0-4)
        
        Returns:
            observation: Current observation
            reward: Reward for this step
            terminated: Whether episode is done (success/failure)
            truncated: Whether episode is truncated (max steps)
            info: Additional info
        """
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        
        # Convert action to OBELIX format
        action_str = self.action_map[action]
        
        # Execute step
        render = self.render_mode == "human"
        observation, reward, done = self.obelix.step(action_str, render=render)
        
        # Convert observation to float32
        observation = observation.copy().astype(np.float32)
        
        # Determine termination and truncation
        terminated = done  # Episode ended due to success/failure
        truncated = self.obelix.current_step >= self.obelix.max_steps  # Max steps reached
        
        info = self._get_info()
        
        return observation, float(reward), terminated, truncated, info
    
    def render(self):
        """Render environment"""
        if self.render_mode == "human":
            self.obelix.render_frame()
        elif self.render_mode == "rgb_array":
            return self.obelix.frame.copy()
        return None
    
    def close(self):
        """Close environment"""
        pass
    
    def _get_info(self):
        """Get environment info"""
        return {
            "bot_position": (self.obelix.bot_center_x, self.obelix.bot_center_y),
            "box_position": (self.obelix.box_center_x, self.obelix.box_center_y),
            "facing_angle": self.obelix.facing_angle,
            "enable_push": self.obelix.enable_push,
            "active_state": self.obelix.active_state,
            "current_step": self.obelix.current_step,
            "stuck_flag": self.obelix.stuck_flag,
            "reward": self.obelix.reward
        }

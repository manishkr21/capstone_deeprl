import gymnasium as gym
from gymnasium import spaces
import numpy as np
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
        
        # Define action space (5 discrete actions)
        self.action_space = spaces.Discrete(5)
        self.action_map = {
            0: "L45",
            1: "L22",
            2: "FW",
            3: "R22",
            4: "R45"
        }
        
        # Define observation space (18-bit sensor feedback)
        self.observation_space = spaces.Box(
            low=0, 
            high=1, 
            shape=(18,), 
            dtype=np.float32
        )
    
    def reset(self, seed=None, options=None):
        """Reset environment"""
        if seed is not None:
            self.obelix.reset(seed=seed)
        else:
            self.obelix.reset()
        
        observation = self.obelix.sensor_feedback.copy().astype(np.float32)
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        """Execute one step"""
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        
        # Convert action to OBELIX format
        action_str = self.action_map[action]
        
        # Execute step
        render = self.render_mode == "human"
        observation, reward, done = self.obelix.step(action_str, render=render)
        
        # Convert observation to float32
        observation = observation.copy().astype(np.float32)
        
        info = self._get_info()
        
        return observation, float(reward), done, False, info
    
    def render(self):
        """Render environment"""
        if self.render_mode == "human":
            self.obelix.render_frame()
        elif self.render_mode == "rgb_array":
            return self.obelix.frame.copy()
    
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
            "stuck_flag": self.obelix.stuck_flag
        }
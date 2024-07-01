import gym
from .EnvBase import EnvBase
class FrozenLakeEnv(EnvBase):
    def __init__(self):
        super().__init__()
        self.env = gym.make('FrozenLake-v1', desc=None, map_name='4x4', is_slippery=False, render_mode='human')
        self.state_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.n_state = self.state_space.n
        self.n_action = self.action_space.n
    
    def reset(self):
        return self.env.reset()
        
    def step(self, action):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return next_state, reward, done, info
    
    def sample_action(self):
        return self.action_space.sample()
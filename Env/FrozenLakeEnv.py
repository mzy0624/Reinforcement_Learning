import gym
from .EnvBase import EnvBase
class FrozenLakeEnv(EnvBase):
    def __init__(self):
        env = gym.make('FrozenLake-v1', desc=None, map_name='4x4', is_slippery=False, render_mode='human')
        super().__init__(
            env=env,
            state_type='Discrete',
            action_type='Discrete',
            state_dim=env.observation_space.n,
            action_dim=env.action_space.n
        )
    
    def reset(self, one_hot=True):
        state, info = self.env.reset()
        if one_hot:
            state = self.one_hot_encode(state, self.state_dim)
        ''' state: int 0~15 or one_hot_encode list len=16'''
        return state, info
    
    def step(self, action, one_hot=True):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        if one_hot:
            next_state = self.one_hot_encode(next_state, self.state_dim)
        return next_state, reward, done, info
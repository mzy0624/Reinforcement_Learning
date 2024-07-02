import gym
from .EnvBase import EnvBase
class CartPoleV1(EnvBase):
    def __init__(self):
        env = gym.make('CartPole-v1')
        super().__init__(
            env=env,
            state_type='Continuous',
            action_type='Discrete',
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n
        )
    def reset(self):
        state, info = self.env.reset()
        ''' state: list len=4 '''
        return state, info
    
    def step(self, action):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return next_state, reward, done, info
import torch
import time
import numpy as np

class Base:
    def __init__(self, env, policy, gamma, episodes):
        self.env = env
        self.policy = policy
        self.gamma = gamma
        self.episodes = episodes
        self.state_dim = env.state_dim
        self.action_dim = env.action_dim
    
    def select_action(self, state):        
        with torch.no_grad():
            probs = self.policy(state)
            return torch.multinomial(probs, num_samples=1).item()
    
    def train_an_episode(self, episode):
        pass

    def train(self):
        for episode in range(self.episodes):
            self.train_an_episode(episode)
    
    def play(self):
        state, info = self.env.reset()
        done = False
        while not done:
            state = torch.tensor(np.array(state), dtype=torch.float)
            action = self.select_action(state)
            next_state, reward, done, info = self.env.step(action)
            state = next_state
            time.sleep(0.01)
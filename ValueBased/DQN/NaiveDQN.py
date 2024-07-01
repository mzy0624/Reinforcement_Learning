from ..Base import Base
from Network import Network
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from itertools import count

class NaiveDQN(Base):
    '''
        DQN without Replay Buffer and Target Network
    '''
    def __init__(self, env, alpha=0.001, gamma=0.95, episodes=1000, max_epsilon=1, min_epsilon=0.05, epsilon_decay_rate=0.005):
        super().__init__(env, alpha, gamma, episodes, max_epsilon, min_epsilon, epsilon_decay_rate)
        
    def build_Q(self):
        self.Q_path = './build/Q_DQN.bin'
        self.Q_Net = Network(self.n_state, self.n_action)
        if os.path.exists(self.Q_path):
            self.Q_Net.load_state_dict(torch.load(self.Q_path))
        self.optimizer = optim.Adam(self.Q_Net.parameters(), lr=self.alpha)
        self.criterion = nn.MSELoss()
        
    def save_Q(self):
        torch.save(self.Q_Net.state_dict(), self.Q_path)

    def epsilon_greedy(self, state):
        if np.random.uniform() < self.epsilon:
            return self.env.sample_action()
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = self.Q_Net(state)
            return q_values.max(1)[1].item()

    def one_hot_encode(self, state):
        one_hot = np.zeros(self.n_state)
        one_hot[state] = 1
        return one_hot
    
    def select_action(self, state):
        return self.epsilon_greedy(state)

    def update_Q(self, state, action, reward, next_state, done=None):
        state      = torch.tensor(state,      dtype=torch.float32).unsqueeze(0)    # (1, 16)
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)    # (1, 16)
        reward     = torch.tensor([reward],   dtype=torch.float32)                 # (1)
        action     = torch.tensor([[action]], dtype=torch.int64)                   # (1, 1)
        
        q_values = self.Q_Net(state)                # (1, n_action)
        next_q_values = self.Q_Net(next_state)      # (1, n_action)
        target = reward + (1 - done) * self.gamma * next_q_values.max(1)[0]
        target_f = q_values.clone()
        target_f[0][action] = target
        
        loss = self.criterion(q_values, target_f)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def train_an_episode(self, episode):
        state, info = self.env.reset()
        for t in count():
            state = self.one_hot_encode(state)
            action = int(self.select_action(state))
            next_state, reward, done, info = self.env.step(action)
            next_state = self.one_hot_encode(next_state)
            if done and reward == 0:
                reward = -1
            self.update_Q(state, action, reward, next_state, done)
            state = next_state

            if done:
                self.update_epsilon(episode)
                if reward > 0:
                    print(f'{episode = }, {t = }, {self.epsilon}')
                    self.save_Q()
                break
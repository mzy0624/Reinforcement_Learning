from .Base import Base
import os
import pandas as pd
from itertools import count

class SARSA(Base):
    '''
        For only discrete action and discrete state
    '''
    def __init__(self, env, alpha=0.5, gamma=0.95, episodes=100, max_epsilon=1, min_epsilon=0.05, epsilon_decay_rate=0.005):
        super().__init__(env, alpha, gamma, episodes, max_epsilon, min_epsilon, epsilon_decay_rate)
        
    def build_Q(self):
        self.Q_path = './build/Q_Table_SARSA.csv'
        if os.path.exists(self.Q_path):
            self.Q = pd.read_csv(self.Q_path, header=0, dtype='float')
        else:
            self.Q = pd.DataFrame([[0.0] * self.action_dim] * self.state_dim)
    
    def save_Q(self):
        self.Q.to_csv(self.Q_path, index=False)
    
    def greedy(self, state):
        return self.Q.iloc[state].idxmax()
    
    def update_Q(self, state, action, reward, next_state, next_action=None):
        Q = self.Q.iloc[state, action]
        Q_new = Q + self.alpha * (reward + self.gamma * self.Q.iloc[next_state, next_action] - Q)
        self.Q.iloc[state, action] = Q_new    

    def train_an_episode(self, episode):
        state, info = self.env.reset(one_hot=False)
        action = int(self.select_action(state))
        for t in count():
            next_state, reward, done, info = self.env.step(action, one_hot=False)
            next_action = int(self.select_action(next_state))
            if done and reward == 0:
                reward = -1
            self.update_Q(state, action, reward, next_state, next_action)
            state, action = next_state, next_action
            
            if done:
                self.update_epsilon(episode)
                if reward > 0:
                    print(f'{episode = }, {t = }')
                    self.save_Q()
                break
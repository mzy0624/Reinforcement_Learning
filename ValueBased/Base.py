import time
import numpy as np

class Base:
    def __init__(self, env, alpha, gamma, episodes, max_epsilon, min_epsilon, epsilon_decay_rate):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.episodes = episodes
        self.epsilon = max_epsilon
        def update_epsilon(episode):
            self.epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-epsilon_decay_rate * episode)
        self.update_epsilon = update_epsilon
        self.n_state = env.n_state
        self.n_action = env.n_action
        self.build_Q()
    
    def build_Q(self):
        self.Q_path = None
        self.Q = None
    
    def epsilon_greedy(self, state):
        if np.random.uniform() < self.epsilon or (self.Q.iloc[state] == 0).all():
            return self.env.sample_action()
        return self.Q.iloc[state].idxmax()
    
    def select_action(self, state):
        return self.epsilon_greedy(state)
    
    def save_Q(self):
        pass
    
    def train_an_episode(self, episode):
        pass
    
    def train(self):
        for episode in range(self.episodes):
            self.train_an_episode(episode)
    
    def play(self):
        state, info = self.env.reset()
        done = False
        while not done:
            action = self.select_action(state)
            next_state, reward, done, info = self.env.step(action)
            state = next_state
            time.sleep(0.01)
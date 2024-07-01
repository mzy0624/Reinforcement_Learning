from Network import Network
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from itertools import count
import time

class Reinforce:
    def __init__(self, env, alpha=0.005, gamma=0.99, episodes=1000):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.episodes = episodes
        self.n_state = env.n_state
        self.n_action = env.n_action
        self.policy = Network(self.n_state, self.n_action)
        self.policy.load_state_dict(torch.load('./build/Q_DQN.bin'))
        self.optimizer = optim.Adam(self.policy.parameters(), lr=alpha)
        self.criterion = nn.CrossEntropyLoss()
        
        from Visdom import Visdom
        self.vis = Visdom()
        
    def one_hot_encode(self, state):
        one_hot = np.zeros(self.n_state)
        one_hot[state] = 1
        return one_hot
    
    def select_action(self, state):        
        with torch.no_grad():
            probs = F.softmax(self.policy(state), dim=-1)
            return torch.multinomial(probs, num_samples=1).item()
    
    def update_policy(self, episode, rewards, log_probs):
        discounted_rewards = []
        cumulative_reward = 0
        for reward in reversed(rewards):
            cumulative_reward = reward + self.gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)
        
        policy_loss = []
        for log_prob, reward in zip(log_probs, discounted_rewards):
            policy_loss.append(-log_prob * reward)  # gradient ascent
        
        self.optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        
        self.vis.plot(episode, policy_loss, 'Training')
        
    def train_an_episode(self, episode):
        state, info = self.env.reset()
        rewards = []
        log_probs = []
        for t in count():
            state = self.one_hot_encode(state)
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action = self.select_action(state)
            next_state, reward, done, info = self.env.step(action)
            if done and reward == 0:
                reward = -1
            
            log_prob = torch.log(F.softmax(self.policy(state), dim=-1)[0][action])
            # log_prob = torch.log(self.policy(state)[0][action])
            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state
            
            if done:
                self.update_policy(episode, rewards, log_probs)
                if reward > 0:
                    print(f"{episode = }, {t = }")
                break
        # return sum(rewards)

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

    

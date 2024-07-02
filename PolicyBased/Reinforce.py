from Network import PolicyNetwork
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import count
from .Base import Base

class Reinforce(Base):
    def __init__(self, env, alpha=0.005, gamma=0.99, episodes=1000):
        policy = PolicyNetwork(env.state_dim, env.action_dim)
        super().__init__(env, policy, gamma, episodes)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=alpha)
        self.criterion = nn.CrossEntropyLoss()
    
    def update_policy(self, rewards, log_probs):
        discounted_rewards = []
        cumulative_reward = 0
        for reward in reversed(rewards):
            cumulative_reward = reward + self.gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)
        
        policy_loss = []
        for log_prob, reward in zip(log_probs, discounted_rewards):
            policy_loss.append(-log_prob * reward)  # gradient ascent
        
        self.optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        
    def train_an_episode(self, episode):
        state, info = self.env.reset()
        rewards = []
        log_probs = []
        for t in count():
            state = torch.tensor(state, dtype=torch.float)
            action = self.select_action(state)
            next_state, reward, done, info = self.env.step(action)
            if done and reward == 0:
                reward = -1
            
            log_prob = torch.log(self.policy(state)[action])
            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state
            
            if done:
                self.update_policy(rewards, log_probs)
                if reward > 0:
                    print(f"{episode = }, {t = }")
                break
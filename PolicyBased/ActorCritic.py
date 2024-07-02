import torch
import torch.optim as optim
import numpy as np
from itertools import count
from Network import PolicyNetwork, ValueNetwork
from ..Base import Base

class ActorCritic(Base):
    def __init__(self, env, alpha_actor=0.001, alpha_critic=0.01, gamma=0.99, episodes=1000):
        policy = PolicyNetwork(env.state_dim, env.action_dim)
        super().__init__(env, policy, gamma, episodes)
        # Actor
        self.actor = policy
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=alpha_actor)
        # Critic
        self.critic = ValueNetwork(self.state_dim)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=alpha_critic)
 
    def td_error(self, state, next_state, reward, done):
        return reward + self.gamma * self.critic(next_state) * (1 - done) - self.critic(state)
    
    def loss_function(self, state, next_state, reward, done, log_prob):
        delta = self.td_error(state, next_state, reward, done)
        actor_loss = -log_prob * delta.detach()
        critic_loss = 0.5 * delta.pow(2).mean()
        return actor_loss, critic_loss
    
    def train_an_episode(self, episode):
        state, info = self.env.reset()
        for t in count():
            state = torch.tensor(np.array(state), dtype=torch.float)
            action = self.select_action(state)
            next_state, reward, done, info = self.env.step(action)
            next_state = torch.tensor(np.array(next_state), dtype=torch.float)
            if done and reward == 0:
                reward = -1
            reward = torch.tensor(reward, dtype=torch.float).unsqueeze(0)
            done = torch.tensor(done, dtype=torch.float).unsqueeze(0)
            log_prob = torch.log(self.actor(state)[action])
            
            value = self.critic(state)
            next_value = self.critic(next_state)
            td_error = reward + self.gamma * next_value * (1 - done) - value
            
            # Update Critic            
            critic_loss = 0.5 * td_error.pow(2).mean()
            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            self.optimizer_critic.step()
            
            # Update Actor
            actor_loss = -log_prob * td_error.detach()
            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            self.optimizer_actor.step()
            
            state = next_state
            if done:
                if reward > 0:
                    print(f"{episode = }, {t = }")
                break
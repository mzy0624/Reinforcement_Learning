import torch
import torch.optim as optim
import numpy as np
from itertools import count
from Network import PolicyNetwork, ValueNetwork
from ..Base import Base

class ActorCritic(Base):
    def __init__(self, env, alpha_actor=0.001, alpha_critic=0.01, gamma=0.99, episodes=1000):
        # Actor
        actor = PolicyNetwork(env.state_dim, env.action_dim)
        super().__init__(env, actor, gamma, episodes)
        # Critic
        self.actor = self.policy
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=alpha_actor)
        self.critic = ValueNetwork(self.state_dim)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=alpha_critic)

    def select_action(self, state):        
        with torch.no_grad():
            probs = self.actor(state)
            action = torch.multinomial(probs, num_samples=1).item()
        return action

    def train_an_episode(self, episode):
        state, info = self.env.reset()
        rewards = []
        log_probs = []
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
            log_probs.append(log_prob)
            rewards.append(reward)
            
            # Update Critic
            td_error = reward + self.gamma * self.critic(next_state) * (1 - done) - self.critic(state)
            critic_loss = td_error.pow(2).mean()
            
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
import torch
import numpy as np
from itertools import count
from .Buffer import ReplayBuffer
from .NaiveDQN import NaiveDQN
from .Network import Network

class DQN(NaiveDQN):
    def __init__(self, env, alpha=0.001, gamma=0.95, episodes=100, max_epsilon=0.3, min_epsilon=0.05, epsilon_decay_rate=0.005, buffer_capacity=10000, batch_size=32, T_update_steps=10):
        super().__init__(env, alpha, gamma, episodes, max_epsilon, min_epsilon, epsilon_decay_rate)        
        self.buffer = ReplayBuffer(buffer_capacity)
        self.batch_size = batch_size
        self.T_Net = Network(self.n_state, self.n_action)   # Target Network
        self.T_update_steps = T_update_steps
        self.update_T()
            
    def update_T(self):
        self.T_Net.load_state_dict(self.Q_Net.state_dict())

    def update_Q(self):
        if len(self.buffer) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        # Creating a tensor from a list of numpy.ndarrays is extremely slow. 
        # Converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor.
        states      = torch.tensor(np.array(states),      dtype=torch.float32)  # (bs, 16)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)  # (bs, 16)
        rewards     = torch.tensor(rewards, dtype=torch.float32)                # (bs)
        actions     = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)     # (bs, 1)
        dones       = torch.tensor(dones,   dtype=torch.float32)                # (bs)
        
        q_values = self.Q_Net(states).gather(1, actions)    # (n_action, bs) -> (n_action, 1)
        next_q_values = self.T_Net(next_states).max(1)[0]   # (n_action, bs) -> (n_action)
        # next_q_values = self.Q_Net(next_states).max(1)[0]
        targets = rewards + (1 - dones) * self.gamma * next_q_values    # (n_action)
        targets = targets.unsqueeze(1)  # (n_action, 1)
        loss = self.criterion(q_values, targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train_an_episode(self, episode):        
        state, info = self.env.reset()
        for t in count():
            action = int(self.select_action(state))
            next_state, reward, done, info = self.env.step(action)
            if done and reward == 0:
                reward = -1
            self.buffer.push(self.one_hot_encode(state), action, reward, self.one_hot_encode(next_state), done)
            self.update_Q()
            state = next_state
            
            if done:
                self.update_epsilon(episode)
                if reward > 0:
                    print(f'{episode = }, {t = }, {self.epsilon = }')
                    self.save_Q()
                break
            
            if episode % self.T_update_steps == 0:
                self.update_T()  # Update the target network periodically
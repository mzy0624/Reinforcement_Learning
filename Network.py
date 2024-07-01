import torch
import torch.nn as nn

class Network(nn.Module):
    def __init__(self, n_state, n_action):
        super().__init__()
        self.fc1 = nn.Linear(n_state, 36)
        self.fc2 = nn.Linear(36, 36)
        self.fc3 = nn.Linear(36, n_action)
    
    def forward(self, state):
        x = state
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
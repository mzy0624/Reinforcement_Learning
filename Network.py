import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim,  hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class PolicyNetwork(Network):
    def __init__(self, state_dim, action_dim):
        super().__init__(state_dim, action_dim)
        
    def forward(self, x):
        return F.softmax(super().forward(x), dim=-1)
    
class ValueNetwork(Network):
    def __init__(self, state_dim):
        super().__init__(state_dim, 1)

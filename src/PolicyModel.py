import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyModel(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_units=[64, 32]):
        super(PolicyModel, self).__init__()
        layers = []

        for units in hidden_units:
            layers.append(nn.Linear(input_dim, units))
            layers.append(nn.LayerNorm(units))  
            layers.append(nn.ELU())  
            input_dim = units

        self.network = nn.Sequential(*layers)
        self.action_head = nn.Linear(input_dim, action_dim)  

    def forward(self, state):
        x = self.network(state)
        action_probs = torch.softmax(self.action_head(x), dim=-1)
        return action_probs

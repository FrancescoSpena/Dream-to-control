import torch
import torch.nn as nn

class PolicyModel(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_units=[128, 64, 32]):
        super(PolicyModel, self).__init__()
        layers = []
        
        # Construct hidden layers with LayerNorm, ELU activation, and Dropout
        for units in hidden_units:
            layers.append(nn.Linear(input_dim, units))  
            layers.append(nn.LayerNorm(units))  
            layers.append(nn.ELU())  
            layers.append(nn.Dropout(0.3))  
            input_dim = units
        
        self.network = nn.Sequential(*layers)
        self.action_head = nn.Linear(input_dim, action_dim)  

    def forward(self, state):
        x = self.network(state)  
        action_probs = torch.softmax(self.action_head(x), dim=-1)  
        return action_probs

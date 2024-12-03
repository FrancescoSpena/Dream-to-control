import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyModel(nn.Module):
    def __init__(self, input_dim=6, action_dim=3, hidden_units=[128, 128]):
        super(PolicyModel, self).__init__()
        layers = []

        # Create hidden layers with LayerNorm
        for units in hidden_units:
            layers.append(nn.Linear(input_dim, units))
            layers.append(nn.LayerNorm(units))  # Replace BatchNorm with LayerNorm
            layers.append(nn.ELU())  # Activation function
            input_dim = units

        self.network = nn.Sequential(*layers)
        self.action_head = nn.Linear(input_dim, action_dim)  # Output layer for actions

    def forward(self, state):
        x = self.network(state)
        action_probs = torch.softmax(self.action_head(x), dim=-1)
        return action_probs

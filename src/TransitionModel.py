import torch 
import torch.nn as nn 

class TransitionModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_units=[128,128], activation=nn.ReLU):
        super(TransitionModel, self).__init__()

        layers = []
        input_dim = state_dim + action_dim
        for units in hidden_units: 
            layers.append(nn.Linear(input_dim, units))
            layers.append(activation())
            input_dim = units
        
        layers.append(nn.Linear(input_dim, state_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, state, action):
        x = torch.cat([state,action], dim=-1)
        return self.net(x)


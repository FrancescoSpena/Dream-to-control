import torch
import torch.nn as nn

class RewardModel(nn.Module):
    def __init__(self, state_dim, hidden_units=[32, 32], activation=nn.ReLU):
        super(RewardModel, self).__init__()

        # Livelli nascosti
        layers = []
        input_dim = state_dim
        for units in hidden_units:
            layers.append(nn.Linear(input_dim, units))
            layers.append(activation())
            input_dim = units

        # Livello di output
        layers.append(nn.Linear(input_dim, 1))  # Ricompensa scalare
        
        self.network = nn.Sequential(*layers)

    def forward(self, state):
        return self.network(state)



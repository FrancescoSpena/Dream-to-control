import torch
import torch.nn as nn
import torch.nn.functional as F

class ValueModel(nn.Module):
    def __init__(self, input_dim, hidden_units=[512, 256, 128, 64]):
        super(ValueModel, self).__init__()
        layers = []
        for units in hidden_units:
            layers.append(nn.Linear(input_dim, units))
            layers.append(nn.ELU())
            input_dim = units
        self.network = nn.Sequential(*layers)
        self.output_layer = nn.Linear(input_dim, 1)

    def forward(self, state):
        x = self.network(state)
        value = self.output_layer(x)
        return value

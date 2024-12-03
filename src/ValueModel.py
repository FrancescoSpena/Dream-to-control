import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueModel(nn.Module):
    def __init__(self, input_dim, hidden_units=[300, 300, 300], activation=nn.ELU):
        super(ValueModel, self).__init__()
        self.layers = nn.ModuleList()
        self.residuals = nn.ModuleList()

        for units in hidden_units:
            self.layers.append(nn.Sequential(
                nn.Linear(input_dim, units),
                activation()
            ))
            self.residuals.append(nn.Linear(input_dim, units) if input_dim != units else None)
            input_dim = units

        self.output_layer = nn.Linear(input_dim, 1)

    def forward(self, state_action):
        x = state_action
        for layer, residual in zip(self.layers, self.residuals):
            x_residual = x
            x = layer(x)
            if residual is not None:
                x_residual = residual(x_residual)
            x += x_residual
        return self.output_layer(x)

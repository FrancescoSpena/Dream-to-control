import torch
import torch.nn as nn

class ValueModel(nn.Module):
    def __init__(self, input_dim, hidden_units=[256, 128, 64, 64, 32]):
        super(ValueModel, self).__init__()
        layers = []
        
        for units in hidden_units:
            layers.append(nn.Linear(input_dim, units))
            layers.append(nn.BatchNorm1d(units))  # Batch Normalization
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))  # Regularization
            input_dim = units

        self.network = nn.Sequential(*layers)
        self.output_layer = nn.Linear(input_dim, 1)

    def forward(self, state):
        x = self.network(state)
        value = self.output_layer(x)
        return value

import torch
import torch.nn as nn

class ValueModel(nn.Module):
    def __init__(self, input_dim, hidden_units=[256, 256, 128, 128], activation=nn.ReLU, dropout_rate=0.2):
        super(ValueModel, self).__init__()
        layers = []
        for units in hidden_units:
            layers.append(nn.Linear(input_dim, units))
            layers.append(nn.BatchNorm1d(units))  # Batch normalization per stabilità
            layers.append(activation())
            layers.append(nn.Dropout(dropout_rate))  # Dropout per regolarizzazione
            input_dim = units
        layers.append(nn.Linear(input_dim, 1))  # Output scalare
        self.network = nn.Sequential(*layers)

    def forward(self, state_action):
        return self.network(state_action)

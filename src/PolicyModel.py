import torch
import torch.nn as nn

class PolicyModel(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_units=[512, 512, 256, 256, 128]):
        super(PolicyModel, self).__init__()
        layers = []

        for units in hidden_units:
            layers.append(nn.Linear(input_dim, units))
            layers.append(nn.LayerNorm(units))  # Layer normalization
            layers.append(nn.ELU())
            layers.append(nn.Dropout(0.2))  # Regularization
            input_dim = units

        # Adding a residual connection for the last layer
        self.network = nn.Sequential(*layers)
        self.residual_layer = nn.Linear(hidden_units[-1], hidden_units[-1])
        self.action_head = nn.Linear(input_dim, action_dim)

    def forward(self, state):
        x = self.network(state)
        x = x + self.residual_layer(x)  # Residual connection
        action_probs = torch.softmax(self.action_head(x), dim=-1)
        return action_probs

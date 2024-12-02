import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyModel(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_units=[128, 128], activation=nn.ReLU, dropout_rate=0.2):
        super(PolicyModel, self).__init__()
        layers = []
        for units in hidden_units:
            layers.append(nn.Linear(input_dim, units))
            layers.append(activation())
            layers.append(nn.Dropout(dropout_rate))
            input_dim = units
        layers.append(nn.Linear(input_dim, action_dim))  
        self.network = nn.Sequential(*layers)

    def forward(self, state):
        logits = self.network(state)  # Ottiene i logit dalle reti neurali
        return F.softmax(logits, dim=-1)  # Applica softmax per generare le probabilità

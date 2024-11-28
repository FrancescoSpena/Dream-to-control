import torch
import torch.nn as nn

class RewardModel(nn.Module):
    def __init__(self, state_dim, hidden_units=[64, 64], activation=nn.ReLU):
        """
        Reward Model per l'ambiente Acrobot.
        Predice ricompense scalari per stati specifici.

        Args:
            state_dim (int): Dimensione del vettore di stato.
            hidden_units (list): Numero di neuroni per ogni livello nascosto.
            activation (torch.nn.Module): Funzione di attivazione da utilizzare nei livelli nascosti.
        """
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
        """
        Forward pass del Reward Model.

        Args:
            state (torch.Tensor): Tensor dello stato corrente (batch_size, state_dim).

        Returns:
            torch.Tensor: Ricompensa prevista (batch_size, 1).
        """
        return self.network(state)



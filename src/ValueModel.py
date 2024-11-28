import torch
import torch.nn as nn

class ValueModel(nn.Module):
    def __init__(self, input_dim, hidden_units=[1024, 512, 512, 256, 256, 128], activation=nn.LeakyReLU, dropout_rate=0.3):
        super(ValueModel, self).__init__()
        self.layers = nn.ModuleList()
        self.residuals = nn.ModuleList()

        original_input_dim = input_dim  # Salva la dimensione iniziale per la connessione globale

        for idx, units in enumerate(hidden_units):
            self.layers.append(nn.Sequential(
                nn.Linear(input_dim, units),
                nn.BatchNorm1d(units),
                activation(),
                nn.Dropout(dropout_rate)
            ))
            self.residuals.append(nn.Linear(input_dim, units) if input_dim != units else None)
            input_dim = units

        # Connessione globale
        self.global_residual = nn.Linear(original_input_dim, 1)

        # Livello di output scalare
        self.output_layer = nn.Sequential(
            nn.Linear(input_dim, 1)
        )

    def forward(self, state_action):
        """
        Args:
            state_action (torch.Tensor): Tensor dello stato (batch_size, input_dim).
        Returns:
            torch.Tensor: Valore predetto (batch_size, 1).
        """
        x = state_action
        for layer, residual in zip(self.layers, self.residuals):
            x_residual = x  # Salva il valore precedente
            x = layer(x)
            if residual is not None:
                x_residual = residual(x_residual)  # Allinea dimensioni
            x += x_residual  # Connessione residua

        # Connessione globale
        global_residual = self.global_residual(state_action)

        # Output finale
        return self.output_layer(x) + global_residual

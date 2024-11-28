import torch 
import torch.nn as nn 

class TransitionModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_units=[128,128], activation=nn.ReLU):
        """
        Modello di transizione per prevedere lo stato successivo dato lo stato attuale e l'azione.

        Args:
            state_dim (int): Dimensione del vettore di stato.
            action_dim (int): Dimensione del vettore di azione (one-hot encoded).
            hidden_units (list): Numero di neuroni per ogni livello nascosto.
            activation (torch.nn.Module): Funzione di attivazione da utilizzare nei livelli nascosti.
        """
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

#=====================================================
if __name__ == "__main__":
    state_dim = 6
    action_dim = 3

    transition_model = TransitionModel(state_dim=state_dim, action_dim=action_dim)

    batch_size = 4
    state = torch.rand(batch_size, state_dim)  
    action = torch.nn.functional.one_hot(torch.randint(0, action_dim, (batch_size,)), num_classes=action_dim).float()  # Azioni one-hot

    next_state = transition_model(state, action)
    print("Stato successivo previsto:", next_state)
#======================================================
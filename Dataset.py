import torch 
from torch.utils.data import Dataset
import numpy as np

class TransitionDataset(Dataset):
    def __init__(self, transitions):
        """
        Args:
            transitions (list): Lista di tuple (state, action, next_state).
        """
        # Converte in NumPy array per verificare che tutti i dati siano consistenti
        self.data = [
            (
                np.array(state, dtype=np.float32),
                np.array(action, dtype=np.float32),
                np.array(next_state, dtype=np.float32),
            )
            for state, action, next_state in transitions
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state, action, next_state = self.data[idx]
        return (
            torch.tensor(state, dtype=torch.float32),
            torch.tensor(action, dtype=torch.float32),
            torch.tensor(next_state, dtype=torch.float32),
        )

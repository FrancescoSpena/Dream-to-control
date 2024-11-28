import torch 
from torch.utils.data import Dataset
import numpy as np

class TransitionDataset(Dataset):
    def __init__(self, transitions):
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


class RewardDataset(torch.utils.data.Dataset):
    def __init__(self, transitions):
        self.data = [
            (torch.tensor(state, dtype=torch.float32), torch.tensor([reward], dtype=torch.float32))
            for state, reward in transitions
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
class ValueDataset(torch.utils.data.Dataset):
    def __init__(self, states, target_values):
        self.states = torch.tensor(states, dtype=torch.float32)
        self.target_values = torch.tensor(target_values, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.target_values[idx]
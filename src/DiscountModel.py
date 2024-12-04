import torch

class DiscountModel(torch.nn.Module):
    def __init__(self, input_dim):
        super(DiscountModel, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid()  
        )

    def forward(self, state):
        return self.network(state)

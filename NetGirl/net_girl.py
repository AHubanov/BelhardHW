import torch.nn as nn
import torch.nn.functional as F

class NetGirl(nn.Module):
    def __init__(self, input_dim, num_hidden, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, num_hidden)
        self.layer2 = nn.Linear(num_hidden, output_dim)

    def forward(self, X):
        X = self.layer1(X)
        X = F.tanh (X)
        X = self.layer2(X)
        X = F.tanh(X)
        return X

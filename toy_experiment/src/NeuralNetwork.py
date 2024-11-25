import torch.nn as nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self, activation:str, in_channels:int, out_channels:int, L: int = 1024, n_hidden:int = 1):
        assert activation in ["relu", "tanh"], "should be a a valid activation function"
        super().__init__()
        self.activation = activation
        self.L = L
        self.n_hidden = n_hidden
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.flatten = nn.Flatten()
        self.first_layer = nn.Linear(in_channels, L)
        self.hidden_layer = nn.Linear(L, L)
        self.last_layer = nn.Linear(L, out_channels)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        identity = x
        x = self.first_layer(x)
        for i in range(self.n_hidden):
            x = self.hidden_layer(x)
            if self.activation == "relu":
                x = self.relu(x)
            elif self.activation == "tanh":
                x = self.tanh(x)
        x = self.last_layer(x)
        return x
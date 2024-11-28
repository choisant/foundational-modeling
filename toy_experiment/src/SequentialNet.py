import torch.nn as nn

class SequentialNet(nn.Module):

    def __init__(self, activation:str, in_channels:int, out_channels:int, L: int = 1024, 
                 n_hidden:int = 1, p:float = 0):
        assert activation in ["relu", "tanh", "sigmoid"], "should be a a valid activation function"
        super().__init__()
        
        layers = []
        for i in range(n_hidden):
            layers.append(nn.Linear(L, L))
            if p > 0:
                layers.append(nn.Dropout(p))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "sigmoid":
                layers.append(nn.Sigmoid())
        self.hidden_layers = nn.Sequential(*layers)
        self.first_layer = nn.Linear(in_channels, L)
        self.last_layer = nn.Linear(L, out_channels)
        self.dropout = nn.Dropout(p) 

    def forward(self, x):
        x = self.first_layer(x)
        x = self.dropout(x)
        x = self.hidden_layers(x)
        x = self.last_layer(x)
        return x
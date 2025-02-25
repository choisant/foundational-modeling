import torchbnn as bnn
import torch.nn as nn

class BayesianSequentialNet(nn.Module):

    def __init__(self, activation:str, in_channels:int, out_channels:int, L: int = 1024, 
                 n_hidden:int = 1, p:float = 0, muprior=0, sigmaprior=0.01):
        assert activation in ["relu", "tanh", "sigmoid"], "should be a a valid activation function"
        super().__init__()
        
        layers = []
        for i in range(n_hidden):
            layers.append(bnn.BayesLinear(prior_mu=muprior, prior_sigma=sigmaprior, in_features=L, out_features=L))
            if p > 0:
                layers.append(nn.Dropout(p))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "sigmoid":
                layers.append(nn.Sigmoid())
        self.hidden_layers = nn.Sequential(*layers)
        self.first_layer = bnn.BayesLinear(prior_mu=muprior, prior_sigma=sigmaprior, in_features=in_channels, out_features=L)
        self.last_layer = bnn.BayesLinear(prior_mu=muprior, prior_sigma=sigmaprior, in_features=L, out_features=out_channels)
        self.dropout = nn.Dropout(p) 

    def forward(self, x):
        x = self.first_layer(x)
        x = self.dropout(x)
        x = self.hidden_layers(x)
        x = self.last_layer(x)
        return x
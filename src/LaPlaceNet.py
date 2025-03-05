import torch.nn as nn

class LaPlaceNet(nn.Module):
    
    def __init__(self, in_channels:int, out_channels:int, L: int = 1024):
        super().__init__()

        self.feature_map = nn.Sequential(
            nn.Linear(in_channels, L),
            nn.BatchNorm1d(L),
            nn.ReLU(), 
            nn.Linear(L, L), 
            nn.BatchNorm1d(L),
            nn.ReLU()
        )

        self.clf = nn.Linear(L, out_channels, bias=False)
    
    def forward(self, x):
        x = self.feature_map(x)
        return self.clf(x)
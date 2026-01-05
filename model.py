import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, inFeatures = 63, outFeatures = 2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(inFeatures, inFeatures*2),
            nn.Linear(inFeatures*2, inFeatures*4),
            nn.Linear(inFeatures*4, inFeatures*2),
            nn.Linear(inFeatures*2, inFeatures),
            nn.Linear(inFeatures, outFeatures),
        )
    def forward(self, x):
        return self.net(x)
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LinearHead(nn.Module):
    def __init__(self, net, dim_in=2048, dim_out=1000):
        super().__init__()
        self.net = net
        self.fc = nn.Linear(dim_in, dim_out)

        for param in self.net.parameters():
            param.requires_grad = False

        self.fc.weight.data.normal_(mean=0.0, std=0.01)
        self.fc.bias.data.zero_()

    def forward(self, x):
        with torch.no_grad():
            feat = self.net(x)
        return self.fc(feat)


class ProjectionHead(nn.Module):
    def __init__(self, dim_in=2048, hidden_dim=4096, dim_out=512):
        super().__init__()
        
        self.linear1 = nn.Linear(dim_in, hidden_dim)
        self.relu1 = nn.ReLU(True)
        self.linear2 = nn.Linear(hidden_dim, dim_out)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        return x

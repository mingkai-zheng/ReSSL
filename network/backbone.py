import torch.nn as nn
from network.resnet import *
from network.head import *


backbone_dict = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
}

dim_dict = {
    'resnet18': 512,
    'resnet34': 512,
    'resnet50': 2048,
}


class BackBone(nn.Module):
    def __init__(self, backbone='resnet50', hidden_dim=4096, dim=512):
        super().__init__()
        dim_in = dim_dict[backbone]
        self.net = backbone_dict[backbone]()
        self.head = ProjectionHead(dim_in=dim_in, hidden_dim=hidden_dim, dim_out=dim)
    
    def forward(self, x):
        feat = self.net(x)
        embedding = self.head(feat)
        return F.normalize(embedding)


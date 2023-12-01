import torch
from torch import nn
import torch.nn.functional as F


class TransE(nn.Module):
    def __init__(self, margin=1.0):
        super(TransE, self).__init__()
        self.margin = margin

    def forward(self, cui_0, cui_1, cui_2, re):
        pos = cui_0 + re - cui_1
        neg = cui_0 + re - cui_2
        return torch.mean(F.relu(self.margin + torch.norm(pos, p=2, dim=1) - torch.norm(neg, p=2, dim=1)))

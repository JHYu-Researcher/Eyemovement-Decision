import torch.nn.functional as F
from torch import nn

class DoubleLayerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(6, 10)
        self.linear2 = nn.Linear(10, 32)
        self.linear3 = nn.Linear(32, 4)

    def forward(self, input):
        x1 = F.leaky_relu(self.linear1(input))
        x2 = F.leaky_relu(self.linear2(x1))
        x3= self.linear3(x2)
        return x3


from config import Configs

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

config = Configs()

class LightCorrectionLayer(nn.Module):
    def __init__(self):
        super(LightCorrectionLayer, self).__init__()
        N = config.num_illu
        self.weights = Parameter(torch.ones(N), requires_grad=config.en_optim_light)

    def forward(self, E_in, idx):
        E_out = self.weights[idx] * E_in
        return E_out
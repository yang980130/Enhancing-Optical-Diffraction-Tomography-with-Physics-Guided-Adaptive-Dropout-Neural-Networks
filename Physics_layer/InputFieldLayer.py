from config import Configs

from math import pi
import torch
import torch.nn as nn

config = Configs()

class IncidentWavefrontLayer(nn.Module):
    def __init__(self):
        super(IncidentWavefrontLayer, self).__init__()
        N = config.slice_pad_size
        self.x = config.HR_pixel_size*torch.linspace(-N/2, N/2-1, N)
        self.yy, self.xx = torch.meshgrid(self.x, self.x)
        self.xx = self.xx.to(config.DEVICE)
        self.yy = self.yy.to(config.DEVICE)
    def forward(self, fxfy):
        output = torch.exp(1j*2*pi*(fxfy[0]*self.xx + fxfy[1]*self.yy))
        return output
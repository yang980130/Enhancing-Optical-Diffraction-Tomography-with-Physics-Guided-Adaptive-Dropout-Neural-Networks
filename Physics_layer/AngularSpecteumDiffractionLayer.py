from config import Configs

from math import pi
import torch
import torch.nn as nn

config = Configs()

class AngularSpecteumDiffractionLayer(nn.Module):
    def __init__(self):
        super(AngularSpecteumDiffractionLayer, self).__init__()
        self.M    = config.slice_pad_size
        psize     = config.HR_pixel_size
        t_k0      = config.k0 * config.n_media / (2*pi)
        kx          = torch.linspace(-pi/psize, pi/psize, self.M+1)[0:self.M] / (2*pi)
        [kxx, kyy]  = torch.meshgrid(kx, kx)
        kz_map_square = t_k0**2-kxx**2-kyy**2
        self.kz_map = self.sqrt__(kz_map_square).to(config.DEVICE)
        return
    
    def sqrt__(self, x):
        mask_1   = (x >= 0)
        mask_2   = (x <  0)
        x[mask_2] = -x[mask_2]
        positive_sqrt = torch.zeros([self.M, self.M], dtype=torch.complex64)
        negative_sqrt = torch.zeros([self.M, self.M], dtype=torch.complex64)
        positive_sqrt = torch.sqrt(x) * mask_1
        negative_sqrt = 1j * torch.sqrt(x) * mask_2
        output = positive_sqrt + negative_sqrt
        return output

    def forward(self, input, z):
        kzm    = self.kz_map
        output = input * torch.exp(1j*kzm*z)
        return output

    def cuda_init(self):
        self.kz_map = self.kz_map.to(config.DEVICE)
        self.mask   = self.mask  .to(config.DEVICE)


class MultiSliceAngularSpecteumDiffractionLayer(nn.Module):
    def __init__(self, wlength):
        super(MultiSliceAngularSpecteumDiffractionLayer, self).__init__()
        self.M     = config.slice_pad_size
        psize      = config.HR_pixel_size
        dfx        = 1/(self.M*psize)
        fx         = dfx*torch.linspace(-self.M/2, self.M/2, self.M+1)[0:self.M]
        [fyy, fxx] = (torch.meshgrid(fx, fx))
        fx         = torch.fft.ifftshift(fx)
        fxx        = torch.fft.ifftshift(fxx)
        fyy        = torch.fft.ifftshift(fyy)
        self.prop_phs   = 1j*2*pi*self.sqrt__((config.n_media/wlength)**2-(fxx**2+fyy**2)).to(config.DEVICE)
        self.prop_phs_x = (1j*2*pi*fxx/config.led_height).to(config.DEVICE)
        self.prop_phs_y = (1j*2*pi*fyy/config.led_height).to(config.DEVICE)
        return
    
    def sqrt__(self, x):
        mask_1   = (x >= 0)
        mask_2   = (x <  0)
        x[mask_2] = -x[mask_2]
        positive_sqrt = torch.zeros([self.M, self.M], dtype=torch.complex64)
        negative_sqrt = torch.zeros([self.M, self.M], dtype=torch.complex64)
        positive_sqrt = torch.sqrt(x) * mask_1
        negative_sqrt = 1j * torch.sqrt(x) * mask_2
        output = positive_sqrt + negative_sqrt
        return output

    def forward(self, input, z):
        if z >= 0:
            output = input * (torch.exp(self.prop_phs*z))
        else:
            output = input * torch.conj(torch.exp(self.prop_phs*(-z)))
        return output

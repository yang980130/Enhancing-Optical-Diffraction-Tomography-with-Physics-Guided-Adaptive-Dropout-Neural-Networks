from config import Configs

import torch
import torch.nn as nn
import torch.nn.functional as F

config = Configs()

class PupilLayer(nn.Module):
    def __init__(self):
        super(PupilLayer, self).__init__()
        M         = config.slice_size
        m         = config.CTFsize
        self.pad  = (M - m) // 2
        if self.pad * 2 == M - m:
            pad_ = self.pad
        else:
            pad_ = self.pad + 1
        pad      = self.pad
        xy                     = torch.linspace(-1, 1, m)
        self.x_map, self.y_map = torch.meshgrid(xy, xy)
        self.x_map    = self.x_map.to(config.DEVICE)
        self.y_map    = self.y_map.to(config.DEVICE)
        self.window   = torch.ones([m, m])
        circular_mask = torch.square(self.x_map) + torch.square(self.y_map) >= 1
        self.window[circular_mask] = 0
        self.window   = F.pad(self.window, (pad,pad_,pad,pad_)).to(config.DEVICE)

    def forward(self, input):
        output = input * self.window
        return output
    

class MultiSlicePupilLayer(nn.Module):
    def __init__(self, wlength):
        super(MultiSlicePupilLayer, self).__init__()
        self.M     = config.slice_pad_size
        psize      = config.HR_pixel_size
        dfx        = 1/(self.M*psize)
        fx         = dfx*torch.linspace(-self.M/2, self.M/2, self.M+1)[0:self.M]
        [fxx, fyy] = (torch.meshgrid(fx, fx))
        fx         = torch.fft.ifftshift(fx)
        fxx        = torch.fft.ifftshift(fxx)
        fyy        = torch.fft.ifftshift(fyy)
        NA_crop    = (fxx**2 + fyy**2) < (config.NA/wlength)**2
        self.NA_crop    = NA_crop.long().to(config.DEVICE)
        return 

    def forward(self, input):
        output = input*self.NA_crop
        return output


class MutiSliceCropPupilLayer(nn.Module):
    def __init__(self, wlength):
        super(MutiSliceCropPupilLayer, self).__init__()
        self.M     = config.slice_pad_size
        self.m     = config.capture_size
        psize      = config.HR_pixel_size
        dfx        = 1/(self.m*psize)
        fx         = dfx*torch.linspace(-self.m/2, self.m/2, self.m+1)[0:self.m]
        [fxx, fyy] = (torch.meshgrid(fx, fx))
        NA_crop    = (fxx**2 + fyy**2) < (config.NA/wlength)**2
        self.NA_crop    = (NA_crop.long() + torch.zeros([self.m, self.m], dtype=torch.complex64)).to(config.DEVICE)
        return 

    def forward(self, input):
        output = input[(config.slice_pad_size-config.capture_size)//2:(config.slice_pad_size+config.capture_size)//2, (config.slice_pad_size-config.capture_size)//2:(config.slice_pad_size+config.capture_size)//2]*self.NA_crop
        return output
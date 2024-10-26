from config import Configs
from tv_reg_torch import tv_reg_2d_grad

from math import pi
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.autograd import Function
from memory_profiler import profile
config = Configs()

    
class ReLU_non_negative(Function):
    @profile
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        
        return input.clamp(min=0.00)
        
        # return input
    @profile
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_output[input < 0.00] = 0
        return grad_output
    
class NonNegativeLayer(nn.Module):
    def __init__(self):
        super(NonNegativeLayer, self).__init__()
        self.relu_non_negative = ReLU_non_negative(inplace=False)
        
    def forward(self, input):
        output = self.relu_non_negative.apply(input)
        return output

class MyReLU_1(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0.003)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0.003] = 0
        return grad_input
    
class MyReLU_2(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        # input = tv_reg_2d_grad(input)
        return input.clamp(min=0.0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # input = tv_reg_2d_grad(input)
        # grad_input[input < 0.015] = 0
        grad_input[input < 0.015] = 0
        grad_input = tv_reg_2d_grad(grad_input)
        return grad_input

class MyReLU_3(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0.024)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0.024] = 0
        return grad_input
    
class AttentionLayer(nn.Module):
    def __init__(self):
        super(AttentionLayer, self).__init__()
        self.attention_weight = Parameter(torch.Tensor(3), requires_grad=False)
        # self.attention_weight = Parameter(torch.Tensor(1), requires_grad=False)
        self.relu_1 = MyReLU_1()
        self.relu_2 = MyReLU_2()
        self.relu_3 = MyReLU_3()
        
    def forward(self, input):
        return self.relu_2(input)

class SliceLayer(nn.Module):
    def __init__(self):
        super(SliceLayer, self).__init__()
        M = config.slice_pad_size
        self.weights   = Parameter(torch.Tensor(M, M), requires_grad=config.en_optim_object)
        self.mask      = Parameter(torch.Tensor(M, M), requires_grad=False)
        self.coeff     = Parameter(torch.Tensor(1), requires_grad=config.en_optim_object)
        # self.non_negative = NonNegativeLayer()
        self.non_negative = ReLU_non_negative()
        return
    # @profile
    def forward(self, input, z, wlength, epoch):
        slice = self.weights
        if config.SCATTER_MODE == 'MS' or config.SCATTER_MODE =='1st born':
            if config.non_negative_layer:
                output = input * torch.exp(1j*2*pi*(self.non_negative.apply(slice))*z/wlength)
            else:
                output = input * torch.exp(1j*2*pi*(slice)*z/wlength)
        elif config.SCATTER_MODE == 'MS_LIAC':
            if config.non_negative_layer:
                output = self.coeff[0] * input * torch.exp(1j*2*pi*(self.non_negative.apply(slice))*z/wlength)
            else:
                output = input * torch.exp(1j*2*pi*slice*z/wlength)
                ####### !!!!!就是这里搞错了，加上去了#######
                # output = input * torch.exp(1j*2*pi*self.non_negative.apply(slice)*z/wlength) 
                ####### ------------------------ #######
        else:
            print('The scatter mode setting is wrong!!')
            raise
        weights = self.weights[config.pad_size: -config.pad_size, config.pad_size: -config.pad_size]
        return output, self.weights


class ScatterLayer(nn.Module):
    def __init__(self):
        super(ScatterLayer, self).__init__()
        self.M = config.slice_pad_size
        psize = config.HR_pixel_size
        dfx        = 1/(self.M*psize)
        fx         = dfx*torch.linspace(-self.M/2, self.M/2, self.M+1)[0:self.M]
        [fxx, fyy] = (torch.meshgrid(fx, fx))
        self.coeff = Parameter(torch.ones(config.slice_num_list[-1]), requires_grad=True).to(config.DEVICE)
        fxx        = torch.fft.ifftshift(fxx)
        fyy        = torch.fft.ifftshift(fyy)
        dis = self.sqrt__((config.n_media/config.wlength)**2-torch.sqrt((fxx**2+fyy**2))).to(config.DEVICE)

        self.prop_phs = 1j*2*pi*self.sqrt__((config.n_media/config.wlength)**2-(fxx**2+fyy**2)).to(config.DEVICE)
        self.Fgreen = (-1j * torch.exp(self.prop_phs*config.imaging_depth/config.slice_num_list[-1]) / (4*pi*dis)).to(config.DEVICE)
        
        self.non_negative = ReLU_non_negative()

    def forward(self, Uin, n_, z, epoch):
        if config.non_negative_layer and epoch < 60:
            n_ = self.non_negative.apply(n_)
        V = (2*pi/config.wlength)**2*(-n_)*(n_+2*config.n_media) # 这个在1st born/MLB那里是正确的
        # V = (2*pi/config.wlength)**2*(n_)*(-n_+2*config.n_media)
        return torch.fft.ifft2(self.Fgreen * torch.fft.fft2(Uin*V)) * z

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
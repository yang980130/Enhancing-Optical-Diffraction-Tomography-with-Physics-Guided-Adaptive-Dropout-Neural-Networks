from config import Configs
from Base_part.Object import PhaseObject3D

import torch
import numpy as np
from matplotlib import pyplot as plt

config = Configs()

def tv_reg_3d(object, iter=10, t=0.1, ep=1, lamda=0.5):
    tv_object = PhaseObject3D(config.phase_obj_shape, config.pad_size, config.n_media, config.n_min, config.n_max)
    tv_object.RI_pad_cuda = object.RI_pad_cuda
    
    im = object.RI_cuda
    imtmp = im
    for i in range(iter):
        ux  = torch.roll(imtmp, 1, dims=1)[:, 1:-1, 1:-1] - torch.roll(imtmp, -1, dims=1)[:, 1:-1, 1:-1]
        uz  = torch.roll(imtmp, 1, dims=2)[:, 1:-1, 1:-1] - torch.roll(imtmp, -1, dims=2)[:, 1:-1, 1:-1]
        uxx = torch.roll(imtmp, 1, dims=1)[:, 1:-1, 1:-1] + torch.roll(imtmp, -1, dims=1)[:, 1:-1, 1:-1] - 2*imtmp[:, 1:-1, 1:-1]
        uzz = torch.roll(imtmp, 1, dims=2)[:, 1:-1, 1:-1] + torch.roll(imtmp, -1, dims=2)[:, 1:-1, 1:-1] - 2*imtmp[:, 1:-1, 1:-1]
        uxz = torch.roll(torch.roll(imtmp, 1, dims=2), 1, dims=1)[:, 1:-1, 1:-1] + torch.roll(torch.roll(imtmp, -1, dims=2),-1, dims=1)[:, 1:-1, 1:-1] 
        -     torch.roll(torch.roll(imtmp, 1, dims=2),-1, dims=1)[:, 1:-1, 1:-1] - torch.roll(torch.roll(imtmp, -1, dims=2), 1, dims=1)[:, 1:-1, 1:-1]
        divp = ((uz * uz + ep) * uxx - 2* ux * uz * uxz + (ux * ux + ep) * uzz) / ((ux * ux + uz * uz + ep))
        imtmp[:, 1:-1, 1:-1] += t *( lamda * (im[:, 1:-1, 1:-1] - imtmp[:, 1:-1, 1:-1]) + divp)

        uz  = torch.roll(imtmp, 1, dims=2)[1:-1, :, 1:-1] - torch.roll(imtmp, -1, dims=2)[1:-1, :, 1:-1]
        uy  = torch.roll(imtmp, 1, dims=0)[1:-1, :, 1:-1] - torch.roll(imtmp, -1, dims=0)[1:-1, :, 1:-1]
        uzz = torch.roll(imtmp, 1, dims=2)[1:-1, :, 1:-1] + torch.roll(imtmp, -1, dims=2)[1:-1, :, 1:-1] - 2*imtmp[1:-1, :, 1:-1]
        uyy = torch.roll(imtmp, 1, dims=0)[1:-1, :, 1:-1] + torch.roll(imtmp, -1, dims=0)[1:-1, :, 1:-1] - 2*imtmp[1:-1, :, 1:-1]
        uzy = torch.roll(torch.roll(imtmp, 1, dims=0), 1, dims=2)[1:-1, :, 1:-1] + torch.roll(torch.roll(imtmp, -1, dims=0),-1, dims=2)[1:-1, :, 1:-1] 
        -     torch.roll(torch.roll(imtmp, 1, dims=0),-1, dims=2)[1:-1, :, 1:-1] - torch.roll(torch.roll(imtmp, -1, dims=0), 1, dims=2)[1:-1, :, 1:-1]
        divp = ((uy * uy + ep) * uzz - 2* uz * uy * uzy + (uz * uz + ep) * uyy) / ((uz * uz + uy * uy + ep))
        imtmp[1:-1, :, 1:-1] += t *( lamda * (im[1:-1, :, 1:-1] - imtmp[1:-1, :, 1:-1]) + divp)

    mask = object.RI_cuda > 0
    tv_object.RI_pad_cuda[config.pad_size+1:-config.pad_size-1, config.pad_size+1:-config.pad_size-1, :] = imtmp[1:-1, 1:-1, :]
    # tv_object.RI_pad_cuda[mask] = imtmp[mask]
    
    return tv_object

def tv_reg_2d(object, iter=10, t=0.1, ep=1, lamda=0.5):
    tv_object = PhaseObject3D(config.phase_obj_shape, config.pad_size, config.n_media, config.n_min, config.n_max)
    tv_object.RI_pad_cuda = object.RI_pad_cuda
    
    im = object.RI_cuda
    imtmp = im
    for i in range(iter):
        ux  = torch.roll(imtmp, 1, dims=1)[1:-1, 1:-1, :] - torch.roll(imtmp, -1, dims=1)[1:-1, 1:-1, :]
        uy  = torch.roll(imtmp, 1, dims=0)[1:-1:,1:-1, :] - torch.roll(imtmp, -1, dims=0)[1:-1:,1:-1, :]
        uxx = torch.roll(imtmp, 1, dims=1)[1:-1, 1:-1, :] + torch.roll(imtmp, -1, dims=1)[1:-1, 1:-1, :] - 2*imtmp[1:-1, 1:-1, :]
        uyy = torch.roll(imtmp, 1, dims=0)[1:-1, 1:-1, :] + torch.roll(imtmp, -1, dims=0)[1:-1, 1:-1, :] - 2*imtmp[1:-1, 1:-1, :]
        uxy = torch.roll(torch.roll(imtmp, 1, dims=0), 1, dims=1)[1:-1, 1:-1, :] + torch.roll(torch.roll(imtmp, -1, dims=0),-1, dims=1)[1:-1, 1:-1, :] 
        -     torch.roll(torch.roll(imtmp, 1, dims=0),-1, dims=1)[1:-1, 1:-1, :] - torch.roll(torch.roll(imtmp, -1, dims=0), 1, dims=1)[1:-1, 1:-1, :]
        divp = ((uy * uy + ep) * uxx - 2* ux * uy * uxy + (ux * ux + ep) * uyy) / ((ux * ux + uy * uy + ep))
        imtmp[1:-1, 1:-1, :] += t *(lamda * (imtmp[1:-1, 1:-1, :] - imtmp[1:-1, 1:-1, :]) + divp)

    tv_object.RI_pad_cuda[config.pad_size+1:-config.pad_size-1, config.pad_size+1:-config.pad_size-1, :] = imtmp[1:-1, 1:-1, :]
    return tv_object

def tv_reg_2d_grad(tensor, iter=10, t=0.1, ep=1, lamda=0.5):
    im = tensor
    imtmp = im
    for i in range(iter):
        ux  = torch.roll(imtmp, 1, dims=1) - torch.roll(imtmp, -1, dims=1)
        uy  = torch.roll(imtmp, 1, dims=0) - torch.roll(imtmp, -1, dims=0)
        uxx = torch.roll(imtmp, 1, dims=1) + torch.roll(imtmp, -1, dims=1) - 2*imtmp
        uyy = torch.roll(imtmp, 1, dims=0) + torch.roll(imtmp, -1, dims=0) - 2*imtmp
        uxy = torch.roll(torch.roll(imtmp, 1, dims=0), 1, dims=1) + torch.roll(torch.roll(imtmp, -1, dims=0),-1, dims=1) 
        -     torch.roll(torch.roll(imtmp, 1, dims=0),-1, dims=1) - torch.roll(torch.roll(imtmp, -1, dims=0), 1, dims=1)
        divp = ((uy * uy + ep) * uxx - 2* ux * uy * uxy + (ux * ux + ep) * uyy) / ((ux * ux + uy * uy + ep))
        imtmp += t *(lamda * (imtmp - imtmp) + divp)
    output = imtmp
    return output

    
def tv3d_loss(model):
    param = torch.Tensor(config.slice_size, config.slice_size, config.slice_num[-1])
    i = 0
    for slice_name, slice in model.SliceLayer_dic.items():
        param[i] = slice.weights.data
        i += 1
    param_dx = torch.abs(torch.roll(param, shifts=1, dims=0) - param)
    param_dy = torch.abs(torch.roll(param, shifts=1, dims=1) - param)
    param_dz = torch.abs(torch.roll(param, shifts=1, dims=2) - param)
    param_dx = torch.sum(param_dx, dim=0)
    param_dy = torch.sum(param_dy, dim=1)
    param_dz = torch.sum(param_dz, dim=2)
    param_dx_norm = torch.norm(param_dx, p=2)
    param_dy_norm = torch.norm(param_dy, p=2)
    param_dz_norm = torch.norm(param_dz, p=2)
    tv_regularization_loss = param_dx_norm + param_dy_norm + param_dz_norm
    return tv_regularization_loss
    

def LowPassFilter(img):
    """
    理想低通滤波器
    """
    fshift = torch.fft.fftshift(torch.fft.fft2(img)).to('cpu')
    # 设置低通滤波器
    rows, cols, slice = img.shape
    crow, ccol, cslice = int(rows / 2), int(cols / 2), int(slice / 2)  # 中心位置
    mask = np.zeros((rows, cols, slice), np.uint8)
    mask[crow - 20:crow + 20, ccol - 20:ccol + 20, cslice - 20:cslice + 20] = 1

    # 掩膜图像和频谱图像乘积
    f = fshift * mask

    # 傅里叶逆变换
    ishift = torch.fft.ifftshift(f)
    iimg = torch.fft.ifft2(ishift)
    # 逆傅里叶变换回来后又有虚部了
    return iimg


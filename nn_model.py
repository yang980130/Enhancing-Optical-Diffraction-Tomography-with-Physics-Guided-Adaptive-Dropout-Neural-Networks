from config import Configs
from Base_part.Object import PhaseObject3D
from Physics_layer.InputFieldLayer import IncidentWavefrontLayer
from Physics_layer.SliceLayer_AttentionLayer import SliceLayer, ScatterLayer
from Physics_layer.AngularSpecteumDiffractionLayer import MultiSliceAngularSpecteumDiffractionLayer
from Physics_layer.PupilLayer import MultiSlicePupilLayer
from CorrectionLayer import LightCorrectionLayer

import torch
import torch.nn as nn

config = Configs()    

class MultiplyLayer(nn.Module):
    def __init__(self):
        super(MultiplyLayer, self).__init__()


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        h_x = x.size()[0]
        w_x = x.size()[1]
        z_x = x.size()[2]
        count_h = self._tensor_size(x[1:,:,:])
        count_w = self._tensor_size(x[:,1:,:])
        count_z = self._tensor_size(x[:,:,1:])
        h_tv = torch.pow((x[1:,:,:]-x[:h_x-1,:,:]),2).sum()
        w_tv = torch.pow((x[:,1:,:]-x[:,:w_x-1,:]),2).sum()
        z_tv = torch.pow((x[:,:,1:]-x[:,:,:z_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w+z_tv/count_z)

    def _tensor_size(self, t):
        return t.size()[0]*t.size()[1]*t.size()[2]


class MultiSliceNN(nn.Module):
    def __init__(self):
        super(MultiSliceNN, self).__init__()
        config  = Configs()
        self.slice_number = config.slice_num
        # self.Dropout = nn.Dropout(0.1)
        self.Inputwave = IncidentWavefrontLayer()
        self.SliceLayer_dic = {}
        self.AttentionLayer_dic = {}
        for i in range(self.slice_number):
            self.SliceLayer_dic['Slice_{}'.format(i)] = SliceLayer()
        self.Prop      = MultiSliceAngularSpecteumDiffractionLayer(config.wlength)
        self.BS        = ScatterLayer()
        self.Pupil     = MultiSlicePupilLayer(config.wlength)
        self.LightCor  = LightCorrectionLayer()
        self.weights   = torch.zeros(config.slice_pad_size, config.slice_pad_size, config.slice_num_list[-1])
        return

    
    def forward(self, input, illu_th, epoch):
        Fourier_E1 = torch.fft.fft2(self.Inputwave(input))
        Fourier_scatter_field = 0
        idx = 0
        for slice_layer in self.SliceLayer_dic.items():

            slice = slice_layer[1] # [0]为layer name
            
            E1_ = torch.fft.ifft2(self.Prop(Fourier_E1, config.imaging_depth/self.slice_number))
            
            if config.SCATTER_MODE == '1st born': # 1st born
                _, weights = slice(E1_, config.imaging_depth/self.slice_number, config.wlength, epoch)
                scatter_field = self.BS(torch.fft.ifft2(Fourier_E1), weights, config.imaging_depth/self.slice_number, epoch)
                Fourier_scatter_field += self.Prop(torch.fft.fft2(scatter_field), ((self.slice_number-1)/2-idx)*config.imaging_depth/self.slice_number)
                E1 = E1_

            elif config.SCATTER_MODE == 'MS':
                E1, weights = slice(E1_, config.imaging_depth/self.slice_number, config.wlength, epoch)

            self.weights[:,:,idx] = weights            
            idx += 1
            Fourier_E1 = torch.fft.fft2(E1)

        Fourier_E1 = self.Prop(Fourier_E1, -config.imaging_depth/2) # experiment simulate -8.125 if total 16.25
        Fourier_E1 += Fourier_scatter_field # 1st born
        Fourier_E1 = self.Pupil(Fourier_E1)
        E1 = torch.fft.ifft2(Fourier_E1)
        intensity = self.LightCor(torch.abs(E1), illu_th)
        # intensity = intensity**2 #
        if config.pad_size == 0:
            return intensity, self.weights
        else:
            return intensity[config.pad_size:-config.pad_size, config.pad_size:-config.pad_size], self.weights
            # return intensity, self.weights


    def initModel(self, phaseobj_3d, mode='init'):
        i = 0
        coeff = len(self.SliceLayer_dic) // phaseobj_3d.RI_pad.shape[2]
        for slice_name, slice in self.SliceLayer_dic.items():
            if mode == 'init':
                idx = i // coeff
            elif mode == 'gt':
                idx = i
            else:
                raise('The param -- init mode is wrong!')
            slice.weights.data = torch.tensor(phaseobj_3d.RI_pad_cuda[..., idx]).type(config.torch_float).to(config.DEVICE)
            slice.coeff.data = torch.ones(1).to(config.DEVICE)
            slice.mask.data = torch.ones([config.slice_pad_size, config.slice_pad_size]).type(config.torch_complex).to(config.DEVICE)
            i += 1
        # self.attention.attention_weight.data = torch.tensor([0.1, 0.5, 0.5], dtype=torch.float32).to(config.DEVICE)
        # self.BS.weights.data = torch.ones(config.slice_num_list[-1]).to(config.DEVICE)
        

    def extractParameters2cpu(self):
        object_3d = PhaseObject3D(config.phase_obj_shape, config.pad_size, config.n_media, config.n_min, config.n_max)
        slice_dic = {}
        for slice_name, slice in self.SliceLayer_dic.items():
            slice_dic[slice_name] = slice.weights.data.to('cpu').detach().numpy() * slice.mask.data.to('cpu').detach().numpy()
        object_3d.createPhaseObject3DfromArrayDic(slice_dic)
        return object_3d

    def extractParameters2cuda(self):
        object_3d = PhaseObject3D(config.phase_obj_shape, config.pad_size, config.n_media, config.n_min, config.n_max)
        slice_dic = {}
        for slice_name, slice in self.SliceLayer_dic.items():
                slice_dic[slice_name] = slice.weights.data
        object_3d.createPhaseObject3DfromTensorDic(slice_dic)
        return object_3d
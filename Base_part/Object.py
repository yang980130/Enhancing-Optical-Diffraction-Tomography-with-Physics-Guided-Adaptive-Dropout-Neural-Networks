from config import Configs

from math import pi
import numpy as np
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image
import SimpleITK as sitk
import scipy.io as io
import h5py

config = Configs()

# np_complex_datatype = config.np_complex_datatype
# np_float_datatype   = config.np_float_datatype


class Slice():
    def __init__(self, size):
        super(Slice, self).__init__()
        self.shape  = [size, size]
        self.distribution = np.zeros(self.shape)                              

    def createSimulateObject(self, path):
        self.distribution = cv2.resize(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY), self.shape, interpolation=cv2.INTER_CUBIC).astype(config.np_float)


class PhaseObject3D():
    def __init__(self, shape, pad, n_media, n_min, n_max):
        assert len(shape) == 3
        self.shape       = shape
        self.pad         = pad
        self.n_media     = n_media
        self.n_min       = n_min
        self.n_max       = n_max
        self.RI          = np   .ones( shape, dtype = config.np_float)
        self.RI_pad      = np   .ones((shape[0]+2*pad, shape[1]+2*pad, shape[2]), dtype = config.np_float)
        self.RI_cuda     = torch.ones( shape, dtype=config.torch_float).to(config.DEVICE)
        self.RI_pad_cuda = torch.ones((shape[0]+2*pad, shape[1]+2*pad, shape[2]), dtype=config.torch_float).to(config.DEVICE)
    
    def createPhaseObject3DfromTensorDic(self, tensor_dic): # 由3DTensor对PhaseObject做初始化，发生在优化过程中，tv正则化后
        i = 0
        for slice_name, slice in tensor_dic.items():
            if slice.shape[0] == config.slice_size:
                self.RI_cuda[..., i] = slice
                self.RI_pad_cuda = self.__pad_torch(self.RI_cuda, self.pad)
            elif slice.shape[0] == config.slice_pad_size:
                self.RI_pad_cuda[..., i] = slice
                self.RI_cuda = self.RI_pad_cuda[self.pad:-self.pad, self.pad:-self.pad]
            else:
                raise('The shape of slice_dic error!')
            i += 1
        if self.pad != 0:
            self.RI_cuda = self.RI_pad_cuda[self.pad:-self.pad, self.pad:-self.pad]
        else:
            self.RI_cuda = self.RI_pad_cuda
        self.__tensor2arrayRI()
    
    def createPhaseObject3DfromArrayDic(self, array_dic): # 由3DArray对PhaseObject做初始化，发生在优化发生前
        i = 0
        for slice_name, slice in array_dic.items():
            if slice.shape[0] == config.slice_size:
                self.RI[..., i] = slice
                self.RI_pad = self.__pad(self.RI, self.pad)
            elif slice.shape[0] == config.slice_pad_size:
                self.RI_pad[..., i] = slice
                self.RI = self.RI_pad[self.pad:-self.pad, self.pad:-self.pad]
            else:
                raise('The shape of slice_dic error!')
            i += 1
        self.__array2tensorRI()

    def createPhaseObject3Dfrom3DArray(self, array_3d):
        RI = array_3d
        RI_max = np.max(RI)
        RI_min = np.min(RI)
        RI = (RI-RI_min)/(RI_max-RI_min)
        if RI.shape[0] == config.slice_size:
            self.RI = (self.n_max-self.n_media) * (RI-1) + self.n_max - self.n_media
            self.RI_pad = self.__pad(self.RI, self.pad)
        elif RI.shape[0] == config.slice_pad_size:
            self.RI_pad = (self.n_max-self.n_media) * (RI-1) + self.n_max - self.n_media
            self.RI = self.RI_pad[self.pad:-self.pad, self.pad:-self.pad]
        self.__array2tensorRI()

    def zeroInitPhaseObject3D(self):
        for i in range(self.RI.shape[2]):
            self.RI[..., i] = np.zeros([config.slice_size, config.slice_size])
            self.RI_cuda[..., i] = torch.zeros([config.slice_size, config.slice_size])
            self.RI_pad[..., i] = np.zeros([config.slice_pad_size, config.slice_pad_size])
            self.RI_pad_cuda[..., i] = torch.zeros([config.slice_pad_size, config.slice_pad_size])
        # self.RI = np.zeros([config.slice_size, config.slice_size, config.slice_num_list[-1]], dtype = config.np_complex)
        # self.RI_pad = np.zeros([config.slice_pad_size, config.slice_pad_size, config.slice_num_list[-1]], dtype = config.np_complex)
        # self.__array2tensorRI()

        
    def __array2tensorRI(self):
        self.RI_cuda = torch.tensor(self.RI).to(config.DEVICE)
        self.RI_pad_cuda = torch.tensor(self.RI_pad).to(config.DEVICE)
    
    def __tensor2arrayRI(self):
        self.RI = self.RI_cuda.cpu().detach().numpy()
        self.RI_pad = self.RI_pad_cuda.cpu().detach().numpy()

    def __pad(self, obj, pad_size):
        return np.pad(obj, ((pad_size, pad_size), (pad_size, pad_size),  (0,0)))
    
    def __pad_(self, obj, pad_size):
        return np.pad(obj, ((pad_size, pad_size), (pad_size, pad_size)))

    def __pad_torch(self, obj, pad_size):
        return F.pad(obj, (pad_size, pad_size, pad_size, pad_size, 0, 0))
    
    def saveObjectAsNpy(self):
        np.save('./3d_sample.npy', self.RI)

    def showObject(self):
        plt.figure()
        plt.subplot(231),  plt.imshow(self.RI[:,:,round(self.RI.shape[2]//2 - 4)], cmap='gray', vmin=config.plot_range[0], vmax=config.plot_range[1]), plt.title('xy mid-4')
        plt.subplot(232),  plt.imshow(self.RI[:,:,round(self.RI.shape[2]//2 - 2)], cmap='gray', vmin=config.plot_range[0], vmax=config.plot_range[1]), plt.title('xy mid-2')
        plt.subplot(233),  plt.imshow(self.RI[:,:,round(self.RI.shape[2]//2 + 0)], cmap='gray', vmin=config.plot_range[0], vmax=config.plot_range[1]), plt.title('xy mid')
        plt.subplot(234),  plt.imshow(self.RI[:,:,round(self.RI.shape[2]//2 + 2)], cmap='gray', vmin=config.plot_range[0], vmax=config.plot_range[1]), plt.title('xy mid+2')
        plt.subplot(235),  plt.imshow(self.RI[:,:,round(self.RI.shape[2]//2 + 4)], cmap='gray', vmin=config.plot_range[0], vmax=config.plot_range[1]), plt.title('xy mid+4')
        plt.subplot(236),  plt.imshow(self.RI[:,:,round(self.RI.shape[2]//2 + 6)], cmap='gray', vmin=config.plot_range[0], vmax=config.plot_range[1]), plt.title('xy mid+6')
        plt.figure()
        plt.subplot(231),  plt.imshow(self.RI[:,round(self.RI.shape[1]//2 - 4),:], cmap='gray', vmin=config.plot_range[0], vmax=config.plot_range[1]), plt.title('xz mid-4')
        plt.subplot(232),  plt.imshow(self.RI[:,round(self.RI.shape[1]//2 - 2),:], cmap='gray', vmin=config.plot_range[0], vmax=config.plot_range[1]), plt.title('xz mid-2')
        plt.subplot(233),  plt.imshow(self.RI[:,round(self.RI.shape[1]//2 + 0),:], cmap='gray', vmin=config.plot_range[0], vmax=config.plot_range[1]), plt.title('xz mid')
        plt.subplot(234),  plt.imshow(self.RI[:,round(self.RI.shape[1]//2 + 2),:], cmap='gray', vmin=config.plot_range[0], vmax=config.plot_range[1]), plt.title('xz mid+2')
        plt.subplot(235),  plt.imshow(self.RI[:,round(self.RI.shape[1]//2 + 4),:], cmap='gray', vmin=config.plot_range[0], vmax=config.plot_range[1]), plt.title('xz mid+4')
        plt.subplot(236),  plt.imshow(self.RI[:,round(self.RI.shape[1]//2 + 6),:], cmap='gray', vmin=config.plot_range[0], vmax=config.plot_range[1]), plt.title('xz mid+6')
        plt.figure()
        plt.subplot(231),  plt.imshow(self.RI[round(self.RI.shape[0]//2 - 4),:,:], cmap='gray', vmin=config.plot_range[0], vmax=config.plot_range[1]), plt.title('yz mid-4')
        plt.subplot(232),  plt.imshow(self.RI[round(self.RI.shape[0]//2 - 2),:,:], cmap='gray', vmin=config.plot_range[0], vmax=config.plot_range[1]), plt.title('yz mid-2')
        plt.subplot(233),  plt.imshow(self.RI[round(self.RI.shape[0]//2 + 0),:,:], cmap='gray', vmin=config.plot_range[0], vmax=config.plot_range[1]), plt.title('yz mid')
        plt.subplot(234),  plt.imshow(self.RI[round(self.RI.shape[0]//2 + 2),:,:], cmap='gray', vmin=config.plot_range[0], vmax=config.plot_range[1]), plt.title('yz mid+2')
        plt.subplot(235),  plt.imshow(self.RI[round(self.RI.shape[0]//2 + 4),:,:], cmap='gray', vmin=config.plot_range[0], vmax=config.plot_range[1]), plt.title('yz mid+4')
        plt.subplot(236),  plt.imshow(self.RI[round(self.RI.shape[0]//2 + 6),:,:], cmap='gray', vmin=config.plot_range[0], vmax=config.plot_range[1]), plt.title('yz mid+6')

        # plt.subplot(235),  plt.imshow(cv2.resize(self.RI[:,145,:], (120, self.RI.shape[0])), cmap='gray', vmin=config.plot_range[0], vmax=config.plot_range[1]), plt.title('xz mid') # self.RI[:,145,:] waller 线虫 360 pixel_num
        # plt.subplot(235),  plt.imshow(cv2.resize(self.RI.shape[1]//2, (round(config.imaging_depth/config.HR_pixel_size), self.RI.shape[0])), cmap='gray', vmin=config.plot_range[0], vmax=config.plot_range[1]), plt.title('xz mid') # 将z轴显示尺寸拉伸到实际比例，显示中间层
        
        # plt.subplot(236),  plt.imshow(cv2.resize(self.RI[90,:,:],  (120, self.RI.shape[0])), cmap='gray', vmin=config.plot_range[0], vmax=config.plot_range[1]), plt.title('yz mid') # self.RI[90,:,:] waller 线虫 360 pixel_num
        # plt.subplot(236),  plt.imshow(cv2.resize(self.RI.shape[0]//2, (round(config.imaging_depth/config.HR_pixel_size), self.RI.shape[0])), cmap='gray', vmin=config.plot_range[0], vmax=config.plot_range[1]), plt.title('xz mid') # 将z轴显示尺寸拉伸到实际比例，显示中间层

        # real = np.real(self.RI)
        # imag = np.imag(self.RI)
        # plt.subplot(231),  plt.imshow(real[:,:,round(real.shape[2]/2 - 2)], cmap='gray', vmin=config.plot_range[0], vmax=config.plot_range[1]), plt.title('real pixel_xy')
        # plt.subplot(232),  plt.imshow(real[:,:,round(real.shape[2]/2 + 0)], cmap='gray', vmin=config.plot_range[0], vmax=config.plot_range[1]), plt.title('real pixel_xy')
        # plt.subplot(233),  plt.imshow(real[:,:,round(real.shape[2]/2 + 2)], cmap='gray', vmin=config.plot_range[0], vmax=config.plot_range[1]), plt.title('real pixel_xy')
        # plt.subplot(234),  plt.imshow(real[:,:,round(real.shape[2]/2 + 4)], cmap='gray', vmin=config.plot_range[0], vmax=config.plot_range[1]), plt.title('real pixel_xy')
        # plt.subplot(235),  plt.imshow(real[:,round(real.shape[1]/2),:], cmap='gray', vmin=config.plot_range[0], vmax=config.plot_range[1]), plt.title('real pixel_xz')
        # plt.subplot(236),  plt.imshow(real[round(real.shape[0]/2),:,:], cmap='gray', vmin=config.plot_range[0], vmax=config.plot_range[1]), plt.title('real pixel_yz')
        # plt.figure()
        # plt.subplot(231),  plt.imshow(imag[:,:,round(imag.shape[2]/2 - 2)], cmap='gray', vmin=-0.01, vmax=0.03), plt.title('imag pixel_xy')
        # plt.subplot(232),  plt.imshow(imag[:,:,round(imag.shape[2]/2 + 0)], cmap='gray', vmin=-0.01, vmax=0.03), plt.title('imag pixel_xy')
        # plt.subplot(233),  plt.imshow(imag[:,:,round(imag.shape[2]/2 + 2)], cmap='gray', vmin=-0.01, vmax=0.03), plt.title('imag pixel_xy')
        # plt.subplot(234),  plt.imshow(imag[:,:,round(imag.shape[2]/2 + 4)], cmap='gray', vmin=-0.01, vmax=0.03), plt.title('imag pixel_xy')
        # plt.subplot(235),  plt.imshow(imag[:,round(imag.shape[1]/2),:], cmap='gray', vmin=-0.01, vmax=0.03), plt.title('imag pixel_xz')
        # plt.subplot(236),  plt.imshow(imag[round(imag.shape[0]/2),:,:], cmap='gray', vmin=-0.01, vmax=0.03), plt.title('imag pixel_yz')

    # def saveOptimObject(self, epoch, loss_list): # experiment
    def saveOptimObject(self, epoch, loss_list, lossgt_list): # simulation
        root_path = './optim_process'
        path_3d = os.path.join(root_path, '3d_results')
        
        if   config.SCATTER_MODE == 'MS':
            if config.tv:
                path_3d = os.path.join(path_3d, 'MS-tv')
            else:
                path_3d = os.path.join(path_3d, 'MS')
        elif config.SCATTER_MODE == 'MS_LIAC':
            if config.tv:
                path_3d = os.path.join(path_3d, 'MS_LIAC-tv')
            else:
                path_3d = os.path.join(path_3d, 'MS_LIAC')
        elif config.SCATTER_MODE == '1st born':
            if config.tv:
                path_3d = os.path.join(path_3d, '1st born-tv')
            else:
                path_3d = os.path.join(path_3d, '1st born')
        
        path_fold = os.path.join(path_3d,'{}'.format(config.savelabel))
        if os.path.exists(path_fold) == False:
            os.mkdir(path_fold)
        path_3d = path_fold
        
        if os.path.exists('epoch{}'.format(epoch)) == False:
            os.mkdir(os.path.join(path_3d, 'epoch{}'.format(epoch)))
        path_3d = os.path.join(path_3d, 'epoch{}'.format(epoch))

        # self.RI = - self.RI

        for i in range(config.slice_num_list[-1]):
            slice = self.RI[:, :, i]
            slice_im = (slice-(config.plot_range[0]))/(config.plot_range[1]-(config.plot_range[0])) * 255
            # min = np.min(slice_im)
            # max = np.max(slice_im)
            # im =  (slice_im-min)/(max-min) * 255
            # im = Image.fromarray(np.uint8(slice_im))
            # im = Image.fromarray(np.uint16(slice_im))
            im = Image.fromarray(slice_im)
            im = im.convert('L')
            im.save(os.path.join(path_3d, 'xy_{}.jpg'.format(i)))

        for i in range(config.slice_size):
            slice = self.RI[:, i, :]
            slice_im = (slice-(config.plot_range[0]))/(config.plot_range[1]-(config.plot_range[0])) * 255
            im = Image.fromarray(slice_im)
            im = im.convert('L')
            im.save(os.path.join(path_3d, 'xz_{}.jpg'.format(i)))

        for i in range(config.slice_size):
            slice = self.RI[i, :, :]
            slice_im = (slice-(config.plot_range[0]))/(config.plot_range[1]-(config.plot_range[0])) * 255
            im = Image.fromarray(slice_im)
            im = im.convert('L')
            im.save(os.path.join(path_3d, 'yz_{}.jpg'.format(i)))

        # np.save(os.path.join(path_3d,'RI.npy'),self.RI)
        # np_array = np.load(os.path.join(path_3d,'RI.npy'))
        # io.savemat(os.path.join(path_3d,'RI.mat'),{'data':np_array})
        # io.savemat(os.path.join(path_3d,'RI.mat'),{'data', self.RI})

        io.savemat(os.path.join(path_3d,'loss.mat'),{'loss':loss_list}) # simulation
        io.savemat(os.path.join(path_3d,'lossgt.mat'),{'lossgt':lossgt_list}) # simulation
        

    def saveObject(self, loss_list=0):
        # self.showObject()
        root_path = './experiment_results'
        path_2d = os.path.join(root_path, '2d_results')
        path_3d = os.path.join(root_path, '3d_results')

        if   config.SCATTER_MODE == 'MS':
            if config.tv:
                path_2d = os.path.join(path_2d, 'MS-tv')
                path_3d = os.path.join(path_3d, 'MS-tv')
            else:
                path_2d = os.path.join(path_2d, 'MS')
                path_3d = os.path.join(path_3d, 'MS')
        elif config.SCATTER_MODE == 'MS_LIAC':
            if config.tv:
                path_2d = os.path.join(path_2d, 'MS_LIAC-tv')
                path_3d = os.path.join(path_3d, 'MS_LIAC-tv')
            else:
                path_2d = os.path.join(path_2d, 'MS_LIAC')
                path_3d = os.path.join(path_3d, 'MS_LIAC')
        elif config.SCATTER_MODE == '1st born':
            if config.tv:
                path_2d = os.path.join(path_2d, '1st born-tv')
                path_3d = os.path.join(path_3d, '1st born-tv')
            else:
                path_2d = os.path.join(path_2d, '1st born')
                path_3d = os.path.join(path_3d, '1st born')

        plt.savefig(os.path.join(path_2d,'epoch{}_{}_lr{}_led{}_imaging_depth{}_tv{}-{}-{}_plt_range[{},{}].jpg'.format(config.num_epoch,config.savelabel,config.learning_rate, config.ILLU_MODE, config.imaging_depth, config.tv_beta[0], config.tv_beta[1], config.tv_beta[2], config.plot_range[0], config.plot_range[1])))
        
        if os.path.exists(os.path.join(path_3d,'epoch{}_{}_lr{}_led{}_imaging_depth{}_tv{}-{}-{}_plt_range[{},{}]'.format(config.num_epoch,config.savelabel,config.learning_rate, config.ILLU_MODE, config.imaging_depth, config.tv_beta[0], config.tv_beta[1], config.tv_beta[2], config.plot_range[0], config.plot_range[1]))) == False:
            os.mkdir(os.path.join(path_3d, 'epoch{}_{}_lr{}_led{}_imaging_depth{}_tv{}-{}-{}_plt_range[{},{}]'.format(config.num_epoch,config.savelabel,config.learning_rate, config.ILLU_MODE, config.imaging_depth, config.tv_beta[0], config.tv_beta[1], config.tv_beta[2], config.plot_range[0], config.plot_range[1])))
        path_3d = os.path.join(path_3d, 'epoch{}_{}_lr{}_led{}_imaging_depth{}_tv{}-{}-{}_plt_range[{},{}]'.format(config.num_epoch,config.savelabel,config.learning_rate, config.ILLU_MODE, config.imaging_depth, config.tv_beta[0], config.tv_beta[1], config.tv_beta[2], config.plot_range[0], config.plot_range[1]))
        
        for i in range(config.slice_num_list[-1]):
            slice = self.RI[:, :, i]
            slice_im = (slice-(config.plot_range[0]))/(config.plot_range[1]-(config.plot_range[0])) * 255
            im = Image.fromarray(slice_im)
            im = im.convert('L')
            im.save(os.path.join(path_3d, 'xy_{}.jpg'.format(i)))

        for i in range(config.slice_size):
            # slice = cv2.resize(self.RI[:, i, :], (self.RI.shape[2]*2, self.RI.shape[0]))
            slice = self.RI[:, i, :]
            slice_im = (slice-(config.plot_range[0]))/(config.plot_range[1]-(config.plot_range[0])) * 255
            im = Image.fromarray(slice_im)
            # im = Image.fromarray(np.uint8(slice_im))
            im = im.convert('L')
            im.save(os.path.join(path_3d, 'xz_{}.jpg'.format(i)))

        for i in range(config.slice_size):
            # slice = cv2.resize(self.RI[i, :, :], (self.RI.shape[2]*2, self.RI.shape[0]))
            slice = self.RI[i, :, :]
            slice_im = (slice-(config.plot_range[0]))/(config.plot_range[1]-(config.plot_range[0])) * 255
            im = Image.fromarray(slice_im)
            # im = Image.fromarray(np.uint8(slice_im))
            im = im.convert('L')
            im.save(os.path.join(path_3d, 'yz_{}.jpg'.format(i)))

    def saveGTobject(self):
        path_3d = './tomography_GT_object'
        for i in range(config.slice_num_list[-1]):
            slice = self.RI[:, :, i]
            slice_im = (slice-(config.plot_range[0]))/(config.plot_range[1]-(config.plot_range[0])) * 255
            im = Image.fromarray(slice_im)
            im = im.convert('L')
            im.save(os.path.join(path_3d, 'xy_{}.jpg'.format(i)))

        for i in range(config.slice_size):
            # slice = cv2.resize(self.RI[:, i, :], (self.RI.shape[2]*2, self.RI.shape[0]))
            slice = self.RI[:, i, :]
            slice_im = (slice-(config.plot_range[0]))/(config.plot_range[1]-(config.plot_range[0])) * 255
            im = Image.fromarray(slice_im)
            # im = Image.fromarray(np.uint8(slice_im))
            im = im.convert('L')
            im.save(os.path.join(path_3d, 'xz_{}.jpg'.format(i)))

        for i in range(config.slice_size):
            # slice = cv2.resize(self.RI[i, :, :], (self.RI.shape[2]*2, self.RI.shape[0]))
            slice = self.RI[i, :, :]
            slice_im = (slice-(config.plot_range[0]))/(config.plot_range[1]-(config.plot_range[0])) * 255
            im = Image.fromarray(slice_im)
            # im = Image.fromarray(np.uint8(slice_im))
            im = im.convert('L')
            im.save(os.path.join(path_3d, 'yz_{}.jpg'.format(i)))



    def npy2mat():
        np_array = np.load('./3d_sample_1000epoch_.npy')
        io.savemat('./3d_sample_1000epoch_.mat',{'data':np_array})

        # for i in range(config.slice_num_list[-1]):
        #     slice_real = np.real(self.RI[:, :, i])
        #     slice_im_real = (slice_real-(config.plot_range[0]))/(config.plot_range[1]-(config.plot_range[0])) * 255
        #     im = Image.fromarray(slice_im_real)
        #     im = im.convert('L')
        #     im.save(os.path.join(path_3d, 'real-xy_{}.jpg'.format(i)))

        #     slice_imag = np.imag(self.RI[:, :, i])
        #     slice_im_imag = (slice_imag-(config.plot_range[0]))/(config.plot_range[1]-(config.plot_range[0])) * 255
        #     im = Image.fromarray(slice_im_imag)
        #     im = im.convert('L')
        #     im.save(os.path.join(path_3d, 'imag-xy_{}.jpg'.format(i)))

        ### 保存loss_list ###
        # filename = open(os.path.join(path_3d, 'epoch{}_lr{}.jpg'.format(config.num_epoch,config.learning_rate)), 'w')
        # for value in loss_list:
        #     filename.write(str(value)+'\n')
        # filename.close()
        ### ------------ ###
        # plt.show()


    def non_negative_regularization(self):

        # mask = self.RI_cuda < 0.01
        # self.RI_cuda[mask] = 0
        # mask = self.RI_pad_cuda < 0.01
        # self.RI_pad_cuda[mask] = 0

        # mask = self.RI_cuda < 0.004
        # self.RI_cuda[mask] = 0
        # mask = self.RI_pad_cuda < 0.004
        # self.RI_pad_cuda[mask] = 0

        mask = self.RI_cuda < 0
        self.RI_cuda[mask] = 1e-9
        mask = self.RI_pad_cuda < 0
        self.RI_pad_cuda[mask] = 1e-9


    def analys_RI(self):
        h, w, c = self.RI.shape[:3]
        numberBins = 300
        RI = (self.RI)/(config.n_max-config.n_media) * numberBins
        pixelSequence = RI.reshape([h*w*c, ])
        plt.figure()
        histogram, bins, patch = plt.hist(pixelSequence, numberBins, facecolor='gray', histtype='bar')
        plt.xlabel('RI')
        plt.ylabel('number of pixels')
        plt.axis([0, numberBins, 0, np.max(histogram)])
        # plt.show()

import numpy as np
import torch
import torch.nn.functional as F
from math import pi
import cv2
from PIL import Image

import os
import matplotlib.pyplot as plt
import SimpleITK as sitk
import scipy.io as io
import h5py

from config import Configs

config = Configs()


class Measurement():
    def __init__(self, num_measurement):
        self.size = num_measurement
        self.show_measurement = list([x,x+1,x+2] for x in range(self.size))        
        self.in_measurement   = list([x,x+1,x+2] for x in range(self.size))  
        self.exposure_coeff   = np.ones(config.num_illu)

    def in2show_transform(self):
        for i in range(self.size):
            self.show_measurement[i] = self.in_measurement[i].to('cpu').detach().numpy()

    def show2in_transform(self):
        for i in range(self.size):
            self.in_measurement[i] = torch.tensor(self.show_measurement[i]).to(config.DEVICE)
    

    def readCEMeasurement(self):
        img = sitk.ReadImage(config.data_path)
        # img = sitk.GetArrayFromImage(img) **0.5
        img = sitk.GetArrayFromImage(img) #**0.5
        img = img.astype(np.float32)
        # img = img / np.mean(img)
        # img = np.resize(img, (img.shape[0], config.slice_size, config.slice_size))
        for i in range(img.shape[0]):
            if i < len(self.show_measurement):
                self.show_measurement[i] = img[i] # total FOV
                # img[i] = img[i] / np.mean(img[i])
                # self.show_measurement[i] = img[i, 1318:1518, 1662:1862]
                # self.show_measurement[i] = img[i, 560:920, 0:360] # Phary
                # self.show_measurement[i] = img[i, 360:720, 780:1140] # mouth

    def readMatMeasurement(self): # 李加基数据
        # path = './IRaw_Diatom_I.mat'
        # path = './IRaw_Diatom_II.mat'
        # path = './IRaw_Cheek.mat'
        # path = './IRaw_CElegans.mat'
        # data = io.loadmat(path)
        path = './Simulation_Granlocyte_phantom.mat'
        # path = './Simulation_Yanny.mat'
        data = h5py.File(path)
        data_ = data['I_Raw'][:]# + 1 # 用MSNN重构IDT数据，要整体+1（补低频）
        # data_ = data_ + 1
        # data_ = data_ ** 0.5
        for i in range(data['I_Raw'].shape[2]):
            if i < len(self.show_measurement):
                self.show_measurement[i] = data_[i,:,:] # 对于DeCAF的数据，这样操作
                # self.show_measurement[i] = data_[:,:,i]
        return

    def readMicrossphereMeasurement(self):
        # path = './LED_algae_data/ROI_uint16/stack.tif'
        # path = './8um_deltan0.03_NA0.75/Stack_2.5umdefocus.tif'
        # path = './8um_deltan0.03_NA0.75/Stack.tif'
        # path = './8um_deltan0.03_NA0.75/4microsphere_Stack.tif'
        # path = './8um_deltan0.03_NA0.75/Stack_imsize200单球.tif'
        path = './Stack.tif'
        darkfield = np.load('./暗场灯序号.npy')
        brightfield = np.load('./明场灯序号.npy')
        edgefield = np.load('边界灯序号.npy')
        coeff = np.load('./全部灯照射均值.npy')
        # coeff = np.load('./全部灯照射均值_椭圆.npy')
        img = sitk.ReadImage(path)
        # img = sitk.GetArrayFromImage(img)**0.5
        img = sitk.GetArrayFromImage(img)
        # img = (img / np.mean(img))**0.5
        # img = img - np.mean(img)
        # img += 1
        img = img.astype(np.float32)
        for i in range(img.shape[0]):
            if i < len(self.show_measurement):
                # self.show_measurement[i] = img[i]
                self.show_measurement[i] = img[i]/np.mean(img[i])
                # self.show_measurement[i] = img[i]/np.mean(img[i]) * coeff[i]
        return

    def save_measurement(self):

        for i in range(config.num_illu):
            im = self.show_measurement[i]
            min = np.min(im)
            max = np.max(im)
            min, max = 0.2, 1.8
            im_show =  (im-min)/(max-min) * 255
            im_show = Image.fromarray(im_show)
            im_show = im_show.convert('L')

            im_show.save('predict/show_measurement/{}.jpg'.format(i))
            cv2.imwrite( 'predict/in_mesurement/{}.tif'.format(i), im)
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

class Illumination():
    def __init__(self):
        super(Illumination, self).__init__()
        self.loc            = []
        self.kxky           = []
        self.fxfy           = []
        self.numxy_center   = []
        self.NA_illu = []
        self.NA_seq  = []
    
    def createLedLocation(self):
        N       = config.arraysize # N代表LED矩阵中一条灯带的灯数
        gap     = config.led_gap
        delta_x = config.x_offset
        delta_y = config.y_offset
        theta   = config.theta_offset

        xlocation = np.zeros([N ** 2])
        ylocation = np.zeros([N ** 2])
        for i in range(1, N+1):
            xlocation[N*(i-1):N+N*(i-1)] = np.arange(-(N-1)/2, N/2, 1)*gap
            ylocation[N*(i-1):N+N*(i-1)] = ((N-1)/2-(i-1))*gap
        x_location = (xlocation * np.cos(theta) + ylocation * np.sin(theta) + delta_x)
        y_location = (xlocation * np.sin(theta) - ylocation * np.cos(theta) + delta_y)

        return x_location, y_location
    
    def generateIlluminationSequence(self):
        N = config.arraysize
        n = (N + 1)/2
        sequence = np.zeros([2, N**2])
        sequence[0, 0] = n
        sequence[1, 0] = n
        dx = 1
        dy = -1
        stepx = 1
        stepy = -1
        direction = 1
        counter = 0
        i = 1
        while i < N**2:
            counter += 1
            if direction == 1:
                sequence[0, i] = sequence[0, i-1] + dx
                sequence[1, i] = sequence[1, i-1]
                if counter == abs(stepx):
                    counter = 0
                    direction = direction * -1
                    dx = dx * -1
                    stepx = stepx * -1
                    if stepx > 0:
                        stepx += 1
                    else:
                        stepx -= 1
            else:
                sequence[0, i] = sequence[0, i-1]
                sequence[1, i] = sequence[1, i-1] + dy
                if counter == abs(stepy):
                    counter = 0
                    direction = direction * -1
                    dy = dy * -1
                    stepy = stepy * -1
                    if stepy > 0:
                        stepy += 1
                    else:
                        stepy -= 1
            i += 1
        illumination_sequence = (sequence[0, :] - 1) * N + sequence[1, :]
        return illumination_sequence

    def createIlluminationAngle(self):
        if config.ILLU_MODE == 'array':
            N    = config.num_illu
            h    = config.led_height
            k0   = config.k0
            x, y = self.createLedLocation()
            seq  = self.generateIlluminationSequence()

            ###以下为矩形照明的伪kxky(fxfy)###
            # kx_relative = - np.sin(np.arctan(x / h))
            # ky_relative = - np.sin(np.arctan(y / h))
            # kx = k0 * ky_relative
            # ky = k0 * kx_relative
            # fx = kx / (2*pi)
            # fy = ky / (2*pi)
            # test = np.zeros([N,2])
            ###------------------###

            ###以下为椭圆照明的正确kxky(fxfy),按照空间位置，左上为起点的顺序###
            rlocation = np.sqrt(x**2 + y**2 + config.led_height**2)
            kx_relative = x / rlocation
            ky_relative = y / rlocation
            kx = - k0 * ky_relative
            ky = - k0 * kx_relative
            fx = kx / (2*pi)
            fy = ky / (2*pi)
            test = np.zeros([N,2])
            ###--------------------###

            ###以下为FPM式的中心向外扩散的顺序###
            # for idx, value in enumerate(seq):
            #     value = int(value) - 1
            #     self.loc.append([x[value], y[value]])
            #     self.kxky.append([-kx[value], ky[value]])
            #     self.fxfy.append([fx[value], fy[value]])
            #     test[idx,0] = fx[value]
            #     test[idx,1] = fy[value]
            ###--------------------###

            ### 以下为按照NA排序的照明初始化 ###
            kx_relative_ = np.zeros(config.using_illu)
            ky_relative_ = np.zeros(config.using_illu)
            x_ = np.zeros(config.using_illu)
            y_ = np.zeros(config.using_illu)
            kx_ = np.zeros(config.using_illu)
            ky_ = np.zeros(config.using_illu)
            fx_ = np.zeros(config.using_illu)
            fy_ = np.zeros(config.using_illu)
            for idx, value in enumerate(seq):
                value = int(value) - 1
                kx_relative_[idx] = kx_relative[value]
                ky_relative_[idx] = ky_relative[value]
                x_[idx] = x[value]
                y_[idx] = y[value]
                kx_[idx] = kx[value]
                ky_[idx] = ky[value]
                fx_[idx] = fx[value]
                fy_[idx] = fy[value]
     
            NA_illu = np.sqrt((ky_relative_**2 + kx_relative_**2))
            NA_seq = np.argsort(NA_illu + np.array([n for n in range(config.using_illu)]) * 1e-7)
            self.NA_illu = NA_illu
            self.NA_seq = NA_seq

            for idx, value in enumerate(NA_seq):
                value = int(value)
                self.loc.append([x_[value], y_[value]])
                self.kxky.append([kx_[value], ky_[value]])
                self.fxfy.append([fx_[value], fy_[value]])
                test[idx,0] = fx_[value]
                test[idx,1] = fy_[value]
            ### -------------------------- ###

        # plt.figure()
        # plt.plot(test[:,0], test[:,1], 'o:')
        # plt.show()

        kmax = pi / config.HR_pixel_size
        dkxy = 2 * kmax / config.capture_size
        return
    
    def generateSpiralPath(self, revs):
        N = config.num_illu
        t = np.linspace(0, 2*pi, N)
        x = t * np.cos(revs*t)/(2*pi)
        y = t * np.sin(revs*t)/(2*pi)
        dist =  np.zeros(N)
        for i in range(1, N):
            dist[i] = np.sqrt((x[i]-x[i-1])**2 + (y[i]-y[i-1])**2) + dist[i-1]
        coef = np.mean(t[N-1]**2/dist[N-1])
        tInc = np.sqrt(np.linspace(0, dist[N-1], N+1)*coef)
        tInc = tInc[0:N]
        x_ = tInc*np.cos(revs*tInc)/(2*pi)
        y_ = tInc*np.sin(revs*tInc)/(2*pi)
        x = config.NA/config.wlength * x_
        y = config.NA/config.wlength * y_
        # x = config.NA/config.wlength * x_ /2
        # y = config.NA/config.wlength * y_ /2
        test = np.zeros([N,2])
        for i in range(N):
            self.kxky.append([2*pi*y[i], 2*pi*x[i]])
            self.fxfy.append([y[i], x[i]])
            test[i,0] = y[i]
            test[i,1] = x[i]
        plt.figure()
        plt.plot(test[:,0], test[:,1], 'o:')
        plt.show()
        self.createLocfromRead()
        return 0
    
    def readkxky(self):
        param = io.loadmat(config.param_path)
        test = np.zeros([config.num_illu,2])
        for i in range(config.num_illu):
            self.kxky.append([-2*pi*param['kx'][0,i], -2*pi*param['ky'][0,i]])
            self.fxfy.append([-param['kx'][0,i], -param['ky'][0,i]])
            # self.fxfy.append([param['kx'][0,i], param['ky'][0,i]])
            test[i,0] = -param['kx'][0,i]
            test[i,1] = -param['ky'][0,i]
        # plt.figure()
        # plt.plot(test[:,0], test[:,1], 'o:'), plt.title('illumination angle')
        # plt.xlabel('fx')
        # plt.ylabel('fy')
        # plt.show()
        return

    def readArraykxky(self):
        feature = h5py.File('./Parameters.mat')  #读取mat文件
        kx = feature['kx'][0:config.num_illu]
        ky = feature['ky'][0:config.num_illu]
        test = np.zeros([config.num_illu,2])
        for i in range(config.num_illu):
            if i < kx.shape[0]:
                self.kxky.append([kx[i,0], ky[i,0]])
                self.fxfy.append([kx[i,0]/(2*pi*1e6), ky[i,0]/(2*pi*1e6)])
                test[i,0] = self.fxfy[i][0]
                test[i,1] = self.fxfy[i][1]
        plt.figure()
        plt.plot(test[:,0], test[:,1], 'o:'), plt.title('illumination angle')
        plt.xlabel('fx')
        plt.ylabel('fy')
        plt.show()
        return
    
    def createAnnular(self):
        test = np.zeros([config.num_illu,2])
        # fxfy_ =     [[0,1.54567071737045e-16], 
        #              [-1.21912968366581, 0.326664814207065],
        #             [-1.09304177176677, 0.631067961165049],
        #             [-0.892464869458749, 0.892464869458750],
        #             [-0.631067961165049, 1.09304177176677],
        #             [-0.326664814207065,	1.21912968366581],
        #             [7.72835358685223e-17,	1.26213592233010],
        #             [0.326664814207065,	1.21912968366581],
        #             [0.631067961165049,	1.09304177176677],
        #             [0.892464869458750,	0.892464869458749],
        #             [1.09304177176677,	0.631067961165049],
        #             [1.21912968366581,	0.326664814207065],
        #             [1.26213592233010,	0],
        #             [1.21912968366581,	-0.326664814207065],
        #             [1.09304177176677,	-0.631067961165049],
        #             [0.892464869458750,	-0.892464869458749],
        #             [0.631067961165049,	-1.09304177176677],
        #             [0.326664814207065,	-1.21912968366581],
        #             [7.72835358685223e-17, -1.26213592233010],
        #             [-0.326664814207065, -1.21912968366581],
        #             [-0.631067961165049, -1.09304177176677],
        #             [-0.892464869458749, -0.892464869458750],
        #             [-1.09304177176677,	-0.631067961165049],
        #             [-1.21912968366581,	-0.326664814207065]]	
        fxfy_ =     [[0,	-1.26213592233010],
                    [-0.326664814207065,	-1.21912968366581],
                    [-0.631067961165049,	-1.09304177176677],
                    [-0.892464869458750,	-0.892464869458749],
                    [-1.09304177176677,	-0.631067961165049],
                    [-1.21912968366581,	-0.326664814207065],
                    [-1.26213592233010,	0],
                    [-1.21912968366581,	0.326664814207065],
                    [-1.09304177176677,	0.631067961165048],
                    [-0.892464869458749,	0.892464869458749],
                    [-0.631067961165049,	1.09304177176677],
                    [-0.326664814207065,	1.21912968366581],
                    [0,	1.26213592233010],
                    [0.326664814207065,	1.21912968366581],
                    [0.631067961165049,	1.09304177176677],
                    [0.892464869458749,	0.892464869458749],
                    [1.09304177176677,	0.631067961165048],
                    [1.21912968366581,	0.326664814207065],
                    [1.26213592233010,	0],
                    [1.21912968366581,	-0.326664814207065],
                    [1.09304177176677,	-0.631067961165049],
                    [0.892464869458749,	-0.892464869458749],
                    [0.631067961165049,	-1.09304177176677],
                    [0.326664814207065,	-1.21912968366581]]												
        self.kxky = fxfy_ #* 2*pi
        self.loc  = fxfy_
        for i in range(config.num_illu):
            self.fxfy.append([fxfy_[i][0], fxfy_[i][1]])
            # self.fxfy.append([-fxfy_[i][1], -fxfy_[i][0]])
            test[i,0] = self.fxfy[i][0]
            test[i,1] = self.fxfy[i][1]
        plt.figure()
        plt.plot(test[:,0], test[:,1], 'o:'), plt.title('illumination angle')
        plt.xlabel('fx')
        plt.ylabel('fy')
        plt.show()
        return
    
    def createAnnular2(self):
        t = np.linspace(0, 2*pi, config.num_illu+1)
        t = t[0:config.num_illu]
        x = np.cos(t)
        y = np.sin(t)
        fx =  x / np.sqrt(1 + (1/np.tan(np.arcsin(config.NA*0.98))**2))
        fy =  y / np.sqrt(1 + (1/np.tan(np.arcsin(config.NA*0.98))**2))
        kx = - config.k0 * fx
        ky =   config.k0 * fy
        # center_x = np.round(ky/dkxy+M/2)
        # center_y = np.round(kx/dkxy+M/2)
        test = np.zeros([config.num_illu,2])
        for i in range(config.num_illu):
            # annular满足match illumination loc未知
            self.fxfy[:,i] = [fx[i], fy[i]]
            # self.kxky[:,i] = [kx[i], ky[i]]
            # self.center_xy[:,i] = [center_x[i], center_y[i]]
            test[i,0] = self.fxfy[i][0]
            test[i,1] = self.fxfy[i][1]
        plt.figure()
        plt.plot(test[:,0], test[:,1], 'o:'), plt.title('illumination angle')
        plt.xlabel('fx')
        plt.ylabel('fy')
        plt.show()
        return

    def createLocfromRead(self):
        wlength = config.wlength
        for i in range(config.num_illu):
            if i < len(self.kxky):
                loc_x = np.tan(np.arcsin(wlength * self.fxfy[i][0])) * config.led_height
                loc_y = np.tan(np.arcsin(wlength * self.fxfy[i][1])) * config.led_height
                self.loc.append([loc_x, loc_y])
            else:
                print('The number of capture img is out of the num_illu!')
        return
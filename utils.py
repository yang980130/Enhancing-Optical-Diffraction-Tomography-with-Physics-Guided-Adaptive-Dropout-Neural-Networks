import numpy as np
import torch
import matplotlib.pyplot as plt
# from PIL import Image
import scipy
import scipy.io
import cv2

from config import np_float_datatype
from config import Configs

from Base_part.illumination import Illumination

config = Configs()


def cuda2np(x):
    return x.to('cpu').detach().numpy()

def generate_sphere(img_hw, img_z, radius, cx, cy, cz, label): # radius, cx, cy, cz单位为0.1um

    x_pixel  = np.array(list(range(img_hw))).reshape([img_hw,1,1])
    y_pixel  = np.array(list(range(img_hw))).reshape([1,img_hw,1])
    z_pixel  = np.array(list(range(img_z))).reshape([1,1,img_z])

    coeff_xy = 10*config.HR_pixel_size*img_hw/img_hw
    coeff_z  = 10*config.imaging_depth/img_z
    x_actual = np.array(list(range(round(coeff_xy*img_hw)))).reshape([round(coeff_xy*img_hw),1,1])
    y_actual = np.array(list(range(round(coeff_xy*img_hw)))).reshape([1,round(coeff_xy*img_hw),1])
    z_actual = np.array(list(range(round(coeff_z*img_z)))).reshape([1,1,round(coeff_z*img_z)])

    # if label == 1:
        # mask_actual = ((x_actual-cx)**2+(y_actual-cy)**2+(z_actual-cz)**2<=radius**2)
        # mask_pixel  = ((coeff_xy*x_pixel-cx)**2+(coeff_xy*y_pixel-cy)**2+(coeff_z*z_pixel-cz)**2<=radius**2)
    mask_actual = ( label**2*((x_actual-cx)**2+(y_actual-cy)**2) + (z_actual-cz)**2 <=radius**2)
    mask_pixel  = ( label**2*((coeff_xy*x_pixel-cx)**2+(coeff_xy*y_pixel-cy)**2) + (coeff_z*z_pixel-cz)**2<=radius**2)

    sphere_actual = np.ones([round(coeff_xy*img_hw), round(coeff_xy*img_hw), round(coeff_z*img_z)])
    sphere_pixel  = np.ones([img_hw, img_hw, img_z])
    return mask_pixel * sphere_pixel, mask_actual * sphere_actual


def generate_sphere_sample(cz_list, show=True, img_xy_size=0, img_z_size=0):
    sphere_num = len(cz_list)
    if img_xy_size == 0 or img_z_size == 0:
        img_xy_size = config.slice_size
        img_z_size  = config.slice_num_list[-1]

    if sphere_num >= 1:
        sphere_pixel_1, sphere_actual_1 = generate_sphere(img_xy_size, img_z_size, 30, 6.5/40*100*10/2, 6.5/40*100*10/2, cz_list[0], 1)
        # sphere_pixel_1, sphere_actual_1 = generate_sphere(img_xy_size, img_z_size, 40, 43.75, 6.5/40*50*10, cz_list[0], 1) # radius, cx, cy, cz单位为0.1um # r = 60
        # sphere_pixel_1, sphere_actual_1 = generate_sphere(img_xy_size, img_z_size, 40, 6.5/40*75*10, 6.5/40*75*10, cz_list[0], 1)
        sphere_pixel = sphere_pixel_1
        sphere_actual= sphere_actual_1
    if sphere_num >= 2:
        sphere_pixel_2, sphere_actual_2 = generate_sphere(img_xy_size, img_z_size, 15, 6.5/40*100*10/2, 6.5/40*100*10/2, cz_list[1], 1) # r = 45
        # sphere_pixel_2, sphere_actual_2 = generate_sphere(img_xy_size, img_z_size, 20, 103.75, 6.5/40*50*10, cz_list[1], 1) # r = 45
        sphere_pixel += sphere_pixel_2
        sphere_actual+= sphere_actual_2
    if sphere_num >= 3:
        sphere_pixel_3, sphere_actual_3 = generate_sphere(img_xy_size, img_z_size, 5, 6.5/40*100*10/2, 6.5/40*100*10/2, cz_list[2], 1)
        # sphere_pixel_3, sphere_actual_3 = generate_sphere(img_xy_size, img_z_size, 10,  133.75, 6.5/40*50*10, cz_list[2], 1) # r = 30

        sphere_pixel += sphere_pixel_3
        sphere_actual+= sphere_actual_3
    if sphere_num >= 4:
        sphere_pixel_4, sphere_actual_4 = generate_sphere(img_xy_size, img_z_size, 2.5, 6.5/40*100*10/2, 6.5/40*100*10/2, cz_list[3], 1)
        # sphere_pixel_4, sphere_actual_4 = generate_sphere(img_xy_size, img_z_size, 5,  148.75, 6.5/40*50*10, cz_list[3], 1) # r = 15

        sphere_pixel += sphere_pixel_4
        sphere_actual+= sphere_actual_4
    if sphere_num >= 5:
        sphere_pixel_5, sphere_actual_5 = generate_sphere(img_xy_size, img_z_size, 2.5, 6.5/40*50*10/2, 6.5/40*50*10/2, cz_list[4], 1)
        # sphere_pixel_5, sphere_actual_5 = generate_sphere(img_xy_size, img_z_size, 2.5,  156.25, 6.5/40*50*10, cz_list[4], 1) #

        sphere_pixel += sphere_pixel_5
        sphere_actual+= sphere_actual_5

    if show:
        plt.figure()
        plt.subplot(231),  plt.imshow(sphere_pixel [:,:,sphere_pixel.shape[2]//2],  cmap='gray', vmin=config.plot_range[0], vmax=config.plot_range[1]), plt.title('pixel_xy')
        plt.subplot(232),  plt.imshow(sphere_pixel [:,sphere_pixel.shape[1]//2,:],  cmap='gray', vmin=config.plot_range[0], vmax=config.plot_range[1]), plt.title('pixel_xz')
        plt.subplot(233),  plt.imshow(sphere_pixel [sphere_pixel.shape[0]//2,:,:],  cmap='gray', vmin=config.plot_range[0], vmax=config.plot_range[1]), plt.title('pixel_yz')
        plt.subplot(234),  plt.imshow(sphere_actual[:,:,sphere_actual.shape[2]//2], cmap='gray', vmin=config.plot_range[0], vmax=config.plot_range[1]), plt.title('actual_xy')
        plt.subplot(235),  plt.imshow(sphere_actual[:,sphere_actual.shape[1]//2,:], cmap='gray', vmin=config.plot_range[0], vmax=config.plot_range[1]), plt.title('actual_xz')
        plt.subplot(236),  plt.imshow(sphere_actual[sphere_actual.shape[0]//2,:,:], cmap='gray', vmin=config.plot_range[0], vmax=config.plot_range[1]), plt.title('actual_yz')

        point_coord_pixel  = np.array(np.where(sphere_pixel==1))
        point_coord_actual = np.array(np.where(sphere_actual==1))

        fig = plt.figure(dpi=128, figsize=(8,8))
        ax = fig.add_subplot(121, projection='3d')
        coeff_xy = 10*config.HR_pixel_size*img_xy_size/img_xy_size
        coeff_z  = 10*config.imaging_depth/img_z_size
        ax.set_xlim(0, coeff_xy*img_xy_size)
        ax.set_ylim(0, coeff_xy*img_xy_size)
        ax.set_zlim(0, round(img_z_size*coeff_z))
        plt.xlabel('x(0.1um)')
        plt.ylabel('y(0.1um)')
        ax.scatter(point_coord_actual[0], point_coord_actual[1], point_coord_actual[2], s=1)
        plt.title('actual microsphere sample')

        bx = fig.add_subplot(122, projection='3d')
        bx.set_xlim(0, img_xy_size)
        bx.set_ylim(0, img_xy_size)
        bx.set_zlim(0, round(img_z_size))
        plt.xlabel('x(pixel)')
        plt.ylabel('y(pixel)')
        bx.scatter(point_coord_pixel[0], point_coord_pixel[1], point_coord_pixel[2], s=1)
        plt.title('pixel microsphere sample')
        # plt.show()

    return sphere_pixel


def read_phantom_sample():
    sample = scipy.io.loadmat(config.PHANTOM_PATH)
    return sample['obj']


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


def generate_optimizer(slice_number, model, lr):
    if slice_number   == 1:
        optimizer = torch.optim.Adam([
        {'params': model.SliceLayer_dic['Slice_0'].parameters(), 'lr': lr}])
    elif slice_number == 2:
        optimizer = torch.optim.Adam([
        {'params': model.SliceLayer_dic['Slice_0'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_1'].parameters(), 'lr': lr}])
    elif slice_number == 3:
        optimizer = torch.optim.Adam([
        {'params': model.SliceLayer_dic['Slice_0'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_1'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_2'].parameters(), 'lr': lr}])
    elif slice_number == 4:
        optimizer = torch.optim.Adam([
        {'params': model.SliceLayer_dic['Slice_0'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_1'].parameters(), 'lr': lr},            
        {'params': model.SliceLayer_dic['Slice_2'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_3'].parameters(), 'lr': lr}])
    elif slice_number == 6:
        optimizer = torch.optim.Adam([
        {'params': model.SliceLayer_dic['Slice_0'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_1'].parameters(), 'lr': lr},            
        {'params': model.SliceLayer_dic['Slice_2'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_3'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_4'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_5'].parameters(), 'lr': lr}])
    elif slice_number == 8:
        # lr = lr / 2
        optimizer = torch.optim.Adam([
        {'params': model.SliceLayer_dic['Slice_0'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_1'].parameters(), 'lr': lr},            
        {'params': model.SliceLayer_dic['Slice_2'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_3'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_4'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_5'].parameters(), 'lr': lr},            
        {'params': model.SliceLayer_dic['Slice_6'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_7'].parameters(), 'lr': lr}])
    elif slice_number == 9:
        # lr = lr / 2
        optimizer = torch.optim.Adam([
        {'params': model.SliceLayer_dic['Slice_0'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_1'].parameters(), 'lr': lr},            
        {'params': model.SliceLayer_dic['Slice_2'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_3'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_4'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_5'].parameters(), 'lr': lr},            
        {'params': model.SliceLayer_dic['Slice_6'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_7'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_8'].parameters(), 'lr': lr}])
    elif slice_number == 10:
        optimizer = torch.optim.Adam([
        {'params': model.SliceLayer_dic['Slice_0'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_1'].parameters(), 'lr': lr},          
        {'params': model.SliceLayer_dic['Slice_2'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_3'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_4'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_5'].parameters(), 'lr': lr},            
        {'params': model.SliceLayer_dic['Slice_6'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_7'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_8'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_9'].parameters(), 'lr': lr}])
    elif slice_number == 20:
        optimizer = torch.optim.Adam([
        {'params': model.SliceLayer_dic['Slice_0'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_1'].parameters(), 'lr': lr},          
        {'params': model.SliceLayer_dic['Slice_2'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_3'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_4'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_5'].parameters(), 'lr': lr},            
        {'params': model.SliceLayer_dic['Slice_6'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_7'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_8'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_9'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_10'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_11'].parameters(), 'lr': lr},            
        {'params': model.SliceLayer_dic['Slice_12'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_13'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_14'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_15'].parameters(), 'lr': lr},            
        {'params': model.SliceLayer_dic['Slice_16'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_17'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_18'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_19'].parameters(), 'lr': lr}])
    elif slice_number == 30:
        optimizer = torch.optim.Adam([
        {'params': model.SliceLayer_dic['Slice_0'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_1'].parameters(), 'lr': lr},          
        {'params': model.SliceLayer_dic['Slice_2'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_3'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_4'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_5'].parameters(), 'lr': lr},            
        {'params': model.SliceLayer_dic['Slice_6'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_7'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_8'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_9'].parameters(), 'lr': lr},            
        {'params': model.SliceLayer_dic['Slice_10'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_11'].parameters(), 'lr': lr},            
        {'params': model.SliceLayer_dic['Slice_12'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_13'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_14'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_15'].parameters(), 'lr': lr},            
        {'params': model.SliceLayer_dic['Slice_16'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_17'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_18'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_19'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_20'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_21'].parameters(), 'lr': lr},            
        {'params': model.SliceLayer_dic['Slice_22'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_23'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_24'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_25'].parameters(), 'lr': lr},            
        {'params': model.SliceLayer_dic['Slice_26'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_27'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_28'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_29'].parameters(), 'lr': lr}
        ])
    elif slice_number == 50:
        optimizer = torch.optim.Adam([
        {'params': model.SliceLayer_dic['Slice_0'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_1'].parameters(), 'lr': lr},          
        {'params': model.SliceLayer_dic['Slice_2'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_3'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_4'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_5'].parameters(), 'lr': lr},            
        {'params': model.SliceLayer_dic['Slice_6'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_7'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_8'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_9'].parameters(), 'lr': lr},            
        {'params': model.SliceLayer_dic['Slice_10'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_11'].parameters(), 'lr': lr},            
        {'params': model.SliceLayer_dic['Slice_12'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_13'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_14'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_15'].parameters(), 'lr': lr},            
        {'params': model.SliceLayer_dic['Slice_16'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_17'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_18'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_19'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_20'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_21'].parameters(), 'lr': lr},            
        {'params': model.SliceLayer_dic['Slice_22'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_23'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_24'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_25'].parameters(), 'lr': lr},            
        {'params': model.SliceLayer_dic['Slice_26'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_27'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_28'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_29'].parameters(), 'lr': lr},            
        {'params': model.SliceLayer_dic['Slice_30'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_31'].parameters(), 'lr': lr},            
        {'params': model.SliceLayer_dic['Slice_32'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_33'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_34'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_35'].parameters(), 'lr': lr},            
        {'params': model.SliceLayer_dic['Slice_36'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_37'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_38'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_39'].parameters(), 'lr': lr},            
        {'params': model.SliceLayer_dic['Slice_40'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_41'].parameters(), 'lr': lr},            
        {'params': model.SliceLayer_dic['Slice_42'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_43'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_44'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_45'].parameters(), 'lr': lr},            
        {'params': model.SliceLayer_dic['Slice_46'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_47'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_48'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_49'].parameters(), 'lr': lr}])
    elif slice_number == 60:
        optimizer = torch.optim.Adam([
        {'params': model.SliceLayer_dic['Slice_0'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_1'].parameters(), 'lr': lr},          
        {'params': model.SliceLayer_dic['Slice_2'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_3'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_4'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_5'].parameters(), 'lr': lr},            
        {'params': model.SliceLayer_dic['Slice_6'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_7'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_8'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_9'].parameters(), 'lr': lr},            
        {'params': model.SliceLayer_dic['Slice_10'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_11'].parameters(), 'lr': lr},            
        {'params': model.SliceLayer_dic['Slice_12'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_13'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_14'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_15'].parameters(), 'lr': lr},            
        {'params': model.SliceLayer_dic['Slice_16'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_17'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_18'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_19'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_20'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_21'].parameters(), 'lr': lr},            
        {'params': model.SliceLayer_dic['Slice_22'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_23'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_24'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_25'].parameters(), 'lr': lr},            
        {'params': model.SliceLayer_dic['Slice_26'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_27'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_28'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_29'].parameters(), 'lr': lr},            
        {'params': model.SliceLayer_dic['Slice_30'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_31'].parameters(), 'lr': lr},            
        {'params': model.SliceLayer_dic['Slice_32'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_33'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_34'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_35'].parameters(), 'lr': lr},            
        {'params': model.SliceLayer_dic['Slice_36'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_37'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_38'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_39'].parameters(), 'lr': lr},            
        {'params': model.SliceLayer_dic['Slice_40'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_41'].parameters(), 'lr': lr},            
        {'params': model.SliceLayer_dic['Slice_42'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_43'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_44'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_45'].parameters(), 'lr': lr},            
        {'params': model.SliceLayer_dic['Slice_46'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_47'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_48'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_49'].parameters(), 'lr': lr},            
        {'params': model.SliceLayer_dic['Slice_50'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_51'].parameters(), 'lr': lr},            
        {'params': model.SliceLayer_dic['Slice_52'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_53'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_54'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_55'].parameters(), 'lr': lr},            
        {'params': model.SliceLayer_dic['Slice_56'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_57'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_58'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_59'].parameters(), 'lr': lr},
        {'params': model.LightCor.parameters(), 'lr': 2e-3}])

    elif slice_number == 100:
        optimizer = torch.optim.Adam([
        {'params': model.SliceLayer_dic['Slice_0'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_1'].parameters(), 'lr': lr},          
        {'params': model.SliceLayer_dic['Slice_2'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_3'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_4'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_5'].parameters(), 'lr': lr},            
        {'params': model.SliceLayer_dic['Slice_6'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_7'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_8'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_9'].parameters(), 'lr': lr},            
        {'params': model.SliceLayer_dic['Slice_10'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_11'].parameters(), 'lr': lr},            
        {'params': model.SliceLayer_dic['Slice_12'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_13'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_14'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_15'].parameters(), 'lr': lr},            
        {'params': model.SliceLayer_dic['Slice_16'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_17'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_18'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_19'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_20'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_21'].parameters(), 'lr': lr},            
        {'params': model.SliceLayer_dic['Slice_22'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_23'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_24'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_25'].parameters(), 'lr': lr},            
        {'params': model.SliceLayer_dic['Slice_26'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_27'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_28'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_29'].parameters(), 'lr': lr},            
        {'params': model.SliceLayer_dic['Slice_30'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_31'].parameters(), 'lr': lr},            
        {'params': model.SliceLayer_dic['Slice_32'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_33'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_34'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_35'].parameters(), 'lr': lr},            
        {'params': model.SliceLayer_dic['Slice_36'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_37'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_38'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_39'].parameters(), 'lr': lr},            
        {'params': model.SliceLayer_dic['Slice_40'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_41'].parameters(), 'lr': lr},            
        {'params': model.SliceLayer_dic['Slice_42'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_43'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_44'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_45'].parameters(), 'lr': lr},            
        {'params': model.SliceLayer_dic['Slice_46'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_47'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_48'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_49'].parameters(), 'lr': lr},            
        {'params': model.SliceLayer_dic['Slice_50'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_51'].parameters(), 'lr': lr},            
        {'params': model.SliceLayer_dic['Slice_52'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_53'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_54'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_55'].parameters(), 'lr': lr},            
        {'params': model.SliceLayer_dic['Slice_56'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_57'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_58'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_59'].parameters(), 'lr': lr},            
        {'params': model.SliceLayer_dic['Slice_60'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_61'].parameters(), 'lr': lr},            
        {'params': model.SliceLayer_dic['Slice_62'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_63'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_64'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_65'].parameters(), 'lr': lr},            
        {'params': model.SliceLayer_dic['Slice_66'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_67'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_68'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_69'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_70'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_71'].parameters(), 'lr': lr},          
        {'params': model.SliceLayer_dic['Slice_72'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_73'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_74'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_75'].parameters(), 'lr': lr},            
        {'params': model.SliceLayer_dic['Slice_76'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_77'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_78'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_79'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_80'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_81'].parameters(), 'lr': lr},          
        {'params': model.SliceLayer_dic['Slice_82'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_83'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_84'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_85'].parameters(), 'lr': lr},            
        {'params': model.SliceLayer_dic['Slice_86'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_87'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_88'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_89'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_90'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_91'].parameters(), 'lr': lr},          
        {'params': model.SliceLayer_dic['Slice_92'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_93'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_94'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_95'].parameters(), 'lr': lr},            
        {'params': model.SliceLayer_dic['Slice_96'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_97'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_98'].parameters(), 'lr': lr},
        {'params': model.SliceLayer_dic['Slice_99'].parameters(), 'lr': lr},
        {'params': model.LightCor.parameters(), 'lr': 1e-3}
        ])
    else:
        optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': lr}])
    return optimizer

def create_illu_mode():
    led = Illumination()
    led.createIlluminationAngle()
    led.draw_illu_mode()

if __name__ == '__main__': # 功能测试
    # generate_sphere_sample()
    create_illu_mode()
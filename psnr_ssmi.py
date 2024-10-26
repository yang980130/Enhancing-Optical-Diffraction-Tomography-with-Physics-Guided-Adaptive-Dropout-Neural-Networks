import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.io import imread
import math

img_height = 200
img_width = 200

# 假设图像存储在一个文件夹中，文件名是按照顺序编号的
# folder_path = 'psnr&ssmi/GT/'  # 图像文件夹路径
# folder_path2 = 'psnr&ssmi/past vision/DO-BPN_NA0.7_ps0.1625/' #'psnr&ssmi/past vision/DO-BPN_NA0.7_ps0.1625/'  'psnr&ssmi/past vision/MSBP_NA0.7_ps0.1625/'
# folder_path = 'psnr&ssmi/GT_4mic/'  # 图像文件夹路径
folder_path2 = 'psnr&ssmi/4_DOBPN/' #'psnr&ssmi/past vision/DO-BPN_NA0.7_ps0.1625/'  'psnr&ssmi/past vision/MSBP_NA0.7_ps0.1625/'
folder_path = 'psnr&ssmi/GT_4mic/'  # 图像文件夹路径
# folder_path2 = 'psnr&ssmi/delta0.04nonnegative/' #'psnr&ssmi/past vision/DO-BPN_NA0.7_ps0.1625/'  'psnr&ssmi/past vision/MSBP_NA0.7_ps0.1625/'
# folder_path2 = 'psnr&ssmi/delta0.04nonnegative/'
slices = 100  # 切片数量

# 初始化两个三维数组来存储读取的图像数据
img1_stack = np.zeros((slices, img_height, img_width), dtype=np.uint8)
img2_stack = np.zeros((slices, img_height, img_width), dtype=np.uint8)

# 读取图像并填充到三维数组中
for i in range(slices):
    # img1_path = f"{folder_path}xy_{i+1}.tiff"  # 根据实际文件名格式修改
    img1_path = f"{folder_path}xy_{i}.jpg"
    img2_path = f"{folder_path2}xy_{i}.jpg"
    img1_stack[i] = imread(img1_path)
    img2_stack[i] = imread(img2_path)

# 计算PSNR和SSIM
psnr_values = []
ssim_values = []
for i in range(slices):
    # 计算每张切片的PSNR和SSIM
    psnr_value = psnr(img1_stack[i], img2_stack[i], data_range=255)
    ssim_value = ssim(img1_stack[i], img2_stack[i], data_range=255)
    # if i == 0 or i == 40 or i == 102 or i == 130 or i == 151:
    if i == 50:
        print("XY Slice{}, PSNR={}".format(i, psnr_value))
        print("XY Slice{}, SSIM={}".format(i, ssim_value))
    if not (math.isinf(psnr_value)): psnr_values.append(psnr_value)
    ssim_values.append(ssim_value)

for i in range(img_height):
    # 计算每张切片的PSNR和SSIM
    psnr_value = psnr(img1_stack[:,:,i], img2_stack[:,:,i], data_range=255)
    ssim_value = ssim(img1_stack[:,:,i], img2_stack[:,:,i], data_range=255)
    # if i == 72 or i == 79:
    if i == 100:
        print("XZ Slice{}, PSNR={}".format(i, psnr_value))
        print("XZ Slice{}, SSIM={}".format(i, ssim_value))


for i in range(img_width):
    # 计算每张切片的PSNR和SSIM
    psnr_value = psnr(img1_stack[:,i,:], img2_stack[:,i,:], data_range=255)
    ssim_value = ssim(img1_stack[:,i,:], img2_stack[:,i,:], data_range=255)
    # if i == 81 or i == 92 or i == 120:
    if i == 100:
        print("YZ Slice{}, PSNR={}".format(i, psnr_value))
        print("YZ Slice{}, SSIM={}".format(i, ssim_value))


# 计算所有切片PSNR和SSIM的平均值
average_psnr = np.mean(psnr_values)
average_ssim = np.mean(ssim_values)

print(f"Average PSNR: {average_psnr}")
print(f"Average SSIM: {average_ssim}")
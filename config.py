from math import pi
import torch
import numpy as np

bit = 32
# np_float_datatype   = np.float16 if bit == 32 else np.float32
np_float_datatype   = np.float32 if bit == 32 else np.float64
np_complex_datatype = np.complex64 if bit == 32 else np.complex128
# torch_float_datatype = torch.half if bit == 32 else torch.float32
torch_float_datatype = torch.float32 if bit == 32 else torch.float64
torch_complex_datatype = torch.complex64 if bit == 32 else torch.complex128

import warnings
warnings.filterwarnings("ignore")

def initDevice(use_gpu=False):
        torch.set_printoptions(precision=8)
        if use_gpu:
            print('cuda availabel is {}'.format(torch.cuda.is_available()))
            return 'cuda'
        else:
            print('using cpu')
            return 'cpu'


class Configs:
    def __init__(self):
        self.np_float       = np_float_datatype
        self.torch_float    = torch_float_datatype

        self.np_complex     = np_complex_datatype
        self.torch_complex  = torch_complex_datatype

        USE_GPU             = True
        self.DEVICE         = initDevice(USE_GPU)
        self.ILLU_MODE      = 'spiral' # 'array' 'annular' 'spiral'
        self.SCATTER_MODE   = 'MS' # 'MS','MS_LIAC','1st born'
        self.tv             = True
        self.tv_beta        = [2, 0.1, 0.5] # iter迭代次数 t正则化度 lamda保真度 [1, 0.05, 0.5] [1, 0, 1] [10, 0.05, 0]
        self.tv_epoch       = 45
        self.non_negative_layer = False
        self.non_negative_eachiter = True
        self.en_optim_light = True

        self.capture_size        = 100  # C.elegan 1200/360 diatom 700
        self.slice_size          = 100 # 注意，仿真时，这个尺寸一定要和读取的phantom的xy平面一致！
        self.pad_size            = 25   # each side
        self.slice_pad_size      = self.slice_size + 2*self.pad_size
        self.slice_num_list      = [100] # diatom 35
        self.slice_num           = self.slice_num_list[-1]
        self.phase_obj_shape     = (self.slice_size, self.slice_size, self.slice_num)
        self.phase_obj_pad_shape = (self.slice_pad_size, self.slice_pad_size, self.slice_num)
        self.plot_range          = [0, 0.03] # [-0.01, 0.07]

        #
        self.imaging_depth = 16.25 # 16.25 # predict 15 recover 30   # C.elegan(waller) 30/24 # diatom 35 simulate 22.5 15.75
        self.n_media       = 1.33  # refractive index of immersion media # C.elegan(waller) 1.33 diatom 35
        self.n_min         = 1.33 # refractive index of min density feature
        self.n_max         = 1.36 # refractive index of max density feature

        # simulation
        self.sample_from  = 'generate'   # generate/read
        self.PHANTOM_PATH = './simulationGT/phantom.mat'
        
        # experiment
        self.data_path  = './FOV_01_reconstruction_CElegan_resources/FOV_01_rawdata_CElegan.tif'
        # self.data_path  = './microsphere_ROI/MultiTIFF.tif'
        self.param_path = './FOV_01_reconstruction_CElegan_resources/FOV_01_reconstruction_CElegan_params.mat'

        self.en_optim_object   = True
        self.learning_rate     = 1e-4 # actual 4.8e-4 5e-4 1e-3 simulate 1e-4? CEtotalfield 1e-3 **0.5 5e-4
        self.num_epoch         = 90
        
        # self.savelabel         = 'phantom1.36_bg1.33_nonnegative_plt[0.00, 0.03]_ps0.1625_NA0.85_tv[2, 0.1, 0.5]_spiral120_保存loss_imsize100_lr1e-4'  #_nonnegativelayer
        # self.savelabel         = '1microsphere8um1.56oil_nonnegativelayer_plt[0.00, 0.04]_NA_seq193_伪矩形照明_**0.5_tv[2, 0.1, 0.5]_保存npy&mat_保存loss_imsize150_lr2e-4'
                                  #'10um_microscope_simulation_as_experiment_justnonnegative' #non_negative_layer0 remove_background_
        self.savelabel         = '4microsphere_不同深度缩小(3、4球略远2)_nonnegative_1.36_bg1.33_plt[0.00, 0.075]_ps0.1625_NA0.85_spiral120_tv[2, 0.075, 0.5]_保存loss_imsize100_lr1e-4'
        self.wlength           = 0.532 # um green waller 0.532 jiaji/DeCAF 0.515 simulate 0.532 #microscope experiment blue0.47? # microsphere(our experiment) red 0.623? 0.632
        self.k0                = 2*pi / self.wlength
        self.NA                = 0.85# 0.85 # 1.0556 Waller # 0.65 Li/DeCAF # simulate 0.75 #microscope experiment 0.6 # 神经元和实验微球0.7
        self.HR_pixel_size     = 6.5/40 # 0.1154 Waller # 6.5/40 Li  #0.225 6.5/40 微球simulation #0.1 phantom simulation
        self.imaging_width     = self.slice_size * self.HR_pixel_size

        self.__num_illu   = 120
        self.using_illu   = 11**2
        self.arraysize    = 11
        self.num_illu     = self.arraysize**2 if self.ILLU_MODE == 'array' else self.__num_illu
        # self.num_illu     = 193
        self.led_gap      = 5    # mm
        self.led_height   = 37.5 # mm
        self.x_offset     = 0    # mm # -0.05655
        self.y_offset     = 0    # mm # 0.0719875
        self.theta_offset = 0    # mm
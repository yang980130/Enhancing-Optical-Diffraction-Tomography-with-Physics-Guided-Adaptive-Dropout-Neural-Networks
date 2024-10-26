from config import Configs
from Base_part.illumination import Illumination
from Base_part.Measurement import Measurement
from Base_part.Object import PhaseObject3D
from optim import Optimization, generate_optimizer
from nn_model import MultiSliceNN
from utils import generate_sphere_sample, read_phantom_sample

import matplotlib.pyplot as plt
import torch


config = Configs()

def simulate_recover():

    print('cuda availabel is {}'.format(torch.cuda.is_available()))
    
    gt_illumination  = Illumination()
    gt_measurement   = Measurement(num_measurement = config.num_illu)
    gt_model         = MultiSliceNN().to(config.DEVICE)
    optimizer, scheduler = generate_optimizer(gt_model, config.learning_rate)
    gt_optim         = Optimization(gt_model, optimizer, scheduler)
    gt_sample_3d     = PhaseObject3D(config.phase_obj_shape, config.pad_size, config.n_media, config.n_min, config.n_max)

    if config.sample_from == 'read':
        sphere_sample_array = read_phantom_sample()
        gt_sample_3d.createPhaseObject3Dfrom3DArray(sphere_sample_array)
        gt_sample_3d.showObject()
        # gt_sample_3d.analys_RI()

    elif config.sample_from == 'generate':
        # sphere_sample_array = generate_sphere_sample(cz_list=[81.25]) # 1 micsphere
        sphere_sample_array = generate_sphere_sample(cz_list=[40, 102, 131, 151]) # 4 micsphere
        gt_sample_3d.createPhaseObject3Dfrom3DArray(sphere_sample_array)

    # 
    if config.ILLU_MODE == 'array':
        gt_illumination.createIlluminationAngle()
    elif config.ILLU_MODE == 'spiral':
        gt_illumination.generateSpiralPath(revs=6)
    elif config.ILLU_MODE == 'annular':
        gt_illumination.createAnnular()
    else:
        raise('config:illu_mode set error!')
    gt_sample_3d.saveGTobject()

    # save with_sample measurement
    gt_model.initModel(gt_sample_3d, mode='gt')
    gt_measurement.in_measurement = gt_optim.test(gt_illumination)
    gt_measurement.in2show_transform()
    gt_measurement.save_measurement()
    show_measurement = gt_measurement.show_measurement

    in_measurement = gt_measurement.in_measurement

    plt.figure()
    plt.subplot(221), plt.imshow(show_measurement[0], cmap='gray')
    plt.subplot(222), plt.imshow(show_measurement[1], cmap='gray')
    plt.subplot(223), plt.imshow(show_measurement[2], cmap='gray')
    plt.subplot(224), plt.imshow(show_measurement[3], cmap='gray')
    # plt.show()

    recover_sample = PhaseObject3D(config.phase_obj_shape, config.pad_size, config.n_media, config.n_min, config.n_max)
    recover_sample.zeroInitPhaseObject3D()
    # plt.show()

    model = MultiSliceNN().to(config.DEVICE)
    model.initModel(recover_sample)
    optimizer, scheduler = generate_optimizer(model, lr=config.learning_rate)
    optim  = Optimization(model, optimizer, scheduler)
    optim.train(gt_illumination, in_measurement, gt_sample_3d)

    recover_sample = model.extractParameters2cpu()
    recover_sample.showObject()
    plt.show()
    return 0


def main():
    simulate_recover()

if __name__ == '__main__':
    main()
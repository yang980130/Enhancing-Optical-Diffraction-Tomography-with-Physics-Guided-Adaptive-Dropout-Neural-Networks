from config import Configs
from tv_reg_torch import tv_reg_3d

import torch
import torch.nn as nn
from torch.utils.tensorboard.writer import SummaryWriter
import numpy as np
import time

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


def generate_optimizer(model, lr):
    optimizer_1 = torch.optim.Adam([
        {'params': x.parameters(), 'lr': lr} for x in model.SliceLayer_dic.values()])
    lr_ = 1e-4
    optimizer_2 = torch.optim.Adam([
        {'params': model.LightCor.parameters(), 'lr': 1e-10}])
    optimizer_3 = torch.optim.Adam([
        {'params': model.BS.parameters(), 'lr': 1e-2}])
    schedular_1 = torch.optim.lr_scheduler.StepLR(optimizer_1, step_size=10, gamma=0.9)  # step_size=10, gamma=0.975
    schedular_2 = torch.optim.lr_scheduler.StepLR(optimizer_2, step_size=10, gamma=0.9)
    schedular_3 = torch.optim.lr_scheduler.StepLR(optimizer_3, step_size=10, gamma=0.9)
    optimizer_list = [optimizer_1, optimizer_2, optimizer_3]
    schedular_list = [schedular_1, schedular_2, schedular_3]
    return optimizer_list, schedular_list


class Optimization():
    def __init__(self, model, optimizer, scheduler):
        config     = Configs()
        self.loss_fn      = nn.L1Loss() # nn.MSELoss()
        self.loss_with_gt = nn.MSELoss()
        self.model      = model
        self.optimizer  = optimizer
        self.scheduler  = scheduler
        self.loss_list  = []
        self.lossgt_list = []
        self.num_illu  = config.num_illu

    def test(self, ledarray):
        input     = torch.tensor(ledarray.fxfy).to(config.DEVICE)
        loc       = torch.tensor(ledarray.loc).to(config.DEVICE)
        predict   = []
        for i in range(self.num_illu):
            output, _ = self.model(input[i], 0, 0)
            output = output.detach()
            predict.append(output)
        return predict

    def train(self, illumination, in_measurement, gt_object): #simulation
        device    = config.DEVICE
        num_epoch = config.num_epoch
        optimizer, scheduler = generate_optimizer(self.model, config.learning_rate)
        target    = in_measurement

        for i in range(config.num_illu):
            target[i] = in_measurement[i]
            # target[i] = in_measurement[i] + background_measurement[i]
            # target[i] = in_measurement[i] + background_measurement[i] - 1

        input     = torch.tensor(illumination.fxfy).to(device)
        tvloss_fn = TVLoss(100) # 传入参数为tvloss的放大系数
        optim_start = time.time()
        ### 启用tensorboard监视训练 ###
        writer = SummaryWriter(comment='NN', filename_suffix='NN')
        # edgefield = np.load('边界灯序号.npy')
        for epoch in range(num_epoch):
            running_loss = 0
            epoch_start = time.time()
            for i in range(self.num_illu):
                output, _ = self.model(input[i], i, epoch)
                loss   = self.loss_fn(output, target[i])
                # loss   = self.loss_fn(output, target[idx])
                optimizer[0].zero_grad()
                optimizer[1].zero_grad()
                optimizer[2].zero_grad()
                # optimizer[3].zero_grad()
                loss.backward(retain_graph=True)
                optimizer[0].step()
                optimizer[1].step()
                optimizer[2].step()
                # optimizer[3].step()
                torch.cuda.empty_cache()
                running_loss = running_loss + loss.item()
                if config.non_negative_eachiter:
                    object = self.model.extractParameters2cuda()
                    object.non_negative_regularization()
                    self.model.initModel(object)

            ###------------------tv regularization-----------------###
            if config.tv and epoch <= config.tv_epoch:
                object = self.model.extractParameters2cuda()
                tv_loss = tvloss_fn(object.RI_cuda)
                print('epoch:{}, before_tv:{}'.format(epoch+1, tv_loss.item()))
                tv_start = time.time()
                tv_object = tv_reg_3d(object, iter=config.tv_beta[0], t=config.tv_beta[1], ep=1, lamda=config.tv_beta[2])
                self.model.initModel(tv_object)
                #---多余，可优化---#
                _, weights = self.model([0, 0], 0, 0)
                tv_loss = tvloss_fn(weights)
                tv_end   = time.time()
                #----------------#
                print('tv cost {}s'.format(tv_end-tv_start))
                print('after_tv:{}'.format(tv_loss.item()))
            else:
                print('epoch:{}'.format(epoch+1))
            ###-------------------------------------------###

            ###-------------------------------------------###

            ###-----------------analys_RI-----------------###
            # if epoch+1 >= 20 and epoch%10==0:
            #     object = self.model.extractParameters2cuda()
            #     object.analys_RI()
            ###-------------------------------------------###
            scheduler[0].step()
            scheduler[1].step()
            scheduler[2].step()
            # scheduler[3].step()
            epoch_loss = running_loss / self.num_illu
            self.loss_list.append(epoch_loss) # 这个loss_list可以 返回/输出
            ###tensorboard###
            if config.SCATTER_MODE == 'MS':
                writer.add_scalars('loss', {'{}'.format('MS'): epoch_loss}, epoch)
            if config.SCATTER_MODE == 'MS_LIAC':
                writer.add_scalars('loss', {'{}'.format('MS_LIAC'): epoch_loss}, epoch)
            if config.SCATTER_MODE == '1st born':
                writer.add_scalars('loss', {'{}'.format('1st born'): epoch_loss}, epoch)

            ###记录每个epoch的优化结果###
            # if (epoch+1) % 5 == 0 or epoch == 0:
            sample = self.model.extractParameters2cpu() #!!! 
            # loss = self.loss_with_gt(sample.RI, gt_object.RI)
            loss = np.sum(np.abs(sample.RI - gt_object.RI)) / (config.slice_size**2 * config.slice_num_list[-1]) 
            self.lossgt_list.append(loss)
            print('The L1 loss with groundtruth {}'.format(loss))

            sample.saveOptimObject(epoch+1, self.loss_list, self.lossgt_list) # !!!
            
            # print('start save model params, until next print, save sucessful')
            # np.save('./3d_sample.npy', sample.RI)

            # print('start save model params, until next print, save sucessful')
            
            # torch.save(self.model.state_dict(), 'epoch_net_params.pth')
            
            # sample.saveOptimObject(epoch+1, A, self.loss_list)
            ###--------------------###
            print('loss:{}'.format(epoch_loss))
            epoch_end = time.time()
            print('last epoch cost {}s'.format(epoch_end-epoch_start))
            # del sample
            # del object
            # del tv_object
            # del tv_loss
        optim_end = time.time()
        print('total optim cost {}s'.format(optim_end-optim_start))

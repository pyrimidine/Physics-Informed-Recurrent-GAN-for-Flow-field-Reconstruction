#!/usr/bin/env python
#%%

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 不显示等级2以下的提示信息
import argparse
import sys
from datetime import datetime
import matplotlib.cm as cm

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

import torchvision
from models import *
from data import *
from visual import *
from loss import *


parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--cuda', type=bool, default=True, help='enables cuda')



parser.add_argument('--LR_imageSize', type=int, default=32, help='the low resolution image size')
parser.add_argument('--SNR', type=int, default=1000, help='signal noise ratio in LR observation, SNR=10*1og10(signal/noise), >=1000 without noise')
parser.add_argument('--upSampling', type=int, default=2, help='low to high resolution scaling factor')
parser.add_argument('--lastTrain', type=str, default='./PI_FRVSR_LR32', help="path to generator weights (to continue training)")



parser.add_argument('--bi', type=bool, default=False, help='Binary image or not')
parser.add_argument('--bi_th', type=float, default=0.2, help='Binary threshold')
parser.add_argument('--depth', type=int, default=32, help='')

opt = parser.parse_args(args=[])
# print(opt)



path = r'E:/RDE_GAN_HR_dataset/p=7e5_17e5/dataset'
dataloader = test_load_data(path, 'test1', opt)

batch_size = opt.batchSize
lr_width = opt.LR_imageSize
lr_hight = opt.LR_imageSize

generator = PI_FRVSR(batch_size, lr_width=lr_width, lr_height=lr_hight, SR_factor=2**3, Channel_num=5)
if opt.lastTrain != '':
    generator = torch.load(os.path.join(opt.lastTrain, 'generator_final.pth'))
generator.init_hidden(device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), batch_size=batch_size)
discriminator = Discriminator(input_channel=5)
if opt.lastTrain != '':
    # discriminator = torch.load(os.path.join(opt.lastTrain, 'discriminator_final.pth'))
    pass

# For the content loss
# print (feature_extractor)
generator_criterion = GeneratorLoss_PI_FRVSR()
discriminator_criterion = nn.BCELoss()
acc_criterion = nn.MSELoss()

ones_const = Variable(torch.ones(opt.batchSize, 1))

# if gpu is to be used
if opt.cuda:
    generator.cuda()
    discriminator.cuda()
    generator_criterion.cuda()
    ones_const = ones_const.cuda()


TIMESTAMP = "{0:%m-%d_%H.%M/}".format(datetime.now())
test_dir = './test/var_LR'
test_log_dir = test_dir + '/log' + str(opt.LR_imageSize) + '1'
try:
    os.makedirs(test_log_dir)
except OSError:
    pass

if __name__ == '__main__':
    print('SR testing')
    writer = SummaryWriter(test_log_dir)
    global iteration_number
    iteration_number = 0


    d_loss = 0
    i = 0
    errs = []

    for i, (var_HR, var_LR) in enumerate(dataloader):
        # for name, m in generator.named_modules():
        #     if isinstance(m, torch.nn.Conv2d):
        #         m.register_forward_pre_hook(hook_func)

        


        # Generate HR form LR inputs
        if opt.cuda:
            var_LR = var_LR.cuda().to(torch.float32)
            var_HR = var_HR.cuda().to(torch.float32)
        
        fake_SR, fake_dt = generator(var_LR)

        err_1 = np.mean((var_HR.detach().cpu().numpy() - fake_SR.detach().cpu().numpy())**2)
        err_2 = np.mean((var_HR.detach().cpu().numpy())**2)
        err = (err_1)/(err_2)

        # err = acc_criterion(var_HR, fake_SR).detach().cpu().numpy()
        errs.append(err)

        # Discriminate and update D network
        
        real_out = discriminator(var_HR) # should be 1
        fake_out = discriminator(fake_SR) # should be 0
        d_loss = discriminator_criterion(real_out, torch.rand_like(real_out)*0.2 + 0.9) + \
                    discriminator_criterion(fake_out, torch.rand_like(fake_out)*0.2)

        # Update G network: minimize 1-D(G(z)) + Physics Loss + Image Loss + TV Loss

        if opt.cuda:
            fake_SR = fake_SR.cuda()
            # fake_LR = fake_LR.cuda()
            fake_out = fake_out.cuda()

        # image_loss, adversarial_loss, physics_loss, tv_loss, flow_loss, residual = generator_criterion(fake_out, fake_SR, var_HR, fake_LR, var_LR, fake_dt, i)
        image_loss, adversarial_loss, physics_loss, tv_loss, residual_t, total_residual = \
            generator_criterion(fake_out, fake_SR, var_HR, fake_dt, i)
        
        # g_loss = image_loss + 0.001 * adversarial_loss + 0.006 * physics_loss + 2e-8 * tv_loss + 0.0001 * flow_loss
        # g_loss = physics_loss
        loss_weight = [1.0, 1.0, 0.05, 0.0001]
        g_loss = loss_weight[0]*image_loss 

        iteration_number += 1

        # writer.add_scalars('Loss items', {'image_loss': loss_weight[0]*image_loss.item(), \
        #                                     'adversarial_loss': loss_weight[3]*adversarial_loss.item(), \
        #                                     'physics_loss': loss_weight[1]*physics_loss.item(), \
        #                                     'tv_loss': loss_weight[2]*tv_loss.item()}, iteration_number)
        writer.add_scalar('G-loss', g_loss.item(), iteration_number)
        writer.add_scalar('D-loss', d_loss.item(), iteration_number)
        writer.add_scalar('Accuracy', err.item(), iteration_number)

        fake_img = fake_SR[0].unsqueeze(1)
        true_img = var_HR[0].unsqueeze(1)
        dt_img = residual_trans(fake_dt[0].unsqueeze(1))
        residual_t_img = residual_trans(residual_t[0].unsqueeze(1))
        residual_total_img = residual_trans(total_residual[0].unsqueeze(1))
        # fake_lr_img = F.interpolate(fake_LR[0].unsqueeze(1), size=(fake_img.size(2), fake_img.size(3)), mode='nearest')
        lr_img = F.interpolate(var_LR[0].unsqueeze(1), size=(fake_img.size(2), fake_img.size(3)), mode='nearest')
        int_img = F.interpolate(var_LR[0].unsqueeze(1), size=(fake_img.size(2), fake_img.size(3)), mode='bicubic')

        cmap = cm.get_cmap('jet')
        vis_img = torch.concat((lr_img, int_img, fake_img, true_img), dim=0)
        # color_vis_img = cmap(cv2.cvtColor(vis_img, cv2.COLOR_BGR2GRAY))
        vis_img = torchvision.utils.make_grid(vis_img, nrow=5, padding=4, normalize=True, scale_each=True)
        vis_img = vis_img.permute(1, 2, 0).cpu().numpy()
        color_vis_img = cmap(cv2.cvtColor(vis_img, cv2.COLOR_BGR2GRAY))
        writer.add_image("train_vis(SR, HR, LR, fakeLR, Flow)(rho, u, v, p, T)", vis_img, iteration_number, dataformats='HWC')
        writer.add_image("color_vis", color_vis_img, iteration_number, dataformats='HWC')


        # cmap = cm.get_cmap('bwr')
        # vis_img = torch.concat((dt_img, residual_t_img, residual_total_img), dim=0)
        # vis_img = torchvision.utils.make_grid(vis_img, nrow=5, padding=4)
        # vis_img = vis_img.permute(1, 2, 0).cpu().numpy()
        # color_vis_img = cmap(cv2.cvtColor(vis_img, cv2.COLOR_BGR2GRAY))
        # writer.add_image("train_vis(SR, HR, LR, fakeLR, Flow)(rho, u, v, p, T)", vis_img, iteration_number, dataformats='HWC')
        # writer.add_image("color_residual_vis", color_vis_img, iteration_number, dataformats='HWC')

        
        sys.stdout.write('\r \n [ %d %d ] Generator_Loss: %.4f Error: %.4f' % (i, len(dataloader), g_loss.item(), err))
    
    sys.stdout.write('\r \n Error: %.4f' % (np.mean(errs)))
    writer.close()

#%%
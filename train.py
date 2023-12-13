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

# 不更新输入下面的
# tensorboard --logdir=path/to/logs --reload_interval=1

import torchvision
from models import *
from data import *
from visual import *
from loss import *


parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=20, help='input batch size')
parser.add_argument('--LR_imageSize', type=int, default=32, help='the low resolution image size')
parser.add_argument('--SNR', type=int, default=1000, help='signal noise ratio in LR observation, SNR=10*1og10(signal/noise), >=1000 without noise') # 改
parser.add_argument('--upSampling', type=int, default=2, help='low to high resolution scaling factor') # 改
parser.add_argument('--out', type=str, default='./PI_FRVSR_LR32', help='folder to output model checkpoints') # 改

parser.add_argument('--nEpochs', type=int, default=100, help='number of epochs to train for')

parser.add_argument('--generatorLR', type=float, default=0.0001, help='learning rate for generator')
parser.add_argument('--discriminatorLR', type=float, default=0.0001, help='learning rate for discriminator')

parser.add_argument('--cuda', type=bool, default=True, help='enables cuda')
parser.add_argument('--lastTrain', type=str, default='', help="pat to generator weights (to continue training)")
parser.add_argument('--physicalWeights', type=float, default=1e-5, help="PINN balance weight")
parser.add_argument('--bi', type=bool, default=False, help='Binary image or not')
parser.add_argument('--bi_th', type=float, default=0.2, help='Binary threshold')
parser.add_argument('--depth', type=int, default=32, help='Sampling depth')

opt = parser.parse_args(args=[])
# print(opt)

try:
    os.makedirs(opt.out)
except OSError:
    pass
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# path = r'E:\RDE_GAN_HR_dataset\phi=0.8_0.5'
path = r'E:\RDE_GAN_HR_dataset\p=7e5_17e5\dataset'
dataloader = train_load_data(path, 'train1000', opt)

batch_size = opt.batchSize
lr_width = opt.LR_imageSize
lr_hight = opt.LR_imageSize

generator = PI_FRVSR(batch_size, lr_width=lr_width, lr_height=lr_hight, SR_factor=2**opt.upSampling, Channel_num=5)
generator.init_hidden(device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
if opt.lastTrain != '':
    generator = torch.load(os.path.join(opt.lastTrain, 'generator_final.pth'))
discriminator = Discriminator(input_channel=5)
if opt.lastTrain != '':
    # discriminator = torch.load(os.path.join(opt.lastTrain, 'discriminator_final.pth'))
    pass


# For the content loss
# print (feature_extractor)
generator_criterion = GeneratorLoss_PI_FRVSR()
discriminator_criterion = nn.BCELoss()

ones_const = Variable(torch.ones(opt.batchSize, 1))

# if gpu is to be used
if opt.cuda:
    generator.cuda()
    discriminator.cuda()
    generator_criterion.cuda()
    ones_const = ones_const.cuda()
    

optim_generator = optim.Adam(generator.parameters(), lr=opt.generatorLR)
optim_discriminator = optim.Adam(discriminator.parameters(), lr=opt.discriminatorLR)

#%%

# def hook_func(module, input):
#     x = input[0][0]
#     x = x.unsqueeze(1)
#     global index
#     image_batch = torchvision.utils.make_grid(x, padding=4)
#     image_batch = image_batch.cpu().numpy().transpose(1, 2, 0)
#     writer.add_image("test", image_batch, index, dataformats='HWC')
#     index += 1

#%%

TIMESTAMP = "{0:%m-%d_%H.%M/}".format(datetime.now())
train_log_dir = opt.out + '/log_' + TIMESTAMP

if __name__ == '__main__':
    print('training')
    writer = SummaryWriter(train_log_dir)
    global iteration_number
    iteration_number = 0

    for epoch in range(opt.nEpochs):
        mean_generator_content_loss = 0.0
        mean_generator_total_loss = 0.0

        d_loss = 0
        i = 0

        for i, (var_HR, var_LR) in enumerate(dataloader):
            # for name, m in generator.named_modules():
            #     if isinstance(m, torch.nn.Conv2d):
            #         m.register_forward_pre_hook(hook_func)

            fake_hrs = []
            fake_lrs = []
            fake_scrs = []
            real_scrs = []

            # Generate HR form LR inputs
            if opt.cuda:
                var_LR = var_LR.cuda().to(torch.float32)
                var_HR = var_HR.cuda().to(torch.float32)
            
            fake_SR, fake_dt = generator(var_LR)

            # Discriminate and update D network
            
            real_out = discriminator(var_HR) # should be 1
            fake_out = discriminator(fake_SR) # should be 0
            d_loss = discriminator_criterion(real_out, torch.rand_like(real_out)*0.2 + 0.9) + \
                     discriminator_criterion(fake_out, torch.rand_like(fake_out)*0.2)

            discriminator.zero_grad()
            d_loss.backward(retain_graph=True)
            optim_discriminator.step()

            # Update G network: minimize 1-D(G(z)) + Physics Loss + Image Loss + TV Loss

            if opt.cuda:
                fake_SR = fake_SR.cuda()
                # fake_LR = fake_LR.cuda()
                fake_out = fake_out.cuda()

            # image_loss, adversarial_loss, physics_loss, tv_loss, flow_loss, residual = generator_criterion(fake_out, fake_SR, var_HR, fake_LR, var_LR, fake_dt, i)
            image_loss, adversarial_loss, physics_loss, tv_loss, residual_t, total_residual = \
                generator_criterion(fake_out, fake_SR, var_HR, fake_dt, i)
            
            # g_loss = image_loss + 0.001 * adversarial_loss + 0.006 * physics_loss + 2e-8 * tv_loss + 0.0001 * flow_loss

            # weight: img, phys, tv, adv
            loss_weight = [10.0, 0.1, 0.05, 0.0001]
            g_loss = loss_weight[0]*image_loss + loss_weight[1]*physics_loss + \
                     loss_weight[2]*tv_loss + loss_weight[3]*adversarial_loss
            
            generator.zero_grad()
            g_loss.backward()
            optim_generator.step()
            iteration_number += 1

            err = (var_HR[:,1,:,:].detach().cpu().numpy() - fake_SR[:,1,:,:].detach().cpu().numpy())**2
            err = err/(var_HR[:,1,:,:].detach().cpu().numpy())**2
            err = np.mean(err)

            writer.add_scalars('Loss items', {'image_loss': loss_weight[0]*image_loss.item(), \
                                              'adversarial_loss': loss_weight[3]*adversarial_loss.item(), \
                                              'physics_loss': loss_weight[1]*physics_loss.item(), \
                                              'tv_loss': loss_weight[2]*tv_loss.item()}, iteration_number)
            writer.add_scalar('G-loss', g_loss.item(), iteration_number)
            writer.add_scalar('D-loss', d_loss.item(), iteration_number)
            writer.add_scalar('Error', err.item(), iteration_number)


            fake_img = fake_SR[0].unsqueeze(1)
            true_img = var_HR[0].unsqueeze(1)
            dt_img = residual_trans(fake_dt[0].unsqueeze(1))
            residual_t_img = residual_trans(residual_t[0].unsqueeze(1))
            residual_total_img = residual_trans(total_residual[0].unsqueeze(1))
            # fake_lr_img = F.interpolate(fake_LR[0].unsqueeze(1), size=(fake_img.size(2), fake_img.size(3)), mode='nearest')
            lr_img = F.interpolate(var_LR[0].unsqueeze(1), size=(fake_img.size(2), fake_img.size(3)), mode='nearest')

            cmap = cm.get_cmap('jet')
            vis_img = torch.concat((fake_img, true_img, lr_img), dim=0)
            vis_img = torchvision.utils.make_grid(vis_img, nrow=5, padding=4, normalize=True, scale_each=True)
            vis_img = vis_img.permute(1, 2, 0).cpu().numpy()
            color_vis_img = cmap(cv2.cvtColor(vis_img, cv2.COLOR_BGR2GRAY))
            # writer.add_image("train_vis(SR, HR, LR, fakeLR, Flow)(rho, u, v, p, T)", vis_img, iteration_number, dataformats='HWC')
            writer.add_image("color_train_vis", color_vis_img, iteration_number, dataformats='HWC')

            cmap = cm.get_cmap('bwr')
            vis_img = torch.concat((dt_img, residual_t_img, residual_total_img), dim=0)
            vis_img = torchvision.utils.make_grid(vis_img, nrow=5, padding=4)
            vis_img = vis_img.permute(1, 2, 0).cpu().numpy()
            color_vis_img = cmap(cv2.cvtColor(vis_img, cv2.COLOR_BGR2GRAY))
            # writer.add_image("train_vis(SR, HR, LR, fakeLR, Flow)(rho, u, v, p, T)", vis_img, iteration_number, dataformats='HWC')
            writer.add_image("color_train_residual_vis", color_vis_img, iteration_number, dataformats='HWC')


            sys.stdout.write('\r[ %d %d ][ %d %d ] Generator_Loss: %.4f' % (epoch, opt.nEpochs, i, len(dataloader), g_loss.item()))
            mean_generator_content_loss += g_loss.item()
        
        
        sys.stdout.write('\r[%d %d][%d %d] Generator_Loss: %.4f\n' % (epoch, opt.nEpochs, i, len(dataloader), mean_generator_content_loss/len(dataloader)))

        # Do checkpointing
        torch.save(generator, '%s/generator_final.pth' % opt.out)
        torch.save(discriminator, '%s/discriminator_final.pth' % opt.out)        
        torch.save(generator, '%s/generator_final.pth' % train_log_dir)
        torch.save(discriminator, '%s/discriminator_final.pth' % train_log_dir)
    
    writer.close()

#%%
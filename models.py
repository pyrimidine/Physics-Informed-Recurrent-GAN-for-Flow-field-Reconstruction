import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from skimage import img_as_ubyte
# an naive implementation of CVPR paper 'Frame-Recurrent Video Super-Resolution' https://arxiv.org/abs/1801.04590
from torchvision.models import vgg16, VGG16_Weights
from loss import *

class ResBlock(nn.Module):
    def __init__(self, conv_dim):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=conv_dim, out_channels=conv_dim,
                               kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=conv_dim, out_channels=conv_dim,
                               kernel_size=3, stride=1, padding=1)

    def forward(self, input):
        out = self.conv1(input)
        out = F.relu(out)
        out = self.conv2(out)
        out = input + out
        return out


class ConvLeaky(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ConvLeaky, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_dim, out_channels=out_dim,
                               kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=out_dim, out_channels=out_dim,
                               kernel_size=3, stride=1, padding=1)

    def forward(self, input):
        out = self.conv1(input)
        out = F.leaky_relu(out, 0.2)
        out = self.conv2(out)
        out = F.leaky_relu(out, 0.2)
        return out


class FNetBlock(nn.Module):
    def __init__(self, in_dim, out_dim, typ):
        super(FNetBlock, self).__init__()
        self.convleaky = ConvLeaky(in_dim, out_dim)
        if typ == "maxpool":
            self.final = nn.MaxPool2d(kernel_size=2)

        elif typ == "bilinear":
            self.final = nn.Upsample(scale_factor=2, mode='bilinear')
        else:
            raise Exception('typ does not match any of maxpool or bilinear')

    def forward(self, input):
        out = self.convleaky(input)
        out = self.final(out)
        return out


class SRNet(nn.Module):
    def __init__(self, in_dim, out_dim, SR_factor):
        '''scaling = 2^sc_factor'''
        super(SRNet, self).__init__()
        scaling_factor = int(np.log2(SR_factor))
        self.inputConv = nn.Conv2d(in_channels=in_dim, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.ResBlocks = nn.Sequential(*[ResBlock(64) for i in range(3)])

        deconv_layers = []
        for _ in range(scaling_factor):
            deconv_layers.append(nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1))
            deconv_layers.append(nn.ReLU(inplace=True))
        self.deconv = nn.Sequential(*deconv_layers)

        self.outputConv = nn.Conv2d(in_channels=64, out_channels=out_dim, kernel_size=3, stride=1, padding=1)
        # self.dropout = nn.Dropout(p=0.5)

    def forward(self, input):
        out = self.inputConv(input)
        out = self.ResBlocks(out)
        out = self.deconv(out)
        out = self.outputConv(out)
        #out = self.dropout(out)
        return out



class FNet(nn.Module):
    def __init__(self, in_dim, out_chans=2):
        super(FNet, self).__init__()
        self.convPool1 = FNetBlock(in_dim, 32, typ="maxpool")
        self.convPool2 = FNetBlock(32, 64, typ="maxpool")
        self.convPool3 = FNetBlock(64, 128, typ="maxpool")
        self.convBinl1 = FNetBlock(128, 256, typ="bilinear")
        self.convBinl2 = FNetBlock(256, 128, typ="bilinear")
        self.convBinl3 = FNetBlock(128, 64, typ="bilinear")
        self.seq = nn.Sequential(self.convPool1, self.convPool2, self.convPool3,
                                 self.convBinl1, self.convBinl2, self.convBinl3)
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=out_chans, kernel_size=3, stride=1, padding=1)

    def forward(self, input):
        out = self.seq(input)
        out = self.conv1(out)
        out = F.leaky_relu(out, 0.2)
        out = self.conv2(out)
        self.out = F.relu(out)
        self.out.retain_grad()
        return self.out


# please ensure that input is (batch_size, depth, height, width)
# courtesy to Hung Nguyen at https://gist.github.com/jalola/f41278bb27447bed9cd3fb48ec142aec.
class SpaceToDepth(nn.Module):
    def __init__(self, block_size):
        super(SpaceToDepth, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size * block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, s_height, s_width, s_depth) = output.size()
        d_depth = s_depth * self.block_size_sq
        d_width = int(s_width / self.block_size)
        d_height = int(s_height / self.block_size)
        t_1 = output.split(self.block_size, 2)
        stack = [t_t.reshape(batch_size, d_height, d_depth) for t_t in t_1]
        output = torch.stack(stack, 1)
        output = output.permute(0, 2, 1, 3)
        output = output.permute(0, 3, 1, 2)
        return output


# please ensure that lr_height and lr_width must be a multiple of 8.
class FRVSR(nn.Module):
    def __init__(self, batch_size, lr_height, lr_width, SR_factor, Channel_num):
        super(FRVSR, self).__init__()
        self.SRFactor = SR_factor
        self.Channel_num = Channel_num
        self.width = lr_width
        self.height = lr_height
        self.batch_size = batch_size
        self.fnet = FNet(2*Channel_num)
        self.todepth = SpaceToDepth(self.SRFactor)
        self.srnet = SRNet(SR_factor * SR_factor * Channel_num + Channel_num, Channel_num, SR_factor)  # 5 is channel number

    # make sure to call this before every batch train.
    def init_hidden(self, device):
        self.lastLrImg = torch.zeros([self.batch_size, self.Channel_num, self.height, self.width]).to(device)
        self.EstHrImg = torch.zeros([self.batch_size, self.Channel_num, self.height * self.SRFactor, self.width * self.SRFactor]).to(device)
        height_gap = 2 / (self.height - 1)
        width_gap = 2 / (self.width - 1)
        height, width = torch.meshgrid([torch.arange(-1, 1+height_gap, height_gap), torch.arange(-1, 1+width_gap, width_gap)], indexing='ij')
        self.lr_identity = torch.stack([width, height]).to(device)

        height_gap = 2 / (self.height * self.SRFactor - 1)
        width_gap = 2 / (self.width * self.SRFactor - 1)
        height, width = torch.meshgrid([torch.arange(-1, 1+height_gap, height_gap), torch.arange(-1, 1+width_gap, width_gap)], indexing='ij')
        self.hr_identity = torch.stack([width, height]).to(device)

    # useless debug info
    '''
    prvs = img_as_ubyte(self.lastLrImg[0].permute(1,2,0).detach().numpy())
    next = img_as_ubyte(input[0].permute(1,2,0).detach().numpy())
    prvs = cv2.cvtColor(prvs, cv2.COLOR_BGR2GRAY)
    next = cv2.cvtColor(next, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flow[...,0] /= flow.shape[1]
    flow[...,1] /= flow.shape[0]
    flow_to_use = flow
    flow = torch.unsqueeze(torch.tensor(flow).permute(2,0,1), 0)
    flow_len = np.expand_dims(np.sqrt((flow_to_use[..., 0] ** 2 + flow_to_use[..., 1] ** 2)), 2)
    flow_to_use /= flow_len

    print(flow)
    self.EstLrImg = F.grid_sample(self.lastLrImg, flow.permute(0, 2, 3, 1))

    self.EstLrImg = F.grid_sample(self.lastLrImg, torch.unsqueeze(torch.tensor(flow_to_use), 0))

    self.EstLrImg = input

    self.EstLrImg = F.grid_sample(self.lastLrImg, torch.unsqueeze(torch.tensor(flow), 0))
    '''

    # x is a 4-d tensor of shape N×C×H×W
    def forward(self, input):
        def trunc(tensor):
            # tensor = tensor.clone()
            tensor[tensor < 0] = 0
            tensor[tensor > 1] = 1
            return tensor
        
        # 数字均为test本文件的结果
        preflow = torch.cat((input, self.lastLrImg), dim=1) # [4, 10, 16, 16] = cat([4, 5, 16, 16], [4, 5, 16, 16])
        flow = self.fnet(preflow) # [4, 2, 16, 16] = F_net([4, 10, 16, 16])
        # flow += self.lr_identity
        relative_place = flow + self.lr_identity # [4, 2, 16, 16] = [4, 2, 16, 16] + [2, 16, 16]
        # debug info goes here
        self.EstLrImg = F.grid_sample(self.lastLrImg, relative_place.permute(0, 2, 3, 1), align_corners=True) # [4, 5, 16, 16]
        # self.EstLrImg = trunc(self.EstLrImg)
        relative_place_HR = F.interpolate(relative_place, scale_factor=self.SRFactor, mode="bilinear") # [4, 2, 128, 128]
        # relative_placeNCHW = torch.unsqueeze(self.hr_identity, dim=0)
        afterWarp = F.grid_sample(self.EstHrImg.detach(), relative_place_HR.permute(0, 2, 3, 1), align_corners=True) # [4, 5, 128, 128]
        self.afterWarp = afterWarp  # for debugging, should be removed later. # [4, 5, 128, 128]
        depthImg = self.todepth(afterWarp) # [4, 320, 16, 16] = [4, 5, 128, 128]
        srInput = torch.cat((input, depthImg), dim=1) # [4, 325, 16, 16] = cat([4, 5, 16, 16], [4, 320, 16, 16])
        estImg = self.srnet(srInput) # [4, 5, 128, 128] = SR_net([4, 325, 16, 16])
        self.lastLrImg = input # [4, 5, 16, 16]
        self.EstHrImg = estImg # [4, 5, 128, 128]
        #self.EstHrImg = trunc(self.EstHrImg)
        self.EstHrImg.retain_grad()
        return self.EstHrImg, self.EstLrImg, afterWarp


# please ensure that lr_height and lr_width must be a multiple of 8.
class PI_FRVSR(nn.Module):
    def __init__(self, batch_size, lr_height, lr_width, SR_factor, Channel_num):
        super(PI_FRVSR, self).__init__()
        self.dt = 1
        self.SRFactor = SR_factor
        self.Channel_num = Channel_num
        self.width = lr_width
        self.height = lr_height
        self.batch_size = batch_size
        self.fnet = FNet(Channel_num, Channel_num)
        self.todepth = SpaceToDepth(self.SRFactor)
        self.srnet = SRNet(Channel_num*2, Channel_num, SR_factor)  # 5 is channel number

    # make sure to call this before every batch train.
    def init_hidden(self, device, batch_size=None):
        if batch_size != None:
            self.batch_size = batch_size

        self.lastLrImg = torch.zeros([self.batch_size, self.Channel_num, self.height, self.width]).to(device)
        self.lastHrImg = torch.zeros([self.batch_size, self.Channel_num, self.height * self.SRFactor, self.width * self.SRFactor]).to(device)
        
        height_gap = 2 / (self.height - 1)
        width_gap = 2 / (self.width - 1)
        height, width = torch.meshgrid([torch.arange(-1, 1+height_gap, height_gap), torch.arange(-1, 1+width_gap, width_gap)], indexing='ij')
        self.lr_identity = torch.stack([width, height]).to(device)

        height_gap = 2 / (self.height * self.SRFactor - 1)
        width_gap = 2 / (self.width * self.SRFactor - 1)
        height, width = torch.meshgrid([torch.arange(-1, 1+height_gap, height_gap), torch.arange(-1, 1+width_gap, width_gap)], indexing='ij')
        self.hr_identity = torch.stack([width, height]).to(device)


    # x is a 4-d tensor of shape N×C×H×W
    def forward(self, input):
        input_Lr_comb = torch.cat((input, self.lastLrImg), dim=1) 
        Sr_diff_T = self.srnet(input_Lr_comb) # upsampling and output dvar/dt 
        Sr_star = Sr_diff_T * self.dt + self.lastHrImg.detach()
        estImg = self.fnet(Sr_star) # output SR results
        self.lastLrImg = input 
        self.lastHrImg = estImg 
        self.lastHrImg.retain_grad()
        return self.lastHrImg, Sr_diff_T


# class Loss(nn.Module):
#     def __init__(self):
#         super(Loss, self).__init__()
#         vgg = vgg16(weights=VGG16_Weights.DEFAULT)
#         loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
#         for param in loss_network.parameters():
#             param.requires_grad = False
#         self.loss_network = loss_network
#         self.mse_loss = nn.MSELoss()
#         self.tv_loss = TVLoss()

#     def forward(self, out_images, target_images):
#         # Adversarial Loss
#         # adversarial_loss = torch.mean(1 - out_labels)
#         # Perception Loss
#         perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
#         # Image Loss
#         image_loss = self.mse_loss(out_images, target_images)
#         # TV Loss
#         tv_loss = self.tv_loss(out_images)
#         return image_loss + 0.006 * perception_loss + 2e-8 * tv_loss


class GeneratorLoss_FRVSR(nn.Module):
    def __init__(self):
        super(GeneratorLoss_FRVSR, self).__init__()
        vgg = vgg16(weights=VGG16_Weights.DEFAULT)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()
        self.mass_eq = mass_eq()

    def forward(self, out_labels, hr_est, hr_img, lr_est, lr_img, dvar_dt, idx):
        # Adversarial Loss
        adversarial_loss = -torch.mean(out_labels)
        # Perception Loss
        # 计算会出错，因为vgg只有三通道
        # perception_loss = self.mse_loss(self.loss_network(hr_est), self.loss_network(hr_img))
        perception_loss = torch.tensor([0.0]).cuda()

        # Image Loss
        image_loss = self.mse_loss(hr_est, hr_img)
        # TV Loss
        tv_loss = self.tv_loss(hr_est)
        # flow loss
        if idx != 0:
            flow_loss = self.mse_loss(lr_est, lr_img)
        else:
            flow_loss = torch.tensor([0.0]).cuda()
        # physics loss
        physics_loss, residual_t = self.mass_eq(hr_est.detach(), dvar_dt)
        # physics_loss = torch.tensor([0.0]).cuda()

        return image_loss, adversarial_loss, physics_loss, tv_loss, flow_loss, residual_t

class GeneratorLoss_PI_FRVSR(nn.Module):
    def __init__(self):
        super(GeneratorLoss_PI_FRVSR, self).__init__()
        vgg = vgg16(weights=VGG16_Weights.DEFAULT)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()
        self.bce_loss = nn.BCELoss()
        self.mass_eq = mass_eq()

    def forward(self, out_labels, hr_est, hr_img, dvar_dt, idx):
        # Adversarial Loss
        adversarial_loss = self.bce_loss(out_labels, torch.ones_like(out_labels))
        # Perception Loss
        # 计算会出错，因为vgg只有三通道
        # perception_loss = self.mse_loss(self.loss_network(hr_est), self.loss_network(hr_img))

        # Image Loss
        image_loss = self.mse_loss(hr_est, hr_img)

        # TV Loss
        tv_loss = self.tv_loss(hr_est)

        # physics loss
        physics_loss_norm, residual_t, total_residual = self.mass_eq(hr_est.detach(), dvar_dt)

        return image_loss, adversarial_loss, physics_loss_norm, tv_loss, residual_t, total_residual



class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]

class mass_eq(nn.Module):
    def __init__(self):
        super(mass_eq, self).__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, var, dvar_dt, X_len=0.2, Y_len=0.4):
        rho = var[:, 0, :, :].unsqueeze(1)*20
        u = (var[:, 1, :, :].unsqueeze(1)+200)*1500
        v = (var[:, 2, :, :].unsqueeze(1)+500)*1500
        p = var[:, 3, :, :].unsqueeze(1)*1e7
        T = var[:, 4, :, :].unsqueeze(1)*3000
        dx = X_len / rho.shape[2] 
        dy = Y_len / rho.shape[1] 

        drho_dt = dvar_dt[:, 0, :, :].unsqueeze(1)*20
        du_dt = dvar_dt[:, 1, :, :].unsqueeze(1)*1500-200
        dv_dt = dvar_dt[:, 2, :, :].unsqueeze(1)*1500-500
        dp_dt = dvar_dt[:, 3, :, :].unsqueeze(1)*1e7
        dT_dt = dvar_dt[:, 4, :, :].unsqueeze(1)*3000

        residual_t = dfdx(rho*u, dx) + dfdy(rho*v, dy)
        residual = drho_dt - residual_t
        residual_norm = residual/torch.max(abs(residual))
        return self.mse(residual_norm, torch.zeros_like(residual)), residual_t, residual

    
def swish(x):
    return x * torch.sigmoid(x)

def residual_trans(x):
    x_upper_bound = torch.max(x)
    x_lower_bound = torch.min(x)
    if x_upper_bound > abs(x_lower_bound):
        return x/(2*x_upper_bound) + 0.5
    else:
        return x/(2*abs(x_lower_bound)) + 0.5

class Discriminator(nn.Module):
    def __init__(self, input_channel):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, 64, 3, stride=1, padding=1)

        self.conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, 3, stride=2, padding=1)
        self.bn8 = nn.BatchNorm2d(512)

        # Replaced original paper FC layers with FCN
        self.conv9 = nn.Conv2d(512, 1, 1, stride=1, padding=1)

    def forward(self, x):
        x = swish(self.conv1(x.to(torch.float32)))

        # x = swish(self.bn2(self.conv2(x)))
        x = swish(self.bn3(self.conv3(x)))
        # x = swish(self.bn4(self.conv4(x)))
        x = swish(self.bn5(self.conv5(x)))
        # x = swish(self.bn6(self.conv6(x)))
        x = swish(self.bn7(self.conv7(x)))
        # x = swish(self.bn8(self.conv8(x)))

        x = self.conv9(x)
        return torch.sigmoid(F.avg_pool2d(x, x.size()[2:])).view(x.size()[0], -1)

#
# if __name__ == "__main__":
#     g_loss = GeneratorLoss()
#     print(g_loss)

# class FRVSR_Criterion(torch.autograd.Function):
#     def __init__(self):
#         super(FRVSR_Criterion, self).__init__()
#
#     def forward(self, lr_est, lr_img, hr_est, hr_img):
#         #= input[0], input[1], input[2], input[3]
#         assert (lr_est.shape == lr_img.shape)
#         assert (hr_est.shape == hr_img.shape)
#         return nn.MSELoss(lr_est, lr_img) + nn.MSELoss(hr_est, hr_img)

# run tests make sure that output is correct.

class TestFRVSR(unittest.TestCase):
    def setUp(self):
        self.Channel_num = 5
        self.Scaling_factor = 8
        self.L_width = 16
        self.L_higth = 16

    def testResBlock(self):
        block = ResBlock(3)
        input = torch.rand(2, 3, 64, 112)
        output = block(input)
        self.assertEqual(input.shape, output.shape)

    def testConvLeaky(self):
        block = ConvLeaky(3, 32)
        input = torch.rand(2, 3, 64, 112)
        output = block(input)
        self.assertEqual(output.shape, torch.empty(2, 32, 64, 112).shape)

    def testFNetBlockMaxPool(self):
        block = FNetBlock(3, 32, "maxpool")
        input = torch.rand(2, 3, 64, 112)
        output = block(input)
        self.assertEqual(output.shape, torch.empty(2, 32, 32, 56).shape)

    def testFNetBlockInterPolate(self):
        block = FNetBlock(3, 32, "bilinear")
        input = torch.rand(2, 3, 32, 56)
        output = block(input)
        self.assertEqual(output.shape, torch.empty(2, 32, 64, 112).shape)

    def testSRNet(self):
        block = SRNet(self.Channel_num, self.Channel_num, self.Scaling_factor)
        input = torch.rand(2, self.Channel_num, self.L_width, self.L_higth)
        output = block(input)
        self.assertEqual(output.shape, torch.empty(2, self.Channel_num, self.L_width*self.Scaling_factor, self.L_higth*self.Scaling_factor).shape)


    def testFNet(self):
        block = FNet(self.Channel_num)
        input = torch.rand(2, 2*self.Channel_num, self.L_width, self.L_higth)
        output = block(input)
        self.assertEqual(output.shape, torch.empty(2, 2, self.L_width, self.L_higth).shape)

    def testFRVSR(self):
        H = self.L_higth
        W = self.L_width
        BC = 4
        block = FRVSR(BC, H, W, SR_factor=self.Scaling_factor, Channel_num=self.Channel_num)
        input = torch.rand(BC, self.Channel_num, H, W)
        block.init_hidden("cpu")

        output1, output2 = block(input)
        self.assertEqual(output1.shape, torch.empty(4, self.Channel_num, H * self.Scaling_factor, W * self.Scaling_factor).shape)
        self.assertEqual(output2.shape, torch.empty(4, self.Channel_num, H, W).shape)
    
    def testDiscriminator(self):
        block = Discriminator(self.Channel_num)
        input = torch.rand(2, self.Channel_num, 64, 64)
        output = block(input)
        self.assertEqual(output.shape, torch.empty(2, 1).shape)

    # def testCriterion(self):
    #     H = 16
    #     W = 16
    #     input = torch.rand(7, 4, 3, H, W)
    #     output = torch.rand(4, 3, H * 4, W * 4)
    #     criterion = FRVSR_Criterion()
    #     self.assertIsInstance(criterion(input, input, output, output), type(0.1))


if __name__ == '__main__':
    unittest.main()

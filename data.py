#%%
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import argparse
from visual import *
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
from loss import *
import random
import numpy as np
from PIL import Image
import os
import random
import matplotlib
import matplotlib.pyplot as plt

class RDC_Dataset(Dataset):
    def __init__(self, path, flag='train', 
                 bi=False, bi_threshold=0.07,
                 depth=32,
                 LR_size=(20, 20), HR_size=(160, 160),
                 SNR=1000):
        
        self.flag = flag
        self.bi = bi
        self.depth = depth
        self.path = os.path.join(path, flag)
        self.data_path_list = os.listdir(self.path)
        self.data_path_list.sort(key=lambda x:int(x[:-3]))
        self.threshold = bi_threshold
        self.LR_size = LR_size
        self.HR_size = HR_size
        self.scale_factor = 2**depth - 1  # 避免重复计算
        self.SNR = SNR

    def __getitem__(self, index):
        # 根据索引返回数据[rho, u, v, p, T]
        # 输出的维度[batch, channel(variables), w, h]
        
        data = self.loader(self.data_path_list[index])

        norm = [10, 1500, 1500, 1e7, 3000]
        bais = [0, 200, 500, 0, 0]
        for i in range(5):
            data[i, :, :] = (data[i, :, :].add(bais[i])).div(norm[i])

        var_t_HR = [self.HR_transform(x) for x in data]
        var_t_LR = [self.LR_transform(x) for x in data]
        return torch.stack(var_t_HR, dim=0), torch.stack(var_t_LR, dim=0)
    
    def __len__(self):
        # 返回数据的长度
        return len(self.data_path_list)

    def loader(self, path):
        # 假如从 csv_paths 中加载数据，可能要遍历文件夹读取文件等，这里忽略
        # 可以拆分训练和验证集并返回train_X, train_Y, valid_X, valid_Y
        data = torch.load(os.path.join(self.path, path))
        return data

    def LR_transform(self, data):
        data = F.interpolate(data.view(1, 1, *data.shape), size=self.LR_size, mode='nearest')
        if self.bi:
            data = torch.where(data < self.threshold, torch.zeros_like(data), torch.ones_like(data))
        data = data * self.scale_factor
        data = data / self.scale_factor
        data = data.view(*data.shape[2:])

        if self.SNR < 1000:
            for i in range(5):
                noise_ratio = torch.max(data) * (10 ** (- self.SNR * 0.1))
                data = data + noise_ratio * torch.randn_like(data)
        return data

    def HR_transform(self, data):
        data = F.interpolate(data.view(1, 1, *data.shape), size=self.HR_size, mode='nearest')
        data = data.view(*data.shape[2:])
        return data



def train_load_data(path, flag, opt):
    LR = (opt.LR_imageSize, opt.LR_imageSize)
    HR = (opt.LR_imageSize*(2**opt.upSampling), opt.LR_imageSize*(2**opt.upSampling))
    train_data = RDC_Dataset(path, flag=flag, LR_size=LR, HR_size=HR, bi=opt.bi, bi_threshold=opt.bi_th, depth=opt.depth, SNR=opt.SNR)

    train_loader = DataLoader(train_data, batch_size=opt.batchSize, shuffle=False, drop_last=True, num_workers=1)

    return train_loader

def test_load_data(path, flag, opt):
    LR = (opt.LR_imageSize, opt.LR_imageSize)
    HR = (opt.LR_imageSize*(2**opt.upSampling), opt.LR_imageSize*(2**opt.upSampling))
    test_data = RDC_Dataset(path, flag=flag, LR_size=LR, HR_size=HR, bi=opt.bi, bi_threshold=opt.bi_th, depth=opt.depth, SNR=opt.SNR)

    test_loader = DataLoader(test_data, batch_size=opt.batchSize, shuffle=False, drop_last=False, num_workers=1)

    return test_loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--LR_imageSize', type=int, default=20, help='the low resolution image size')
    parser.add_argument('--upSampling', type=int, default=3, help='low to high resolution scaling factor')
    parser.add_argument('--bi', type=bool, default=False, help='Binary image or not')
    parser.add_argument('--bi_th', type=float, default=0.2, help='Binary threshold')
    parser.add_argument('--depth', type=int, default=32, help='')

    opt = parser.parse_args(args=[])

    path = r'E:/RDE_GAN_HR_dataset/p=7e5_17e5/dataset'
    dataloader = train_load_data(path, 'train1000', opt)
    for i, (var_HR, var_LR) in enumerate(dataloader):
        # 根据索引返回数据[rho, u, v, p, T]
        # 维度[batch, channel(variables), w, h]
        print(i, var_HR.shape, var_LR.shape)
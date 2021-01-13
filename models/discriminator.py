from models.candidate_cell import *

import torch.nn as nn
import torch.nn.functional as F

def _downsample(x,n):# 1.4.1確認済
    # Downsample (Mean Avg Pooling with 2x2 kernel)
    for _ in range(n):
        x = nn.AvgPool2d(kernel_size=2)(x)
    return x

class DisBlock(nn.Module):# 1.4.1確認済
    def __init__(self,in_channels, out_channels):
        super(DisBlock,self).__init__()
        self.cell5 = NormalCell(in_channels, in_channels, norm=True)
        self.cell3 = NormalCell(in_channels, in_channels, norm=True)
        self.cell1 = DownCell(in_channels, out_channels)

    def forward(self,designs,h,skip_h):
        h = self.cell5(h, designs[5])
        h = self.cell3(h, designs[3])
        h = self.cell1(h, designs[1])
        return h + skip_h

class Discriminator_Net(nn.Module):# 1.4.1確認済
    def __init__(self,option):
        super(Discriminator_Net,self).__init__()
        self.channels = option.df_dim
        self.blockRGB = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(            3, self.channels, kernel_size=3,padding=1)),
            nn.ReLU(),
            nn.utils.spectral_norm(nn.Conv2d(self.channels, self.channels, kernel_size=3,padding=1)),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2)
        )
        self.block2 = DisBlock(self.channels, self.channels)
        self.block1 = DisBlock(self.channels, self.channels)
        self.block0 = DisBlock(self.channels, self.channels)
        if option.loss == 'BCE':
            self.fc = nn.Sequential(
                nn.utils.spectral_norm(nn.Linear(self.channels, 1, bias=False)),
                nn.Sigmoid()
            )
        elif option.loss == 'Hinge':
            self.fc = nn.Sequential(
                nn.utils.spectral_norm(nn.Linear(self.channels, 1, bias=False)),
            )
        self.use_skip = option.use_skip
        if self.use_skip:
            keys = [(2,2), (2,1), (2,0), (1,1), (1,0), (0,0)]
            self.skip_ops = nn.ModuleDict(
                key: nn.utils.spectral_norm(nn.Conv2d(self.channels, self.channels, kernel_size=1)) for key in keys
            )
    
    def forward(self,x,design):
        skip_h0, skip_h1, skip_h2 = 0.0, 0.0, 0.0
        h = self.blockRGB(x)
        if 3 <= len(design):
            if self.use_skip:
                if design[2][7][2]: skip_h2 += self.skip_ops[(2,2)](_downsample(h,1))
                if design[2][7][1]: skip_h1 += self.skip_ops[(2,1)](_downsample(h,2))
                if design[2][7][0]: skip_h0 += self.skip_ops[(2,0)](_downsample(h,3))
            h = self.block2(design[2],h,skip_h2)
        if 2 <= len(design):
            if self.use_skip:
                if design[1][7][1]: skip_h1 += self.skip_ops[(1,1)](_downsample(h,1))
                if design[1][7][0]: skip_h0 += self.skip_ops[(1,0)](_downsample(h,2))
            h = self.block1(design[1],h,skip_h1)
        if 1 <= len(design):
            if self.use_skip:
                if design[0][7][0]: skip_h0 += self.skip_ops[(0,0)](_downsample(h,1))
            h = self.block0(design[0],h,skip_h0)
        h0 = F.relu(h).sum(2).sum(2)
        z = self.fc(h0)
        return z
from models.candidate_cell import *

import torch.nn as nn
import torch.nn.functional as F

def _downsample(x):# 1.4.1確認済
    # Downsample (Mean Avg Pooling with 2x2 kernel)
    return nn.AvgPool2d(kernel_size=2)(x)

class DisBlock(nn.Module):# 1.4.1確認済
    def __init__(self,in_channels, out_channels, use_skip, num_skip_in):
        super(DisBlock,self).__init__()
        self.cell5 = NormalCell(in_channels, in_channels, norm=True)
        self.cell3 = NormalCell(in_channels, in_channels, norm=True)
        self.cell1 = DownCell(in_channels, out_channels)
        self.use_skip = use_skip
        if use_skip:
            self.skip_in_ops = nn.ModuleList(
                [nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1)) for _ in range(num_skip_in)]
            )
    def forward(self,designs,h,skip_ft):
        h = self.cell5(h, designs[5])
        h = self.cell3(h, designs[3])
        h = self.cell1(h, designs[1])
        if self.use_skip:
            for flag_input, ft, skip_in_op in zip(designs[7], skip_ft, self.skip_in_ops):
                if flag_input:
                    h += skip_in_op(ft)
        return h

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
        self.block3 = DisBlock(self.channels, self.channels, option.use_skip, num_skip_in=1)
        self.block2 = DisBlock(self.channels, self.channels, option.use_skip, num_skip_in=2)
        self.block1 = DisBlock(self.channels, self.channels, option.use_skip, num_skip_in=3)
        if option.loss == 'BCE':
            self.fc = nn.Sequential(
                nn.utils.spectral_norm(nn.Linear(self.channels, 1, bias=False)),
                nn.Sigmoid()
            )
        elif option.loss == 'Hinge':
            self.fc = nn.Sequential(
                nn.utils.spectral_norm(nn.Linear(self.channels, 1, bias=False)),
            )
    
    def forward(self,x,design):
        h = self.blockRGB(x)
        skip_ft = [h,]
        if 3 <= len(design):
            skip_ft = [_downsample(ft) for ft in skip_ft]
            h = self.block3(design[2],h,skip_ft)
            skip_ft.append(h)
        if 2 <= len(design):
            skip_ft = [_downsample(ft) for ft in skip_ft]
            h = self.block2(design[1],h,skip_ft)
            skip_ft.append(h)
        if 1 <= len(design):
            skip_ft = [_downsample(ft) for ft in skip_ft]
            h = self.block1(design[0],h,skip_ft)
        h0 = F.relu(h).sum(2).sum(2)
        z = self.fc(h0)
        return z
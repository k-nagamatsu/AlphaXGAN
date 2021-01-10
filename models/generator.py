from models.candidate_cell import *

import torch.nn as nn
import torch.nn.functional as F

class GenBlock(nn.Module):# 1.4.1 確認済
    def __init__(self,in_channels, out_channels, use_skip, skip_up_mode, num_skip_in):
        super(GenBlock,self).__init__()
        self.cell0 = UpCell(in_channels, out_channels)
        self.cell2 = NormalCell(out_channels, out_channels)
        self.cell4 = NormalCell(out_channels, out_channels)
        self.use_skip = use_skip
        self.skip_up_mode = skip_up_mode
        if use_skip:
            self.skip_in_ops = nn.ModuleList(
                [nn.Conv2d(in_channels, out_channels, kernel_size=1) for _ in range(num_skip_in)]
            )
    def forward(self,designs,h,skip_ft):
        h = self.cell0(h, designs[0])
        h = self.cell2(h, designs[2])
        h = self.cell4(h, designs[4])
        if self.use_skip:
            _, _, ht, wt = h.size()
            for flag_input, ft, skip_in_op in zip(designs[6], skip_ft, self.skip_in_ops):
                if flag_input:
                    if self.skip_up_mode == 'nearest':
                        h += skip_in_op(F.interpolate(ft, size=(ht, wt), mode=self.skip_up_mode))
                    elif self.skip_up_mode == 'bilinear':
                        h += skip_in_op(F.interpolate(ft, size=(ht, wt), mode=self.skip_up_mode,align_corners=False))
        return h

class Generator_Net(nn.Module):# 1.4.1 確認済
    def __init__(self,option):
        super(Generator_Net,self).__init__()
        self.channels = option.gf_dim
        self.bottom_width = option.bottom_width
        self.fc = nn.Linear(option.latent_dim, (self.bottom_width ** 2) * self.channels)
        self.block1 = GenBlock(self.channels, self.channels, option.use_skip, 'nearest' , num_skip_in=1)
        self.block2 = GenBlock(self.channels, self.channels, option.use_skip, 'bilinear', num_skip_in=2)
        self.block3 = GenBlock(self.channels, self.channels, option.use_skip, 'nearest' , num_skip_in=3)
        self.blockRGB = nn.Sequential(
            nn.BatchNorm2d(self.channels),
            nn.ReLU(),
            nn.Conv2d(self.channels, 3, 3, 1, 1),
            nn.Tanh()
        )
    def forward(self,x,design):
        h0 = self.fc(x).view(-1, self.channels, self.bottom_width, self.bottom_width)
        h1 = self.block1(design[0],h0,(h0,))
        if len(design) == 1:
            return self.blockRGB(h1)
        h2 = self.block2(design[1],h1,(h0,h1))
        if len(design) == 2:
            return self.blockRGB(h2)
        h3 = self.block3(design[2],h2,(h0,h1,h2))
        if len(design) == 3:
            return self.blockRGB(h3)
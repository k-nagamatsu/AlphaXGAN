import torch.nn as nn
import torch.nn.functional as F

class NormalCell(nn.Module):# 1.4.1確認済
    def __init__(self, in_channels, out_channels,norm=False):
        super(NormalCell, self).__init__()
        '''('conv_1x1_1','conv_3x3_1','conv_5x5_1','conv_3x3_2','conv_5x5_2')'''
        self.conv11 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1)
        self.conv31 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv51 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, dilation=1)
        self.conv32 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv52 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=4, dilation=2)
        if norm:
            self.conv11 = nn.utils.spectral_norm(self.conv11)
            self.conv31 = nn.utils.spectral_norm(self.conv31)
            self.conv51 = nn.utils.spectral_norm(self.conv51)
            self.conv32 = nn.utils.spectral_norm(self.conv32)
            self.conv52 = nn.utils.spectral_norm(self.conv52)

    def forward(self,h,design):
        h = F.relu(h)
        if   design == 'conv_1x1_1':
            h = self.conv11(h)
        elif design == 'conv_3x3_1':
            h = self.conv31(h)
        elif design == 'conv_5x5_1':
            h = self.conv51(h)
        elif design == 'conv_3x3_2':
            h = self.conv32(h)
        elif design == 'conv_5x5_2':
            h = self.conv52(h)
        return h

class UpCell(nn.Module):# 1.4.1確認済
    def __init__(self, in_channels, out_channels):
        super(UpCell, self).__init__()
        '''('deconv','nearest','bilinear')'''
        self.deconv = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self,h,design):
        h = F.relu(h)
        if design == 'deconv':
            h = self.deconv(h)
        elif design == 'nearest':
            h = F.interpolate(h, scale_factor=2, mode=design)
        elif design == 'bilinear':
            h = F.interpolate(h, scale_factor=2, mode=design, align_corners=False)
        h = self.conv(h)
        return h

class DownCell(nn.Module):# 1.4.1確認済
    def __init__(self, in_channels, out_channels):
        super(DownCell, self).__init__()
        '''('ave','max','conv_3x3_1','conv_5x5_1','conv_3x3_2','conv_5x5_2')'''
        self.ave_pool = nn.AvgPool2d(kernel_size=2)
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.conv31 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, dilation=1)
        self.conv51 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=2, padding=2, dilation=1)
        self.conv32 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=2, dilation=2)
        self.conv52 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=2, padding=4, dilation=2)

        self.conv31 = nn.utils.spectral_norm(self.conv31)
        self.conv51 = nn.utils.spectral_norm(self.conv51)
        self.conv32 = nn.utils.spectral_norm(self.conv32)
        self.conv52 = nn.utils.spectral_norm(self.conv52)

    def forward(self,h,design):
        h = F.relu(h)
        if design == 'ave':
            h = self.ave_pool(h)
        elif design == 'max':
            h = self.max_pool(h)
        elif design == 'conv_3x3_1':
            h = self.conv31(h)
        elif design == 'conv_5x5_1':
            h = self.conv51(h)
        elif design == 'conv_3x3_2':
            h = self.conv32(h)
        elif design == 'conv_5x5_2':
            h = self.conv52(h)
        return h
import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, bath_normal=False):
        super().__init__()
        in_channels = int(in_channels)
        out_channels = int(out_channels)
        channels = int(out_channels / 2) if in_channels <= out_channels else int(in_channels / 2)
        layers = [
            nn.Conv3d(in_channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv3d(channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
        ]
        if bath_normal:
            layers.insert(1, nn.BatchNorm3d(channels))
            layers.insert(len(layers) - 1, nn.BatchNorm3d(out_channels))
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)


class DownSampling(nn.Module):
    def __init__(self, in_channels, out_channels, batch_normal=False):
        super().__init__()
        self.maxpool_to_conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1)),
            DoubleConv(in_channels, out_channels, batch_normal),
        )

    def forward(self, x):
        return self.maxpool_to_conv(x)


class ECAAttention(nn.Module):
    def __init__(self, channels, b=1, gamma=2):
        super().__init__()
        self.channels = channels
        k = int(abs((math.log(channels, 2) + b) / gamma))
        self.kernel_size = k + 1 if k % 2 == 0 else k
        self.conv = nn.Conv1d(1, 1, kernel_size=self.kernel_size, padding=self.kernel_size // 2, bias=False)

    def forward(self, x):
        y = F.adaptive_avg_pool3d(x, (1, 1, 1))
        y = y.squeeze(-1).squeeze(-1).transpose(1, 2)
        y = torch.sigmoid(y)
        y = y.transpose(1, 2).unsqueeze(-1).unsqueeze(-1)
        return x * y.expand_as(x)


class UpSampling(nn.Module):
    def __init__(self, in_channels, out_channels, batch_normal=False, trilinear=True):
        super().__init__()
        if trilinear:
            self.up = nn.Upsample(scale_factor=(2, 2, 1), mode="trilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(int(in_channels + in_channels / 2), out_channels, batch_normal)

    def forward(self, inputs1, inputs2):
        inputs1 = self.up(inputs1)
        outputs = torch.cat([inputs1, inputs2], dim=1)
        outputs = self.conv(outputs)
        return outputs


class LastConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels, num_classes=32, batch_normal=False, trilinear=True):
        super().__init__()
        self.in_channels = in_channels
        self.batch_normal = batch_normal
        self.trilinear = trilinear
        self.inputs = DoubleConv(in_channels, 32, self.batch_normal)
        self.down_1 = DownSampling(32, 64, self.batch_normal)
        self.up_3 = UpSampling(64, 32, self.batch_normal, self.trilinear)
        self.outputs = LastConv(32, num_classes)

    def forward(self, x):
        x1 = self.inputs(x)
        x2 = self.down_1(x1)
        x7 = self.up_3(x2, x1)
        x = self.outputs(x7)
        return x

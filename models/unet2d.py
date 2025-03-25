import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
from models.BaseModelClass import BaseModel
import numpy as np

def x2conv(in_channels, out_channels, inner_channels=None):
    inner_channels = out_channels // 2 if inner_channels is None else inner_channels
    down_conv = nn.Sequential(
        nn.Conv2d(in_channels, inner_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(inner_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(inner_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True))
    return down_conv


class encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(encoder, self).__init__()
        self.down_conv = x2conv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, ceil_mode=True)

    def forward(self, x):
        x = self.down_conv(x)
        x = self.pool(x)
        return x


class decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(decoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.up_conv = x2conv(in_channels, out_channels)

    def forward(self, x_copy, x, interpolate=True):
        x = self.up(x)

        if (x.size(2) != x_copy.size(2)) or (x.size(3) != x_copy.size(3)):
            if interpolate:
                # Iterpolating instead of padding
                x = F.interpolate(x, size=(x_copy.size(2), x_copy.size(3)),
                                  mode="bilinear", align_corners=True)
            else:
                # Padding in case the incomping volumes are of different sizes
                diffY = x_copy.size()[2] - x.size()[2]
                diffX = x_copy.size()[3] - x.size()[3]
                x = F.pad(x, (diffX // 2, diffX - diffX // 2,
                              diffY // 2, diffY - diffY // 2))

        # Concatenate
        x = torch.cat([x_copy, x], dim=1)
        x = self.up_conv(x)
        return x


class UNet2D(BaseModel):
    def __init__(self, in_channels=3, num_classes=1, encode=False, freeze_bn=False, **_):
        super(UNet2D, self).__init__()

        self.start_conv = x2conv(in_channels, 64)
        self.down1 = encoder(64, 128)
        self.down2 = encoder(128, 256)
        self.down3 = encoder(256, 512)
        self.down4 = encoder(512, 1024)

        self.middle_conv = x2conv(1024, 1024)

        self.up1 = decoder(1024, 512)
        self.up2 = decoder(512, 256)
        self.up3 = decoder(256, 128)
        self.up4 = decoder(128, 64)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        self._initialize_weights()
        # self.outconv = nn.Sequential(
        #     nn.Conv2d(num_classes, 1, kernel_size=1, padding=0),
        #     nn.Sigmoid()
        # )

        if freeze_bn:
            self.freeze_bn()

        self.encode = encode

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def forward(self, x):
        x1 = self.start_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.middle_conv(x5)

        d4 = self.up1(x4, x)
        d3 = self.up2(x3, d4)
        d2 = self.up3(x2, d3)
        d1 = self.up4(x1, d2)

        x = self.final_conv(d1)
        x = torch.sigmoid(x)  # todo
        if self.encode:
            return x, [x5, d4, d3, d2, d1]
        return x

    def get_backbone_params(self):
        # There is no backbone for unet, all the parameters are trained from scratch
        return []

    def get_decoder_params(self):
        return self.parameters()

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()

    def detach_model(self):
        for param in self.parameters():
            param.detach_()

    def ema_update(self, student, ema_decay=0.99, cur_step=None):
        if cur_step is not None:
            ema_decay = min(1 - 1 / (cur_step + 1), ema_decay)
        for t_param, s_param in zip(self.parameters(), student.parameters()):
            t_param.data.mul_(ema_decay).add_(1 - ema_decay, s_param.data)

    def weighted_update(self, teacher1, teacher2, coefficient=0.99, ema_decay=0.99, cur_step=None):
        if cur_step is not None:
            ema_decay = min(1 - 1 / (cur_step + 1), ema_decay)
        for t_param, t1_param, t2_param in zip(self.parameters(), teacher1.parameters(), teacher2.parameters()):
            t_param.data.mul_(ema_decay).add_(coefficient*(1-ema_decay), t1_param.data).add_((1-coefficient)*(1-ema_decay), t2_param.data)

    def test(self):
        pass



"""
-> Unet with a resnet backbone
"""


if __name__ == '__main__':
    # model = torchvision.models.densenet121()
    model = UNet2D(3, 2)
    # print(model)

    input = torch.randn(1, 3, 128, 128)
    out = model(input)
    print(out.shape)  # torch.Size([2, 1, 224, 224])


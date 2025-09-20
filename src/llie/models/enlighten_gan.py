from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import lightning as pl
import numpy as np
from einops import rearrange
import loguru
from typing import Tuple, Optional, Any

from src.llie.utils.config import get_optimizer, get_scheduler
from src.llie.models.utils import pad_tensor, pad_tensor_back


# ========================================================================================================
#                                              Submodules
# ========================================================================================================

class CRB(nn.Module):
    """
    Convolution-LeakyReLU-BatchNorm
    """
    def __init__(self, in_channels: int, mid_channels: int, depth: int):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            *[
                nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1)
                for _ in range(depth - 1)
            ],
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.bn = nn.BatchNorm2d(mid_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.act(self.convs(x)))


class Generator(nn.Module):
    def __init__(self, config):
        super().__init__()

        in_channels = config.get("in_channels", 4)
        base_channels = config.get("base_channels", 32)

        # downsampling for attention map
        self.down1 = nn.MaxPool2d(2)
        self.down2 = nn.MaxPool2d(2)
        self.down3 = nn.MaxPool2d(2)
        self.down4 = nn.MaxPool2d(2)

        # layers for input image
        self.crb1 = CRB(in_channels, base_channels, 2)
        self.pool1 = nn.MaxPool2d(2)
        self.crb2 = CRB(base_channels, base_channels * 2, 2)
        self.pool2 = nn.MaxPool2d(2)
        self.crb3 = CRB(in_channels * 2, base_channels * 4, 2)
        self.pool3 = nn.MaxPool2d(2)
        self.crb4 = CRB(in_channels * 4, base_channels * 8, 2)
        self.pool4 = nn.MaxPool2d(2)

        # fuse input image and attention map
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.deconv5 = nn.Conv2d(base_channels * 16, base_channels * 8, kernel_size=3, padding=1)
        self.deconv4 = nn.Conv2d(base_channels * 8, base_channels * 4, kernel_size=3, padding=1)
        self.deconv3 = nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=3, padding=1)
        self.deconv2 = nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1)

        self.crb5_1 = CRB(base_channels * 8, base_channels * 16, 1)
        self.crb5_2 = CRB(base_channels * 16, base_channels * 16, 1)
        self.crb6 = CRB(base_channels * 16, base_channels * 8, 2)
        self.crb7 = CRB(base_channels * 8, base_channels * 4, 2)
        self.crb8 = CRB(base_channels * 4, base_channels * 2, 2)
        self.crb9_1 = CRB(base_channels * 2, base_channels, 1)
        self.conv9_2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, 3, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, attn_map: torch.Tensor):
        x, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(x)
        attn_map, _, _, _, _ = pad_tensor(attn_map)

        # downsampling for attention map
        attn_map2 = self.down1(attn_map)
        attn_map3 = self.down2(attn_map2)
        attn_map4 = self.down3(attn_map3)
        attn_map5 = self.down4(attn_map4)

        # process input image
        conv1 = self.crb1(torch.cat([x, attn_map], dim=1))
        x = self.pool1(conv1)
        conv2 = self.crb2(x)
        x = self.pool2(conv2)
        conv3 = self.crb3(x)
        x = self.pool3(conv3)
        conv4 = self.crb4(x)
        x = self.pool4(conv4)
        x = self.crb5_1(x)
        x = x * attn_map5
        conv5 = self.crb5_2(x)

        # upsampling and fusion
        conv5 = self.deconv5(self.upsample(conv5))
        conv4 = conv4 * attn_map4
        conv6 = self.crb6(torch.cat([conv5, conv4], dim=1))

        conv6 = self.deconv4(self.upsample(conv6))
        conv3 = conv3 * attn_map3
        conv7 = self.crb7(torch.cat([conv6, conv3], dim=1))

        conv7 = self.deconv3(self.upsample(conv7))
        conv2 = conv2 * attn_map2
        conv8 = self.crb8(torch.cat([conv7, conv2], dim=1))

        conv8 = self.deconv2(self.upsample(conv8))
        conv1 = conv1 * attn_map
        conv9 = self.crb9_1(torch.cat([conv8, conv1], dim=1))
        latent = self.crb9_2(conv9)

        # output
        latent = latent * attn_map
        out = latent + x

        out = pad_tensor_back(out, pad_left, pad_right, pad_top, pad_bottom)
        latent = pad_tensor_back(latent, pad_left, pad_right, pad_top, pad_bottom)
        attn_map = pad_tensor_back(attn_map, pad_left, pad_right, pad_top, pad_bottom)

        return out, latent


class Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()

        in_channels = config.get("in_channels", 3)
        mid_channels = config.get("mid_channels", 64)
        num_layers = config.get("num_layers", 3)

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=4, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.down_convs = nn.ModuleList()
        multiplier = 1
        for i in range(1, num_layers):
            next_multiplier = min(2 ** i, 8)
            self.down_convs.append(nn.Sequential(
                nn.Conv2d(mid_channels * multiplier, mid_channels * next_multiplier, kernel_size=4, stride=2, padding=2),
                nn.LeakyReLU(0.2, inplace=True),
            ))
            multiplier = next_multiplier

        self.out_conv = nn.Sequential(
            nn.Conv2d(mid_channels * multiplier, mid_channels * multiplier, kernel_size=4, stride=1, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid_channels * multiplier, 1, kernel_size=4, stride=1, padding=2),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_conv(x)
        for down_conv in self.down_convs:
            x = down_conv(x)
        x = self.out_conv(x)
        return x


class VGG16(nn.Module):
    def __init__(self, config):
        super().__init__()

        in_channels = config.get("in_channels", 3)
        base_channels = config.get("base_channels", 64)

        self.conv1_1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(base_channels * 8, base_channels * 8, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(base_channels * 8, base_channels * 8, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(base_channels * 8, base_channels * 8, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(base_channels * 8, base_channels * 8, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(base_channels * 8, base_channels * 8, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1_1(x))
        x = self.relu(self.conv1_2(x))
        x = self.pool(x)

        x = self.relu(self.conv2_1(x))
        x = self.relu(self.conv2_2(x))
        x = self.pool(x)

        x = self.relu(self.conv3_1(x))
        x = self.relu(self.conv3_2(x))
        x = self.relu(self.conv3_3(x))
        x = self.pool(x)

        x = self.relu(self.conv4_1(x))
        x = self.relu(self.conv4_2(x))
        x = self.relu(self.conv4_3(x))
        x = self.pool(x)

        x = self.relu(self.conv5_1(x))
        # x = self.relu(self.conv5_2(x))
        # x = self.relu(self.conv5_3(x))
        return x

    def _freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def load_pretrained(self, path: str):
        self.load_state_dict(torch.load(path))
        self._freeze()

# ========================================================================================================
#                                             Loss functions
# ========================================================================================================

class PerceptualLoss(nn.Module):
    def __init__(self, num_features: int = 512):
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(num_features, affine=False)

    @staticmethod
    def _preprocess(batch):
        r, g, b = torch.chunk(batch, 3, dim=1)
        batch = torch.cat([b, g, r], dim=1)
        batch = (batch + 1) / 2 * 255
        return batch

    def forward(self, vgg: nn.Module, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        x = self._preprocess(x)
        target = self._preprocess(target)
        x_features = vgg(x)
        target_features = vgg(target)
        return F.mse_loss(self.instance_norm(x_features), self.instance_norm(target_features))


class GANLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.real_label= 1.0
        self.fake_label = 0.0
        self.real_label_var = None
        self.fake_label_var = None

    def forward(self, x: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        target_tensor = None
        if target_is_real:
            if (self.real_label_var is None or
                self.real_label_var.numel() != x.numel()):
                real_tensor = torch.ones_like(x) * self.real_label
                self.real_label_var = Variable(real_tensor, requires_grad=False)
                target_tensor = self.real_label_var
        else:
            if (self.fake_label_var is None or
                self.fake_label_var.numel() != x.numel()):
                fake_tensor = torch.zeros_like(x) * self.fake_label
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
                target_tensor = self.fake_label_var
        return F.mse_loss(x, target_tensor)


class EnlightenGAN(pl.LightningModule):
    def __init__(self, config, logger: "loguru.Logger"):
        super().__init__()

        self.config = config
        model_config = config["model"]
        self.generatorA = Generator(model_config["generator"])
        self.discriminatorA = Discriminator(model_config["discriminator"])
        self.discriminatorP = Discriminator(model_config["patch_discriminator"])
        self.vgg = VGG16(model_config["vgg"])
        self.vgg.load_pretrained(model_config["vgg_pretrained_path"])
        self.start_decay_epoch = model_config["start_decay_epoch"]
        self.extra_logger = logger

        # loss functions
        self.perceptual_loss = PerceptualLoss(num_features=model_config["vgg"]["base_channels"] * 8)
        self.gan_loss = GANLoss()
        self.cycle_loss = nn.L1Loss()
        self.identity_loss = nn.L1Loss()
        self.l1_loss = nn.L1Loss()

        # metrics


    def forward(self, ):
        pass

    def configure_optimizers(self):
        train_config = self.config["train"]
        optimizerGA = get_optimizer(self.generatorA.parameters(), train_config["optimizer"], self.extra_logger)
        optimizerDA = get_optimizer(self.discriminatorA.parameters(), train_config["optimizer"], self.extra_logger)
        optimizerDP = get_optimizer(self.discriminatorP.parameters(), train_config["optimizer"], self.extra_logger)
        schedulerGA = get_scheduler(optimizerGA, train_config["scheduler"], self.extra_logger)
        schedulerDA = get_scheduler(optimizerDA, train_config["scheduler"], self.extra_logger)
        schedulerDP = get_scheduler(optimizerDP, train_config["scheduler"], self.extra_logger)
        return [optimizerGA, optimizerDA, optimizerDP], [schedulerGA, schedulerDA, schedulerDP]

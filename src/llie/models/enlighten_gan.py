from __future__ import annotations

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import lightning as pl
import loguru
import random
import pyiqa
from typing import Optional

from src.llie.utils.config import get_optimizer, get_scheduler
from src.llie.models.utils import pad_tensor, pad_tensor_back, save_batch_tensor


# region Submodules

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
        self.crb3 = CRB(base_channels * 2, base_channels * 4, 2)
        self.pool3 = nn.MaxPool2d(2)
        self.crb4 = CRB(base_channels * 4, base_channels * 8, 2)
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
        identity = x.clone()

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
        latent = self.conv9_2(conv9)

        # output
        latent = latent * attn_map
        out = latent + identity

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

    def load_pretrained(self, path: str):
        self.load_state_dict(torch.load(path, weights_only=True))
        self.requires_grad_(False)

# endregion

# region Loss Functions

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
                fake_tensor = torch.ones_like(x) * self.fake_label
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return F.mse_loss(x, target_tensor)

# endregion

# region Main Model

class EnlightenGAN(pl.LightningModule):
    def __init__(self, config, logger: "loguru.Logger"):
        super().__init__()

        self.config = config
        self.model_config = config["model"]
        self.generatorA = Generator(self.model_config["generator"])
        self.discriminatorA = Discriminator(self.model_config["discriminator"])
        self.discriminatorP = Discriminator(self.model_config["patch_discriminator"])
        self.vgg = VGG16(self.model_config["vgg"])
        self.vgg.load_pretrained(self.model_config["vgg_pretrained_path"])
        self.start_decay_epoch = self.model_config["start_decay_epoch"]
        self.extra_logger = logger
        self.automatic_optimization = False

        self.save_path = config["data"].get("save_path", "")
        if self.save_path:
            os.makedirs(self.save_path, exist_ok=True)

        # loss functions
        self.perceptual_loss = PerceptualLoss(num_features=self.model_config["vgg"]["base_channels"] * 8)
        self.gan_loss = GANLoss()

        # metrics
        self.niqe = pyiqa.create_metric("niqe")

    def forward(self, real_a: Variable, real_b: Optional[Variable], attn_map: Variable, real_image: Variable):
        self.fake_b, latent_real_a = self.generatorA(real_image, attn_map)

        # select patch randomly
        h, w = real_a.shape[-2:]
        patch_size = self.model_config["patch_size"]
        num_patches = self.model_config["num_patches"]
        self.fake_patches = []
        self.real_b_patches = []
        self.real_a_patches = []
        for _ in range(num_patches + 1):    # +1 for what?
            h_offset = random.randint(0, max(0, h - patch_size - 1))
            w_offset = random.randint(0, max(0, w - patch_size - 1))
            self.fake_patches.append(self.fake_b[:, :, h_offset: h_offset + patch_size, w_offset: w_offset + patch_size])
            if real_b is not None:
                self.real_b_patches.append(real_b[:, :, h_offset: h_offset + patch_size, w_offset: w_offset + patch_size])
            self.real_a_patches.append(real_a[:, :, h_offset: h_offset + patch_size, w_offset: w_offset + patch_size])

    def configure_optimizers(self):
        train_config = self.config["train"]
        optimizerGA = get_optimizer(train_config, self.generatorA, self.extra_logger)
        optimizerDA = get_optimizer(train_config, self.discriminatorA, self.extra_logger)
        optimizerDP = get_optimizer(train_config, self.discriminatorP, self.extra_logger)
        schedulerGA = get_scheduler(train_config, optimizerGA, self.extra_logger)
        schedulerDA = get_scheduler(train_config, optimizerDA, self.extra_logger)
        schedulerDP = get_scheduler(train_config, optimizerDP, self.extra_logger)
        return [optimizerGA, optimizerDA, optimizerDP], [schedulerGA, schedulerDA, schedulerDP]

    def _compute_loss_generator(self, real_a: Variable, real_b: Variable):
        pred_fake = self.discriminatorA(self.fake_b)
        pred_real = self.discriminatorA(real_b)
        loss_generatorA = (self.gan_loss(pred_real - pred_fake.mean(), False) +
                           self.gan_loss(pred_fake - pred_real.mean(), True)) / 2

        num_patches = self.model_config["num_patches"]
        for i in range(num_patches + 1):
            pred_fake_patch = self.discriminatorP(self.fake_patches[i])
            loss_generatorA += self.gan_loss(pred_fake_patch, True) / (num_patches + 1)

        loss_vgg = self.perceptual_loss(self.vgg, self.fake_b, real_a)
        for i in range(num_patches + 1):
            loss_vgg += self.perceptual_loss(self.vgg, self.fake_patches[i], self.real_a_patches[i]) / (num_patches + 1)

        return loss_generatorA + loss_vgg

    def _compute_loss_discriminatorA(self, real_b: Variable):
        pred_fake = self.discriminatorA(self.fake_b.detach())
        pred_real = self.discriminatorA(real_b)
        return (self.gan_loss(pred_real - pred_fake.mean(), True) +
                self.gan_loss(pred_fake - pred_real.mean(), False)) / 2

    def _compute_loss_discriminatorP(self):
        num_patches = self.model_config["num_patches"]
        loss_discriminatorP = 0.
        for i in range(num_patches + 1):
            pred_fake_patch = self.discriminatorP(self.fake_patches[i].detach())
            pred_real_patch = self.discriminatorP(self.real_b_patches[i])
            loss_discriminatorP += ((self.gan_loss(pred_real_patch, True) + self.gan_loss(pred_fake_patch, False)) / 2
                                    / (num_patches + 1))

        return loss_discriminatorP

    def _compute_metrics(self):
        fake_b = torch.clip((self.fake_b + 1) / 2, 0, 1)
        self.niqe_score = self.niqe(fake_b).mean()

    def on_train_epoch_start(self):
        self.extra_logger.info(f"Epoch {self.current_epoch} starts.")
        self.extra_logger.info(f"Start training stage.")

    def training_step(self, batch, batch_idx):
        img_a, img_b, attn_map = batch["low"], batch["high"], batch["attn_map"]
        real_a = Variable(img_a)
        real_b = Variable(img_b)
        attn_map = Variable(attn_map)
        real_image = Variable(img_a)

        self.forward(real_a, real_b, attn_map, real_image)

        optimizerGA, optimizerDA, optimizerDP = self.optimizers()
        # update generator
        optimizerGA.zero_grad()
        self.loss_generatorA = self._compute_loss_generator(real_a, real_b)
        self.manual_backward(self.loss_generatorA)
        optimizerGA.step()

        # update discriminatorA
        optimizerDA.zero_grad()
        self.loss_discriminatorA = self._compute_loss_discriminatorA(real_b)
        self.manual_backward(self.loss_discriminatorA)
        optimizerDA.step()

        # update discriminatorP
        optimizerDP.zero_grad()
        self.loss_discriminatorP = self._compute_loss_discriminatorP()
        self.manual_backward(self.loss_discriminatorP)
        optimizerDP.step()

        self._compute_metrics()

        # log
        self.log_dict({
            "train/loss_generatorA": self.loss_generatorA,
            "train/loss_discriminatorA": self.loss_discriminatorA,
            "train/loss_discriminatorP": self.loss_discriminatorP,
            "train/niqe": self.niqe_score,
        }, on_step=True, on_epoch=True, batch_size=real_a.shape[0])

    def on_validation_epoch_start(self):
        self.extra_logger.info(f"Start validation stage.")

    def validation_step(self, batch, batch_idx):
        img_a, img_b, attn_map = batch["low"], batch["high"], batch["attn_map"]
        real_a = Variable(img_a)
        real_b = Variable(img_b)
        attn_map = Variable(attn_map)
        real_image = Variable(img_a)

        self.forward(real_a, real_b, attn_map, real_image)
        self.loss_generatorA = self._compute_loss_generator(real_a, real_b)
        self.loss_discriminatorA = self._compute_loss_discriminatorA(real_b)
        self.loss_discriminatorP = self._compute_loss_discriminatorP()
        self._compute_metrics()
        # log
        self.log_dict({
            "val/loss_generatorA": self.loss_generatorA,
            "val/loss_discriminatorA": self.loss_discriminatorA,
            "val/loss_discriminatorP": self.loss_discriminatorP,
            "val/niqe": self.niqe_score,
        }, on_step=True, on_epoch=True, batch_size=real_a.shape[0])

        # log images
        if batch_idx == 0:
            real_a = (img_a[0].detach().cpu() + 1) / 2
            attn_map = (attn_map[0].detach().cpu()).repeat(3, 1, 1)
            real_b = (img_b[0].detach().cpu() + 1) / 2
            fake_b = (self.fake_b[0].detach().cpu() + 1) / 2

            image = torch.cat([
                torch.cat([real_a, attn_map], dim=2),
                torch.cat([fake_b, real_b], dim=2)
            ], dim=1)
            image = torch.clip(image * 255, 0, 255).to(torch.uint8)
            self.logger.experiment.add_image("val/image", image, self.current_epoch)

    def on_validation_epoch_end(self):
        schedulerGA, schedulerDA, schedulerDP = self.lr_schedulers()

        if self.current_epoch >= self.start_decay_epoch:
            schedulerGA.step(self.loss_generatorA)
            schedulerDA.step(self.loss_discriminatorA)
            schedulerDP.step(self.loss_discriminatorP)

        self.log_dict({
            "lr/generatorA": schedulerGA.get_last_lr()[0],
            "lr/discriminatorA": schedulerDA.get_last_lr()[0],
            "lr/discriminatorP": schedulerDP.get_last_lr()[0],
        }, on_step=False, on_epoch=True)

    def on_fit_start(self):
        self.extra_logger.info(f"Start training on {self.device}.")

    def on_fit_end(self):
        self.extra_logger.info(f"Training finished.")

    def test_step(self, batch, batch_idx):
        img_a, img_b, attn_map = batch["low"], batch["high"], batch["attn_map"]
        real_a = Variable(img_a)
        real_b = Variable(img_b)
        attn_map = Variable(attn_map)
        real_image = Variable(img_a)

        self.forward(real_a, real_b, attn_map, real_image)
        self.loss_generatorA = self._compute_loss_generator(real_a, real_b)
        self.loss_discriminatorA = self._compute_loss_discriminatorA(real_b)
        self.loss_discriminatorP = self._compute_loss_discriminatorP()
        self._compute_metrics()
        # log
        self.log_dict({
            "test/loss_generatorA": self.loss_generatorA,
            "test/loss_discriminatorA": self.loss_discriminatorA,
            "test/loss_discriminatorP": self.loss_discriminatorP,
            "test/niqe": self.niqe_score,
        }, on_step=False, on_epoch=True, batch_size=real_a.shape[0])

        # log images
        if batch_idx == 0:
            real_a = (img_a[0].detach().cpu() + 1) / 2
            attn_map = attn_map[0].detach().cpu()
            real_b = (img_b[0].detach().cpu() + 1) / 2
            fake_b = (self.fake_b[0].detach().cpu() + 1) / 2

            image = torch.cat([
                torch.cat([real_a, attn_map], dim=2),
                torch.cat([fake_b, real_b], dim=2)
            ], dim=1)
            image = torch.clip(image * 255, 0, 255).to(torch.uint8)
            self.logger.experiment.add_image("test/image", image, self.current_epoch)

    def on_test_start(self):
        self.extra_logger.info("Start testing.")

    def on_test_end(self):
        self.extra_logger.info(f"Testing finished.")

    def predict_step(self, batch, batch_idx):
        img_a, attn_map = batch["low"], batch["attn_map"]
        real_a = Variable(img_a)
        attn_map = Variable(attn_map)
        real_image = Variable(img_a)

        self.forward(real_a, None, attn_map, real_image)
        self._compute_metrics()
        self.extra_logger.info(f"NIQE score: {self.niqe_score}")

        # save results
        fake_b = (self.fake_b.detach().cpu() + 1) / 2
        fake_b = torch.clip(fake_b * 255, 0, 255).to(torch.uint8)
        save_batch_tensor(fake_b, self.save_path, batch)

# endregion
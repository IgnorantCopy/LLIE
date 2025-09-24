from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
import numpy as np
from bm3d import bm3d
from einops import rearrange
import loguru
from typing import Tuple, Optional

from src.llie.utils.config import get_optimizer, get_scheduler


class DecomNet(nn.Module):
    def __init__(self, in_channels, num_layers, hidden_dim: int = 64, kernel_size: int = 3):
        super().__init__()

        self.in_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=kernel_size * 3, padding=4, padding_mode='replicate')
        self.conv_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.conv_layers.append(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=1, padding_mode='replicate')
            )
            self.conv_layers.append(nn.ReLU())
        self.out_conv = nn.Conv2d(hidden_dim, in_channels, kernel_size=kernel_size, padding=1, padding_mode='replicate')
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([x_max, x], dim=1)

        x = self.in_conv(x)
        for layer in self.conv_layers:
            x = layer(x)
        x = self.out_conv(x)

        R = self.sigmoid(x[:, 0:3, :, :])
        L = self.sigmoid(x[:, 3:4, :, :])

        return R, L

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False


class RelightNet(nn.Module):
    def __init__(self,in_channels: int = 4, hidden_dim: int = 64, kernel_size: int = 3):
        super().__init__()

        self.in_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=kernel_size, padding=1, padding_mode='replicate')

        self.en_conv1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=2, padding=1, padding_mode='replicate')
        self.en_conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=2, padding=1, padding_mode='replicate')
        self.en_conv3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=2, padding=1, padding_mode='replicate')

        self.de_conv1 = nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=kernel_size, padding=1, padding_mode='replicate')
        self.de_conv2 = nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=kernel_size, padding=1, padding_mode='replicate')
        self.de_conv3 = nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=kernel_size, padding=1, padding_mode='replicate')

        self.fusion = nn.Conv2d(hidden_dim * 3, hidden_dim, kernel_size=1, padding=1, padding_mode='replicate')
        self.out_conv = nn.Conv2d(hidden_dim, 1, kernel_size=3)

        self.relu = nn.ReLU()

    def forward(self, L: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        x = torch.cat([L, R], dim=1)
        x0 = self.in_conv(x)
        x1 = self.relu(self.en_conv1(x0))
        x2 = self.relu(self.en_conv2(x1))
        x3 = self.relu(self.en_conv3(x2))

        x3_up = F.interpolate(x3, size=(x2.shape[2], x2.shape[3]), mode='nearest')
        x2_cat = torch.cat([x3_up, x2], dim=1)
        x2 = self.relu(self.de_conv1(x2_cat))
        x2_up = F.interpolate(x2, size=(x1.shape[2], x1.shape[3]), mode='nearest')
        x1_cat = torch.cat([x2_up, x1], dim=1)
        x1 = self.relu(self.de_conv2(x1_cat))
        x1_up = F.interpolate(x1, size=(x0.shape[2], x0.shape[3]), mode='nearest')
        x0_cat = torch.cat([x1_up, x0], dim=1)
        x0 = self.relu(self.de_conv3(x0_cat))

        x2_res = F.interpolate(x2, size=(R.shape[2], R.shape[3]), mode='nearest')
        x1_res = F.interpolate(x1, size=(R.shape[2], R.shape[3]), mode='nearest')
        x = torch.cat([x2_res, x1_res, x0], dim=1)
        x = self.fusion(x)
        x = self.out_conv(x)
        return x


class RetinexNet(pl.LightningModule):
    def __init__(self, config, logger: "loguru.Logger"):
        super().__init__()

        self.config = config
        self.extra_logger = logger
        model_config = config["model"]
        self.in_channels = model_config.get("in_channels", 3)
        self.hidden_dim = model_config.get("hidden_dim", 64)
        self.kernel_size = model_config.get("kernel_size", 3)
        self.num_layers = model_config.get("num_layers", 5)
        self.decom_epochs = model_config.get("decom_epochs", 100)
        self.relight_epochs = model_config.get("relight_epochs", 100)

        self.decom_net = DecomNet(self.in_channels, self.num_layers, self.hidden_dim, self.kernel_size)
        self.relight_net = RelightNet(self.in_channels, self.hidden_dim, self.kernel_size)

        self.automatic_optimization = False

    def forward(self, low: torch.Tensor, high: Optional[torch.Tensor] = None, denoise: bool = False):
        self.R_low, I_low = self.decom_net(low)
        self.I_low = torch.cat([I_low, I_low, I_low], dim=1)

        I_delta = self.relight_net(I_low, self.R_low)
        self.I_delta = torch.cat([I_delta, I_delta, I_delta], dim=1)

        if denoise:
            self.R_low = self._denoise(self.R_low)
        self.S = self.R_low * self.I_delta

        if high is not None:
            self.R_high, I_high = self.decom_net(high)
            self.I_high = torch.cat([I_high, I_high, I_high], dim=1)

    @staticmethod
    def _denoise(x: torch.Tensor):
        upper_bounds = [1, 0.08, 0.03]
        alphas = [1, 10, 100]
        sigmas = [10, 20, 40]
        for i in range(x.shape[0]):
            x_i = rearrange(x[i], 'c h w -> h w c').detach().cpu().numpy()
            x0 = x_i.copy()
            for j in range(3):
                m = np.clip(x_i, 0, upper_bounds[j]) / upper_bounds[j]
                m = m ** alphas[j]
                x_i = x_i * m + bm3d(x0, sigma_psd=sigmas[j]) * (1 - m)
            x[i] = rearrange(torch.from_numpy(x_i).to(x), 'h w c -> c h w')
        return x

    def _compute_loss(self, low: torch.Tensor, high: torch.Tensor):
        self.recon_loss_low = F.l1_loss(self.R_low * self.I_low, low)
        self.recon_loss_high = F.l1_loss(self.R_high * self.I_high, high)
        self.recon_loss_mutual_low = F.l1_loss(self.R_high * self.I_low, low)
        self.recon_loss_mutual_high = F.l1_loss(self.R_low * self.I_high, high)
        self.equal_R_loss = F.l1_loss(self.R_low, self.R_high)
        self.relight_loss = F.l1_loss(self.R_low * self.I_delta, high)

        self.smooth_loss_low = self._smooth_loss(self.I_low, self.R_low)
        self.smooth_loss_high = self._smooth_loss(self.I_high, self.R_high)
        self.smooth_loss_delta = self._smooth_loss(self.I_delta, self.R_low)

        self.decom_loss = self.recon_loss_low + self.recon_loss_high + \
                          0.001 * self.recon_loss_mutual_low + 0.001 * self.recon_loss_mutual_high + \
                          0.1 * self.smooth_loss_low + 0.1 * self.smooth_loss_high + \
                          0.01 * self.equal_R_loss
        self.relight_loss = self.relight_loss + 3 * self.smooth_loss_delta

    def _gradient(self, x: torch.Tensor, direction: str):
        self.smooth_kernel_x = torch.FloatTensor([[0, 0], [-1, 1]]).view((1, 1, 2, 2)).to(x)
        self.smooth_kernel_y = self.smooth_kernel_x.transpose(2, 3)

        if direction == 'x':
            kernel = self.smooth_kernel_x
        elif direction == 'y':
            kernel = self.smooth_kernel_y
        else:
            self.extra_logger.error('RetinexNet._gradient: direction should be x or y')
            raise ValueError('direction should be x or y')

        grad = torch.abs(F.conv2d(x, kernel, stride=1, padding=1))
        return grad

    def _avg_gradient(self, x: torch.Tensor, direction: str):
        return F.avg_pool2d(self._gradient(x, direction), kernel_size=3, stride=1, padding=1)

    def _smooth_loss(self, I: torch.Tensor, R: torch.Tensor):
        R = 0.299 * R[:, 0, :, :] + 0.587 * R[:, 1, :, :] + 0.114 * R[:, 2, :, :]
        R = R.unsqueeze(1)
        I = I[:, 0, :, :]
        I = I.unsqueeze(1)
        return torch.mean(
            self._gradient(I, 'x') * torch.exp(-10 * self._avg_gradient(R, 'x')) +
            self._gradient(I, 'y') * torch.exp(-10 * self._avg_gradient(R, 'y'))
        )

    def configure_optimizers(self):
        train_config = self.config["train"]
        decom_optimizer = get_optimizer(train_config, self.decom_net, self.extra_logger)
        relight_optimizer = get_optimizer(train_config, self.relight_net, self.extra_logger)
        decom_scheduler = get_scheduler(train_config, decom_optimizer, self.extra_logger)
        relight_scheduler = get_scheduler(train_config, relight_optimizer, self.extra_logger)

        return [decom_optimizer, relight_optimizer], [decom_scheduler, relight_scheduler]

    def on_train_epoch_start(self):
        self.extra_logger.info(f"Epoch {self.current_epoch} starts.")
        self.extra_logger.info(f"Start training stage.")

    def training_step(self, batch, batch_idx):
        high, low = batch["high"], batch["low"]
        self.forward(low, high)
        self._compute_loss(low, high)

        decom_optimizer, relight_optimizer = self.optimizers()
        if self.current_epoch < self.decom_epochs:
            decom_optimizer.zero_grad()
            self.manual_backward(self.decom_loss)
            decom_optimizer.step()
            self.log("train/decom_loss", self.decom_loss, on_step=True, on_epoch=True, prog_bar=True)
            return self.decom_loss
        else:
            relight_optimizer.zero_grad()
            self.manual_backward(self.relight_loss)
            relight_optimizer.step()
            self.log("train/relight_loss", self.relight_loss, on_step=True, on_epoch=True, prog_bar=True)
            return self.relight_loss

    def on_validation_epoch_start(self):
        self.extra_logger.info(f"Start validation stage.")

    def validation_step(self, batch, batch_idx):
        high, low = batch["high"], batch["low"]
        self.forward(low, high)
        self._compute_loss(low, high)

        if self.current_epoch < self.decom_epochs:
            self.log("val/decom_loss", self.decom_loss, on_step=False, on_epoch=True, prog_bar=True)
        else:
            self.log("val/relight_loss", self.relight_loss, on_step=False, on_epoch=True, prog_bar=True)
        # log images
        if batch_idx == 0:
            R_low, I_low = self.R_low[0].detach().cpu(), self.I_low[0].detach().cpu()
            R_high, I_high = self.R_high[0].detach().cpu(), self.I_high[0].detach().cpu()
            recon_low, recon_high = R_low * I_low, R_high * I_high
            low, high = low[0].detach().cpu(), high[0].detach().cpu()

            low_image = torch.cat([low, R_low, I_low, recon_low], dim=1)
            high_image = torch.cat([high, R_high, I_high, recon_high], dim=1)
            image = torch.cat([low_image, high_image], dim=2)
            image = torch.clip(image * 255, 0, 255).to(torch.uint8)
            out = torch.clip(self.S[0].detach().cpu() * 255, 0, 255).to(torch.uint8)
            self.logger.experiment.add_image("val/image", image)
            self.logger.experiment.add_image("val/out", out)

    def on_validation_epoch_end(self):
        decom_scheduler, relight_scheduler = self.lr_schedulers()
        if self.current_epoch < self.decom_epochs:
            decom_scheduler.step(self.decom_loss)
            self.log("lr", decom_scheduler.get_last_lr()[0])
        else:
            relight_scheduler.step(self.relight_loss)
            self.log("lr", relight_scheduler.get_last_lr()[0])

        if self.current_epoch + 1 == self.decom_epochs:
            self.decom_net.freeze()
            self.extra_logger.info("DecomNet training is finished, start RelightNet training.")

    def on_fit_start(self):
        self.extra_logger.info(f"Start training on {self.device}.")

    def on_fit_end(self):
        self.extra_logger.info("Training finished.")

    def test_step(self, batch, batch_idx):
        high, low = batch["high"], batch["low"]
        self.forward(low, high, denoise=True)
        self._compute_loss(low, high)

        if self.current_epoch < self.decom_epochs:
            self.log("test/decom_loss", self.decom_loss, on_step=False, on_epoch=True, prog_bar=True)
        else:
            self.log("test/relight_loss", self.relight_loss, on_step=False, on_epoch=True, prog_bar=True)
        # log images
        if batch_idx == 0:
            R_low, I_low = self.R_low[0].detach().cpu(), self.I_low[0].detach().cpu()
            R_high, I_high = self.R_high[0].detach().cpu(), self.I_high[0].detach().cpu()
            recon_low, recon_high = R_low * I_low, R_high * I_high
            low, high = low[0].detach().cpu(), high[0].detach().cpu()

            low_image = torch.cat([low, R_low, I_low, recon_low], dim=1)
            high_image = torch.cat([high, R_high, I_high, recon_high], dim=1)
            image = torch.cat([low_image, high_image], dim=2)
            image = torch.clip(image * 255, 0, 255).to(torch.uint8)
            out = torch.clip(self.S[0].detach().cpu() * 255, 0, 255).to(torch.uint8)
            self.logger.experiment.add_image("test/image", image)
            self.logger.experiment.add_image("test/out", out)

    def on_test_start(self):
        self.extra_logger.info("Start testing.")

    def on_test_end(self):
        self.extra_logger.info("Testing finished.")

    def predict_step(self, batch, batch_idx):
        low = batch["low"]
        self.forward(low, denoise=True)
        return torch.clip(self.S * 255, 0, 255).to(torch.uint8)

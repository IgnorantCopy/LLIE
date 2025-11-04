from __future__ import annotations

import os
import torch
import torch.nn as nn
from torchvision.transforms import Grayscale, GaussianBlur
import lightning as pl
import loguru
import pyiqa
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from typing import Optional

from src.llie.metrics.ssim import SSIM
from src.llie.utils.config import get_optimizer, get_scheduler
from src.llie.models.utils import gradient, save_batch_tensor


# region Layer Decomposition Network

class DecomNet(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 32):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1)
        self.deconv1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2, bias=False)
        self.conv4 = nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=3, padding=1)
        self.deconv2 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2, bias=False)
        self.conv5 = nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1)
        self.out_conv1 = nn.Conv2d(base_channels, 3, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.lrelu = nn.LeakyReLU(0.2)

        self.conv6 = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)
        self.out_conv2 = nn.Conv2d(base_channels * 2, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(self.pool1(x1)))
        x3 = self.lrelu(self.conv3(self.pool2(x2)))
        x4 = self.lrelu(self.conv4(torch.cat([self.deconv1(x3), x2], dim=1)))
        x5 = self.lrelu(self.conv5(torch.cat([self.deconv2(x4), x1], dim=1)))
        r_out = self.sigmoid(self.out_conv1(x5))

        x6 = torch.cat([self.lrelu(self.conv6(x1)), x5], dim=1)
        i_out = self.sigmoid(self.out_conv2(x6))

        return r_out, i_out

# endregion

# region Reflectance Restoration Network

class MultiScaleModule(nn.Module):
    def __init__(self, base_channels: int):
        super().__init__()

        self.level1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
        )
        self.level2 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
            nn.ConvTranspose2d(base_channels, base_channels, kernel_size=2, stride=2),
        )
        self.level4 = nn.Sequential(
            nn.MaxPool2d(4, 4),
            nn.Conv2d(base_channels, base_channels, kernel_size=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
            nn.ConvTranspose2d(base_channels, base_channels, kernel_size=2, stride=2),
            nn.ConvTranspose2d(base_channels, base_channels, kernel_size=2, stride=2),
        )
        self.out_conv = nn.Conv2d(4 * base_channels, base_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.level1(x)
        x2 = self.level2(x1)
        x4 = self.level4(x2)
        x_out = self.out_conv(torch.cat([x, x1, x2, x4], dim=1))
        return x_out


class MSIA(nn.Module):
    """Multi-scale Illumination Attention"""
    def __init__(self, base_channels: int):
        super().__init__()

        self.in_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.multi_scale = MultiScaleModule(base_channels)

    def forward(self, features: torch.Tensor, illumination: torch.Tensor) -> torch.Tensor:
        illumination = self.sigmoid(self.in_conv(illumination))
        features = illumination * features
        features = self.multi_scale(features)
        return features


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.msia = MSIA(out_channels)

    def forward(self, reflection: torch.Tensor, illumination: torch.Tensor) -> torch.Tensor:
        features = self.layers(reflection)
        features = self.msia(features, illumination)
        return features


class RestorationNet(nn.Module):
    def __init__(self, base_channels: int = 32):
        super().__init__()

        self.block1 = BasicBlock(3, base_channels, base_channels * 2)
        self.block2 = BasicBlock(base_channels * 2, base_channels * 4, base_channels * 8)
        self.block3 = BasicBlock(base_channels * 8, base_channels * 16, base_channels * 8)
        self.block4 = BasicBlock(base_channels * 8, base_channels * 4, base_channels * 2)
        self.out_conv = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channels, 3, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, reflection: torch.Tensor, illumination: torch.Tensor) -> torch.Tensor:
        features = self.block1(reflection, illumination)
        features = self.block2(features, illumination)
        features = self.block3(features, illumination)
        features = self.block4(features, illumination)
        out = self.out_conv(features)
        return out

# endregion

# region Illumination Adjustment Network

class AdjustmentNet(nn.Module):
    def __init__(self, alpha: float, base_channels: int = 32):
        super().__init__()

        self.alpha = alpha
        self.layers = nn.Sequential(
            nn.Conv2d(2, base_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channels, 1, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Sigmoid(),
        )

    def forward(self, illumination: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha * torch.ones_like(illumination)
        illumination = torch.cat([illumination, alpha], dim=1)
        illumination = self.layers(illumination)
        return illumination

# endregion

# region Loss Functions

class DecomLoss(nn.Module):
    def __init__(self, lambda_re: float, lambda_rs: float, lambda_mc: float, lambda_is: float):
        super().__init__()
        self.lambda_re = lambda_re
        self.lambda_rs = lambda_rs
        self.lambda_mc = lambda_mc
        self.lambda_is = lambda_is

    @staticmethod
    def grad_norm(x: torch.Tensor, direction: str):
        grad = gradient(x, direction)
        grad_norm = (grad - grad.min()) / (grad.max() - grad.min() + 1e-4)
        return grad_norm

    @staticmethod
    def recon_loss(x: torch.Tensor, reflection: torch.Tensor, illumination: torch.Tensor) -> torch.Tensor:
        illumination = torch.cat([illumination, illumination, illumination], dim=1)
        return torch.abs(reflection * illumination - x).mean()

    @staticmethod
    def reflectance_similarity(r_low: torch.Tensor, r_high: torch.Tensor) -> torch.Tensor:
        return torch.abs(r_low - r_high).mean()

    def mutual_consistency(self, i_low: torch.Tensor, i_high: torch.Tensor) -> torch.Tensor:
        grad_low_x = self.grad_norm(i_low, "x")
        grad_high_x = self.grad_norm(i_high, "x")
        m_x = grad_low_x + grad_high_x
        x_loss = m_x * torch.exp(-10 * m_x)

        grad_low_y = self.grad_norm(i_low, "y")
        grad_high_y = self.grad_norm(i_high, "y")
        m_y = grad_low_y + grad_high_y
        y_loss = m_y * torch.exp(-10 * m_y)

        return (x_loss + y_loss).mean()

    def illumination_smoothness(self, image: torch.Tensor, illumination: torch.Tensor) -> torch.Tensor:
        rgb_to_gray = Grayscale(num_output_channels=1)
        gray_image = rgb_to_gray(image)

        grad_i_x = self.grad_norm(illumination, "x")
        grad_image_x = self.grad_norm(gray_image, "x")
        x_loss = torch.abs(grad_i_x / torch.clip(grad_image_x, min=0.01, max=None))

        grad_i_y = self.grad_norm(illumination, "y")
        grad_image_y = self.grad_norm(gray_image, "y")
        y_loss = torch.abs(grad_i_y / torch.clip(grad_image_y, min=0.01, max=None))

        return (x_loss + y_loss).mean()

    def forward(self, low: torch.Tensor, high: torch.Tensor, r_low: torch.Tensor, r_high: torch.Tensor,
                i_low: torch.Tensor, i_high: torch.Tensor):
        return (
            self.lambda_re * (self.recon_loss(low, r_low, i_low) + self.recon_loss(high, r_high, i_high)) +
            self.lambda_rs * self.reflectance_similarity(r_low, r_high) +
            self.lambda_mc * self.mutual_consistency(i_low, i_high) +
            self.lambda_is * (self.illumination_smoothness(low, i_low) + self.illumination_smoothness(high, i_high))
        )


class RestorationLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.ssim = SSIM()
        self.mse = nn.MSELoss()

    def forward(self, r_low: torch.Tensor, r_high: torch.Tensor):
        ssim_loss =  1 - self.ssim(r_low, r_high)
        mse_loss = self.mse(r_low, r_high)
        return ssim_loss + mse_loss


class AdjustmentLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    @staticmethod
    def grad_norm(x: torch.Tensor, direction: str):
        grad = gradient(x, direction, no_abs=True)
        grad_norm = (grad - grad.min()) / (grad.max() - grad.min() + 1e-4)
        return grad_norm

    def grad_loss(self, i_low: torch.Tensor, i_high: torch.Tensor):
        grad_low_x = self.grad_norm(i_low, "x")
        grad_high_x = self.grad_norm(i_high, "x")
        x_loss = (grad_low_x - grad_high_x) ** 2

        grad_low_y = self.grad_norm(i_low, "y")
        grad_high_y = self.grad_norm(i_high, "y")
        y_loss = (grad_low_y - grad_high_y) ** 2

        return (x_loss + y_loss).mean()

    def forward(self, i_low: torch.Tensor, i_high: torch.Tensor):
        grad_loss = self.grad_loss(i_low, i_high)
        mse_loss = self.mse(i_low, i_high)
        return grad_loss + mse_loss

# endregion

# region Main Model

class KinDPlus(pl.LightningModule):
    def __init__(self, config, logger: "loguru.Logger"):
        super().__init__()

        self.config = config
        model_config = config["model"]
        base_channels = model_config["base_channels"]
        self.decom_net = DecomNet(model_config["in_channels"], base_channels)
        self.restoration_net = RestorationNet(base_channels)
        self.adjustment_net = AdjustmentNet(model_config["alpha"], base_channels)
        self.restoration_net.requires_grad_(False)
        self.adjustment_net.requires_grad_(False)

        self.decom_epochs = model_config["decom_epochs"]
        self.restoration_epochs = model_config["restoration_epochs"]
        self.adjustment_epochs = model_config["adjustment_epochs"]
        self.extra_logger = logger
        self.automatic_optimization = False

        self.save_path = config["data"].get("save_path", "")
        if self.save_path:
            os.makedirs(self.save_path, exist_ok=True)

        # loss functions
        decom_loss_config = model_config["decom_loss"]
        self.decom_loss = DecomLoss(
            lambda_re=decom_loss_config["lambda_re"],
            lambda_rs=decom_loss_config["lambda_rs"],
            lambda_mc=decom_loss_config["lambda_mc"],
            lambda_is=decom_loss_config["lambda_is"],
        )
        self.restoration_loss = RestorationLoss()
        self.adjustment_loss = AdjustmentLoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.niqe = pyiqa.create_metric("niqe")

    def forward(self, low: torch.Tensor, high: Optional[torch.Tensor] = None):
        self.r_low_decom, self.i_low_decom = self.decom_net(low)
        if high is not None:
            self.r_high, self.i_high = self.decom_net(high)
        if not self.training or self.current_epoch > self.decom_epochs:
            self.r_low_restoration = self.restoration_net(self.r_low_decom, self.i_low_decom)
        if not self.training or self.current_epoch > self.decom_epochs + self.restoration_epochs:
            self.i_low_adjustment = self.adjustment_net(self.i_low_decom)

    def get_high_pred(self):
        rgb_to_gray = Grayscale(num_output_channels=1)
        gaussian_blur = GaussianBlur(kernel_size=11, sigma=3.0)
        r_gray = rgb_to_gray(self.r_low_restoration)
        r_blur = gaussian_blur(r_gray)
        i_low = torch.clip((r_blur * 2) ** 0.5, min=1, max=None)
        r_denoised = self.r_low_restoration * i_low
        high_pred = r_denoised * self.i_low_adjustment
        return high_pred

    def configure_optimizers(self):
        train_config = self.config["train"]
        decom_optimizer = get_optimizer(train_config, self.decom_net, self.extra_logger)
        restoration_optimizer = get_optimizer(train_config, self.restoration_net, self.extra_logger)
        adjustment_optimizer = get_optimizer(train_config, self.adjustment_net, self.extra_logger)

        decom_scheduler = get_scheduler(train_config, decom_optimizer, self.extra_logger)
        restoration_scheduler = get_scheduler(train_config, restoration_optimizer, self.extra_logger)
        adjustment_scheduler = get_scheduler(train_config, adjustment_optimizer, self.extra_logger)

        return ([decom_optimizer, restoration_optimizer, adjustment_optimizer],
                [decom_scheduler, restoration_scheduler, adjustment_scheduler])

    def _compute_loss(self, low: torch.Tensor, high: torch.Tensor):
        self.decom_loss_val = self.decom_loss(low, high, self.r_low_decom, self.r_high, self.i_low_decom, self.i_high)
        if not self.training or self.current_epoch > self.decom_epochs:
            self.restoration_loss_val = self.restoration_loss(self.r_low_restoration, self.r_high)
        if not self.training or self.current_epoch > self.decom_epochs + self.restoration_epochs:
            self.adjustment_loss_val = self.adjustment_loss(self.i_low_adjustment, self.i_high)

    def _compute_metrics(self, high_pred: torch.Tensor, high: torch.Tensor):
        high_pred = torch.clip(high_pred, 0, 1)
        self.psnr_val = self.psnr(high_pred, high)
        self.ssim_val = self.ssim(high_pred, high)
        self.niqe_val = self.niqe(high_pred).mean()

    def on_train_epoch_start(self):
        self.extra_logger.info(f"Epoch {self.current_epoch} starts.")
        self.extra_logger.info(f"Start training stage.")

    def training_step(self, batch, batch_idx):
        low, high = batch["low"], batch["high"]
        self.forward(low, high)
        self._compute_loss(low, high)

        decom_optimizer, restoration_optimizer, adjustment_optimizer = self.optimizers()
        if self.current_epoch <= self.decom_epochs:
            decom_optimizer.zero_grad()
            self.manual_backward(self.decom_loss_val)
            decom_optimizer.step()
            self.log("train/decom_loss", self.decom_loss_val, on_step=True, on_epoch=True, batch_size=low.shape[0])
        elif self.current_epoch <= self.decom_epochs + self.restoration_epochs:
            restoration_optimizer.zero_grad()
            self.manual_backward(self.restoration_loss_val)
            restoration_optimizer.step()
            self.log("train/restoration_loss", self.restoration_loss_val, on_step=True, on_epoch=True, batch_size=low.shape[0])
        else:
            adjustment_optimizer.zero_grad()
            self.manual_backward(self.adjustment_loss_val)
            adjustment_optimizer.step()
            self.log("train/adjustment_loss", self.adjustment_loss_val, on_step=True, on_epoch=True, batch_size=low.shape[0])

    def on_validation_epoch_start(self):
        self.extra_logger.info(f"Start validation stage.")

    def validation_step(self, batch, batch_idx):
        low, high = batch["low"], batch["high"]
        self.forward(low, high)
        self._compute_loss(low, high)

        if self.current_epoch <= self.decom_epochs:
            self.log("val/decom_loss", self.decom_loss_val, on_step=False, on_epoch=True, batch_size=low.shape[0])
            if batch_idx == 0:
                r_low_decom = self.r_low_decom[0].detach().cpu()
                i_low_decom = self.i_low_decom[0].detach().cpu()
                r_high_decom = self.r_high[0].detach().cpu()
                i_high_decom = self.i_high[0].detach().cpu()
                image1 = torch.cat([r_low_decom, r_high_decom], dim=2)
                image2 = torch.cat([i_low_decom, i_high_decom], dim=2)
                image2 = image2.repeat(3, 1, 1)
                image = torch.cat([image1, image2], dim=1)
                image = torch.clip(image * 255, 0, 255).to(torch.uint8)
                self.logger.experiment.add_image("val/decomposition", image, self.current_epoch)
        elif self.current_epoch <= self.decom_epochs + self.restoration_epochs:
            self.log("val/restoration_loss", self.restoration_loss_val, on_step=False, on_epoch=True, batch_size=low.shape[0])
            if batch_idx == 0:
                r_low_restoration = self.r_low_restoration[0].detach().cpu()
                r_high_restoration = self.r_high[0].detach().cpu()
                image = torch.cat([r_low_restoration, r_high_restoration], dim=2)
                image = torch.clip(image * 255, 0, 255).to(torch.uint8)
                self.logger.experiment.add_image("val/restoration", image, self.current_epoch)
        else:
            high_pred = self.get_high_pred()
            self._compute_metrics(high_pred, high)
            self.log_dict({
                "val/adjustment_loss": self.adjustment_loss_val,
                "val/psnr": self.psnr_val,
                "val/ssim": self.ssim_val,
                "val/niqe": self.niqe_val,
            }, on_step=False, on_epoch=True, batch_size=low.shape[0])
            if batch_idx == 0:
                low = low[0].detach().cpu()
                high = high[0].detach().cpu()
                high_pred = high_pred[0].detach().cpu()
                image1 = torch.cat([low, high_pred, high], dim=2)
                image1 = torch.clip(image1 * 255, 0, 255).to(torch.uint8)

                i_low_adjustment = self.i_low_adjustment[0].detach().cpu()
                i_high_adjustment = self.i_high[0].detach().cpu()
                image2 = torch.cat([i_low_adjustment, i_high_adjustment], dim=2)
                image2 = image2.repeat(3, 1, 1)
                image2 = torch.clip(image2 * 255, 0, 255).to(torch.uint8)

                self.logger.experiment.add_image("val/image", image1, self.current_epoch)
                self.logger.experiment.add_image("val/adjustment", image2, self.current_epoch)

    def on_validation_epoch_end(self):
        decom_scheduler, restoration_scheduler, adjustment_scheduler = self.lr_schedulers()
        if self.current_epoch <= self.decom_epochs:
            decom_scheduler.step()
            self.log("lr/decom_lr", decom_scheduler.get_last_lr()[0], on_step=False, on_epoch=True)
        elif self.current_epoch <= self.decom_epochs + self.restoration_epochs:
            restoration_scheduler.step()
            self.log("lr/restoration_lr", restoration_scheduler.get_last_lr()[0], on_step=False, on_epoch=True)
        else:
            adjustment_scheduler.step()
            self.log("lr/adjustment_lr", adjustment_scheduler.get_last_lr()[0], on_step=False, on_epoch=True)

        if self.current_epoch == self.decom_epochs:
            self.decom_net.requires_grad_(False)
            self.restoration_net.requires_grad_(True)
            self.extra_logger.info(f"DecomNet training finished, start training RestorationNet.")
        elif self.current_epoch == self.decom_epochs + self.restoration_epochs:
            self.restoration_net.requires_grad_(False)
            self.adjustment_net.requires_grad_(True)
            self.extra_logger.info(f"RestorationNet training finished, start training AdjustmentNet.")

    def on_fit_start(self):
        self.extra_logger.info(f"Start training on {self.device}.")

    def on_fit_end(self):
        self.extra_logger.info(f"All training finished.")

    def test_step(self, batch, batch_idx):
        low, high = batch["low"], batch["high"]
        self.forward(low, high)
        high_pred = self.get_high_pred()
        self._compute_loss(low, high)
        self._compute_metrics(high_pred, high)

        self.log_dict({
            "test/decom_loss": self.decom_loss_val,
            "test/restoration_loss": self.restoration_loss_val,
            "test/adjustment_loss": self.adjustment_loss_val,
            "test/psnr": self.psnr_val,
            "test/ssim": self.ssim_val,
            "test/niqe": self.niqe_val,
        }, on_step=False, on_epoch=True, batch_size=low.shape[0])

        if batch_idx == 0:
            low = low[0].detach().cpu()
            high = high[0].detach().cpu()
            high_pred = high_pred[0].detach().cpu()
            image = torch.cat([low, high_pred, high], dim=2)
            image = torch.clip(image * 255, 0, 255).to(torch.uint8)
            self.logger.experiment.add_image("test/image", image, self.current_epoch)

    def on_test_start(self):
        self.extra_logger.info("Start testing.")

    def on_test_end(self):
        self.extra_logger.info(f"Testing finished.")

    def predict_step(self, batch, batch_idx):
        low = batch["low"]
        self.forward(low)
        high_pred = self.get_high_pred()
        high_pred = torch.clip(high_pred * 255, 0, 255).to(torch.uint8)
        save_batch_tensor(high_pred, self.save_path, batch)

# endregion
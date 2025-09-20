from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
import loguru
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio, UniversalImageQualityIndex
from typing import Tuple, List, Optional

from src.llie.metrics.ssim import SSIM
from src.llie.utils.config import get_optimizer, get_scheduler


# ========================================================================================================
#                              Sparse Gradient Minimization Network (SGMN)
# ========================================================================================================

class BasicBlock(nn.Module):
    def __init__(self, in_channels: int = 1, mid_channels: int = 8):
        super().__init__()
        self.in_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, stride=1, dilation=1)
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, stride=1, dilation=1)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=2, stride=1, dilation=2)
        self.conv3 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=3, stride=1, dilation=3)
        self.out_conv = nn.Conv2d(mid_channels, in_channels, kernel_size=3, padding=1, stride=1, dilation=1)

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.in_conv(x))
        x = self.relu(self.conv1(x)) + x
        x = self.relu(self.conv2(x)) + x
        x = self.relu(self.conv3(x)) + x
        x = self.relu(self.out_conv(x))
        return x


class SGMN(nn.Module):
    def __init__(self, depth: int, in_channels: int = 1, mid_channels: int = 8):
        super().__init__()
        self.layers = nn.ModuleList([
            BasicBlock(in_channels, mid_channels)
            for _ in range(depth)
        ])
        # self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        output = []
        for layer in self.layers:
            x = layer(x)
            output.append(x)
        return output


# ========================================================================================================
#                               Multi-Scale Residual Dense Network (MSRDN)
# ========================================================================================================

class DBCR(nn.Module):
    """
    Conv-ReLU block for DenseBlock
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, stride=1),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layer(x)
        return torch.cat([x, out], dim=1)


class DenseBlock(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int, num_layers: int, kernel_size: int = 3):
        super().__init__()
        self.layer = nn.Sequential(*[
            DBCR(in_channels + i * mid_channels, mid_channels, kernel_size)
            for i in range(num_layers)
        ])
        self.conv = nn.Conv2d(in_channels + num_layers * mid_channels, in_channels, kernel_size=1, padding=0, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.layer(x)) + x


class RDB(nn.Module):
    """
    Residual Dense Block
    """
    def __init__(self, in_channels: int, mid_channels: int, num_layers: int, depth: int, kernel_size: int = 3):
        super().__init__()
        self.layers = nn.ModuleList([
            DenseBlock(in_channels, mid_channels, num_layers, kernel_size)
            for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = []
        for layer in self.layers:
            x = layer(x)
            out.append(x)
        return torch.cat(out, dim=1)


class RDN(nn.Module):
    """
    Residual Dense Network
    """
    def __init__(self, in_channels: int, mid_channels: int, num_layers: int, depth: int, kernel_size: int = 3):
        super().__init__()
        self.rdb = RDB(in_channels, mid_channels, num_layers, depth, kernel_size)
        self.compress = nn.Conv2d(depth * in_channels, in_channels, kernel_size=1, padding=0, stride=1)
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, in_channels, kernel_size, padding=(kernel_size - 1) // 2, stride=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.compress(self.rdb(x)) + x
        return self.up(x)


class MSRDN(nn.Module):
    """
    Multi-Scale Residual Dense Network
    """
    def __init__(self, in_channels: int, mid_channels: int, num_layers: int, depth: int, kernel_size: int = 3):
        super().__init__()

        keep_size_pad = (kernel_size - 1) // 2

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size, padding=keep_size_pad, stride=1),
            nn.Conv2d(mid_channels, mid_channels, kernel_size, padding=keep_size_pad, stride=1)
        )   # Shallow feature extraction

        self.rdb0 = RDB(mid_channels, mid_channels // 2, num_layers, depth, kernel_size)
        self.rdn1 = RDN(mid_channels, mid_channels // 2, num_layers, depth, kernel_size)
        self.rdn2 = RDN(mid_channels, mid_channels // 2, num_layers, depth, kernel_size)
        self.rdn3 = RDN(mid_channels, mid_channels // 2, num_layers, depth, kernel_size)

        self.down1 = nn.Conv2d(mid_channels, mid_channels, kernel_size, padding=keep_size_pad, stride=2)
        self.down2 = nn.Conv2d(mid_channels, mid_channels, kernel_size, padding=keep_size_pad, stride=2)
        self.down3 = nn.Conv2d(mid_channels, mid_channels, kernel_size, padding=keep_size_pad, stride=2)

        self.out_conv = nn.Sequential(
            nn.Conv2d(depth * mid_channels, mid_channels, kernel_size=1, padding=0, stride=1),
            nn.Conv2d(mid_channels, mid_channels, kernel_size, padding=keep_size_pad, stride=1)
        )   # Global Feature Fusion
        self.up_net = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels // 2, kernel_size, padding=keep_size_pad, stride=1),
            nn.Conv2d(mid_channels // 2, in_channels, kernel_size, padding=keep_size_pad, stride=1)
        )

        # self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_conv(x)
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)

        x3 = self.rdn3(x3)

        x2 = x2 + x3
        x2 = self.rdn2(x2)

        x1 = x1 + x2
        x1 = self.rdn1(x1)

        identity = x
        x = x + x1
        x = self.out_conv(self.rdb0(x)) + identity

        return self.up_net(x)


# ========================================================================================================
#                                             RetinexNetV2
# ========================================================================================================

class RetinexNetV2(pl.LightningModule):
    def __init__(self, config, logger: "loguru.Logger"):
        super().__init__()

        self.config = config
        model_config = config['model']
        self.decom_net_config = model_config['decom_net']
        self.enhance_net_config = model_config['enhance_net']
        self.restore_net_config = model_config['restore_net']
        self.lambda_IE = model_config.get('lambda_IE', 1)
        self.lambda_RF = model_config.get('lambda_RF', 1)
        self.lambda_FF = model_config.get('lambda_FF', 100)
        self.lambda_rect = model_config.get('lambda_rect', 0.01)
        self.lambda_sparse = model_config.get('lambda_sparse', 1e-7)
        self.decom_epochs = model_config.get('decom_epochs', 30)

        self.decom_net = self._make_sgmn(self.decom_net_config)
        self.enhance_net = self._make_msrdn(self.enhance_net_config)
        self.restore_net = self._make_msrdn(self.restore_net_config)

        self.ssim = SSIM(window_size=11)
        self.l1_loss = nn.L1Loss()
        self.extra_logger = logger
        self.automatic_optimization = False

        # metrics
        self._ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self._psnr = PeakSignalNoiseRatio(data_range=1.0)
        self._uqi = UniversalImageQualityIndex()

    @staticmethod
    def _make_sgmn(config):
        depth = config['depth']
        in_channels = config['in_channels']
        mid_channels = config['mid_channels']
        return SGMN(depth, in_channels, mid_channels)

    @staticmethod
    def _make_msrdn(config):
        in_channels = config['in_channels']
        mid_channels = config['mid_channels']
        num_layers = config['num_layers']
        depth = config['depth']
        kernel_size = config['kernel_size']
        return MSRDN(in_channels, mid_channels, num_layers, depth, kernel_size)

    def forward(self, low: torch.Tensor, high: Optional[torch.Tensor] = None):
        low = (low + 0.05) / 1.05
        self.low_max = torch.max(low, dim=1, keepdim=True)[0].detach()
        self.low_y = self.decom_net(self.low_max)
        low_l = torch.clip(self.low_y[-1], 0.05 / 1.05, 1)
        self.low_l = torch.cat([low_l, low_l, low_l], dim=1)

        if high is not None:
            high = (high + 0.05) / 1.05
            self.high_max = torch.max(high, dim=1, keepdim=True)[0].detach()
            self.high_y = self.decom_net(self.high_max)
            high_l = torch.clip(self.high_y[-1], 0.05 / 1.05, 1)
            self.high_l = torch.cat([high_l, high_l, high_l], dim=1)

        low_r = (low / self.low_l).detach()
        self.high_l_pred = self.enhance_net(low)
        self.high_r_pred = self.restore_net(low_r)
        self.high_pred = self.high_l_pred.detach() * self.high_r_pred

    def _sparse_loss(self, y: List[torch.Tensor], eps_list: List[float]):
        D_x = torch.tensor([[
            [1, 0, -1],
            [1, 0, -1],
            [1, 0, -1],
        ]]).view(1, 1, 3, 3).to(y[0])
        D_y = torch.tensor([[
            [1, 1, 1],
            [0, 0, 0],
            [-1, -1, -1],
        ]]).view(1, 1, 3, 3).to(y[0])

        weight_x, weight_y = 1, 1
        sparse_loss = 0.
        for i in range(len(y)):
            eps_i = eps_list[i]
            y_i_x = F.conv2d(y[i], D_x)
            y_i_y = F.conv2d(y[i], D_y)
            coef = 1 - 0.9 ** self.current_epoch
            if i == 0:
                sparse_loss += coef * torch.sum(torch.abs(weight_x * y_i_x) + torch.abs(weight_y * y_i_y) + 0.001)
            else:
                sparse_loss += coef * torch.sum(torch.abs(weight_x * y_i_x + eps_i) + torch.abs(weight_y * y_i_y + eps_i))
            weight_x = (1 / (torch.abs(y_i_x) + eps_i)).detach()
            weight_y = (1 / (torch.abs(y_i_y) + eps_i)).detach()
        return sparse_loss

    def _rect_loss(self, y: List[torch.Tensor], x_max: torch.Tensor):
        loss = 0.
        for i in range(len(y)):
            y_i = y[i]
            loss += self.l1_loss(y_i, x_max)
        return loss

    def _compute_loss(self, high: torch.Tensor):
        loss_IE = -self.ssim(self.high_l_pred, self.high_l.detach())
        loss_FF = -self.ssim(self.high_pred, high.detach())
        loss_RF = -self.ssim(self.high_r_pred, (high / self.high_l_pred).detach())
        self.map_loss = self.lambda_IE * loss_IE + self.lambda_RF * loss_RF + self.lambda_FF * loss_FF

        eps_list_low = [0.03, 0.06, 0.12, 0.18]
        eps_list_high = [0.09, 0.18, 0.27, 0.36]
        loss_sparse_low = self._sparse_loss(self.low_y, eps_list_low)
        loss_sparse_high = self._sparse_loss(self.high_y, eps_list_high)
        self.sparse_loss = self.lambda_sparse * (loss_sparse_low + loss_sparse_high)

        rect_loss_low = self._rect_loss(self.low_y, self.low_max)
        rect_loss_high = self._rect_loss(self.high_y, self.high_max)
        self.rect_loss = self.lambda_rect * (rect_loss_low + rect_loss_high)

        self.total_loss = self.map_loss + self.sparse_loss + self.rect_loss

    def _compute_metrics(self, high: torch.Tensor):
        pred = torch.clip(self.high_pred, 0, 1.0)
        target = torch.clip(high.detach(), 0, 1.0)
        self.ssim_val = self._ssim(pred, target)
        self.psnr_val = self._psnr(pred, target)
        self.uqi_val = self._uqi(pred, target)
        if torch.isnan(self.ssim_val) or torch.isnan(self.psnr_val) or torch.isnan(self.uqi_val):
            import pdb; pdb.set_trace()
            print("nan detected")

    def configure_optimizers(self):
        train_config = self.config['train']
        decom_optimizer = get_optimizer(train_config, self.decom_net, self.extra_logger)
        enhance_optimizer = get_optimizer(train_config, self.enhance_net, self.extra_logger)
        restore_optimizer = get_optimizer(train_config, self.restore_net, self.extra_logger)
        decom_scheduler = get_scheduler(train_config, decom_optimizer, self.extra_logger)
        enhance_scheduler = get_scheduler(train_config, enhance_optimizer, self.extra_logger)
        restore_scheduler = get_scheduler(train_config, restore_optimizer, self.extra_logger)

        return [decom_optimizer, enhance_optimizer, restore_optimizer], [decom_scheduler, enhance_scheduler, restore_scheduler]

    def on_train_epoch_start(self):
        self.extra_logger.info(f"Epoch {self.current_epoch} starts.")
        self.extra_logger.info(f"Start training stage.")

    def training_step(self, batch, batch_idx):
        high, low = batch
        self.forward(low, high)
        self._compute_loss(high)
        self._compute_metrics(high)

        decom_optimizer, enhance_optimizer, restore_optimizer = self.optimizers()
        decom_optimizer.zero_grad()
        enhance_optimizer.zero_grad()
        restore_optimizer.zero_grad()

        self.manual_backward(self.total_loss)

        if self.current_epoch < self.decom_epochs:
            decom_optimizer.step()
        enhance_optimizer.step()
        restore_optimizer.step()

        self.log_dict({
            "train/map_loss": self.map_loss,
            "train/sparse_loss": self.sparse_loss,
            "train/rect_loss": self.rect_loss,
            "train/total_loss": self.total_loss,
            "train/ssim": self.ssim_val,
            "train/psnr": self.psnr_val,
            "train/uqi": self.uqi_val,
        }, on_step=False, on_epoch=True)

        return self.total_loss

    def on_validation_epoch_start(self):
        self.extra_logger.info(f"Start validation stage.")

    def validation_step(self, batch, batch_idx):
        high, low = batch
        self.forward(low, high)
        self._compute_loss(high)
        self._compute_metrics(high)

        self.log_dict({
            "val/map_loss": self.map_loss,
            "val/sparse_loss": self.sparse_loss,
            "val/rect_loss": self.rect_loss,
            "val/total_loss": self.total_loss,
            "val/ssim": self.ssim_val,
            "val/psnr": self.psnr_val,
            "val/uqi": self.uqi_val,
        }, on_step=False, on_epoch=True)

        # log images
        if batch_idx == 0:
            low, high = low[0].detach().cpu(), high[0].detach().cpu()
            low_l, low_r = self.high_l_pred[0].detach().cpu(), self.high_r_pred[0].detach().cpu()
            high_l = self.high_l[0].detach().cpu()
            high_r = high / high_l
            high_pred = self.high_pred[0].detach().cpu()

            low_image = torch.cat([low, low_l, low_r], dim=1)
            high_image = torch.cat([high, high_l, high_r], dim=1)
            image = torch.cat([low_image, high_image], dim=2)
            image = torch.clip(image * 255, 0, 255).to(torch.uint8)
            out = torch.clip(high_pred * 255, 0, 255).to(torch.uint8)
            self.logger.experiment.add_image("val/image", image)
            self.logger.experiment.add_image("val/out", out)

        return self.total_loss

    def on_validation_epoch_end(self):
        decom_scheduler, enhance_scheduler, restore_scheduler = self.lr_schedulers()
        if self.current_epoch < self.decom_epochs:
            decom_scheduler.step(self.total_loss)
        enhance_scheduler.step(self.total_loss)
        restore_scheduler.step(self.total_loss)

        self.log("lr", enhance_scheduler.get_last_lr()[0])

        if self.current_epoch + 1 == self.decom_epochs:
            self.extra_logger.info(f"DecomNet training finished.")
            self.decom_net.freeze()

    def on_fit_start(self):
        self.extra_logger.info(f"Start training on {self.device}.")

    def on_fit_end(self):
        self.extra_logger.info(f"Training finished.")

    def test_step(self, batch, batch_idx):
        high, low = batch
        self.forward(low, high)
        self._compute_loss(high)
        self._compute_metrics(high)

        self.log_dict({
            "test/map_loss": self.map_loss,
            "test/sparse_loss": self.sparse_loss,
            "test/rect_loss": self.rect_loss,
            "test/total_loss": self.total_loss,
            "test/ssim": self.ssim_val,
            "test/psnr": self.psnr_val,
            "test/uqi": self.uqi_val,
        }, on_step=False, on_epoch=True)

        # log images
        if batch_idx == 0:
            low, high = low[0].detach().cpu(), high[0].detach().cpu()
            low_l, low_r = self.high_l_pred[0].detach().cpu(), self.high_r_pred[0].detach().cpu()
            high_l = self.high_l[0].detach().cpu()
            high_r = high / high_l
            high_pred = self.high_pred[0].detach().cpu()

            low_image = torch.cat([low, low_l, low_r], dim=1)
            high_image = torch.cat([high, high_l, high_r], dim=1)
            image = torch.cat([low_image, high_image], dim=2)
            image = torch.clip(image * 255, 0, 255).to(torch.uint8)
            out = torch.clip(high_pred * 255, 0, 255).to(torch.uint8)
            self.logger.experiment.add_image("test/image", image)
            self.logger.experiment.add_image("test/out", out)

        return self.total_loss

    def on_test_start(self):
        self.extra_logger.info("Start testing.")

    def on_test_end(self):
        self.extra_logger.info(f"Testing finished.")

    def predict_step(self, batch, batch_idx):
        _, low = batch
        self.forward(low)
        return torch.clip(self.high_pred.detach().cpu() * 255, 0, 255).to(torch.uint8)
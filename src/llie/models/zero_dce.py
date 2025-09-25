from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
import loguru
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.regression import MeanAbsoluteError

from src.llie.utils.config import get_optimizer, get_scheduler


# ========================================================================================================
#                                             Loss functions
# ========================================================================================================

class SpatialConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        left_kernel = torch.FloatTensor([[0, 0, 0],
                                         [-1, 1, 0],
                                         [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        right_kernel = torch.FloatTensor([[0, 0, 0],
                                          [0, 1, -1],
                                          [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        up_kernel = torch.FloatTensor([[0, -1, 0],
                                       [0, 1, 0],
                                       [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        down_kernel = torch.FloatTensor([[0, 0, 0],
                                         [0, 1, 0],
                                         [0, -1, 0]]).unsqueeze(0).unsqueeze(0)

        self.left_weight = nn.Parameter(data=left_kernel, requires_grad=False)
        self.right_weight = nn.Parameter(data=right_kernel, requires_grad=False)
        self.up_weight = nn.Parameter(data=up_kernel, requires_grad=False)
        self.down_weight = nn.Parameter(data=down_kernel, requires_grad=False)

        self.pool = nn.AvgPool2d(4)

    def forward(self, x: torch.Tensor, enhanced: torch.Tensor) -> torch.Tensor:
        self.left_weight = self.left_weight.to(x.device)
        self.right_weight = self.right_weight.to(x.device)
        self.up_weight = self.up_weight.to(x.device)
        self.down_weight = self.down_weight.to(x.device)

        pred_mean = torch.mean(x, dim=1, keepdim=True)
        target_mean = torch.mean(enhanced, dim=1, keepdim=True)
        pred_pool = self.pool(pred_mean)
        target_pool = self.pool(target_mean)

        diff_pred_left = F.conv2d(pred_pool, self.left_weight, padding=1)
        diff_pred_right = F.conv2d(pred_pool, self.right_weight, padding=1)
        diff_pred_up = F.conv2d(pred_pool, self.up_weight, padding=1)
        diff_pred_down = F.conv2d(pred_pool, self.down_weight, padding=1)

        diff_target_left = F.conv2d(target_pool, self.left_weight, padding=1)
        diff_target_right = F.conv2d(target_pool, self.right_weight, padding=1)
        diff_target_up = F.conv2d(target_pool, self.up_weight, padding=1)
        diff_target_down = F.conv2d(target_pool, self.down_weight, padding=1)

        diff_left = (diff_pred_left - diff_target_left) ** 2
        diff_right = (diff_pred_right - diff_target_right) ** 2
        diff_up = (diff_pred_up - diff_target_up) ** 2
        diff_down = (diff_pred_down - diff_target_down) ** 2

        return torch.mean(diff_left + diff_right + diff_up + diff_down)


class ExposureControlLoss(nn.Module):
    def __init__(self, patch_size, mean_val):
        super().__init__()

        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = torch.FloatTensor([mean_val])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.mean_val = self.mean_val.to(x.device)
        mean = self.pool(torch.mean(x, dim=1, keepdim=True))
        return torch.mean((mean - self.mean_val) ** 2)


class ColorConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean_rgb = torch.mean(x, dim=[2, 3], keepdim=True)
        mean_r, mean_g, mean_b = torch.split(mean_rgb, 1, dim=1)
        diff_rg = (mean_r - mean_g) ** 2
        diff_rb = (mean_r - mean_b) ** 2
        diff_gb = (mean_g - mean_b) ** 2
        return torch.sqrt(diff_rg ** 2 + diff_rb ** 2 + diff_gb ** 2).mean()


class IlluminationSmoothnessLoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        h_count = (h - 1) * w
        w_count = h * (w - 1)
        h_diff = (x[:, :, 1:, :] - x[:, :, :-1, :]) ** 2
        w_diff = (x[:, :, :, 1:] - x[:, :, :, :-1]) ** 2
        h_loss = torch.sum(h_diff) / h_count
        w_loss = torch.sum(w_diff) / w_count
        return 2 * self.weight * (h_loss + w_loss) / b

# ========================================================================================================
#                                               Main Model
# ========================================================================================================

class DCENet(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 32, n_iterations: int = 8):
        super().__init__()

        self.in_channels = in_channels
        self.n_iterations = n_iterations

        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv4 = nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv5 = nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv6 = nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv7 = nn.Conv2d(base_channels * 2, n_iterations * in_channels, kernel_size=3, stride=1, padding=1, bias=True)

        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0., 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1., 0.02)
                m.bias.data.fill_(0)

    def forward(self, x: torch.Tensor, return_a: bool = False):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))

        x5 = self.relu(self.conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.conv6(torch.cat([x2, x5], 1)))
        x7 = self.tanh(self.relu(self.conv7(torch.cat([x1, x6], 1))))

        a_list = torch.split(x7, self.in_channels, dim=1)
        for a in a_list:
            x = x + a * (x ** 2 - x)
        a = torch.cat(a_list, dim=1)

        if return_a:
            return x, a
        return x


class ZeroDCE(pl.LightningModule):
    def __init__(self, config, logger: "loguru.Logger"):
        super().__init__()

        self.config = config
        model_config = config["model"]
        self.in_channels = model_config.get('in_channels', 3)
        self.base_channels = model_config.get('base_channels', 32)
        self.n_iterations = model_config.get('n_iterations', 8)
        self.dce_net = DCENet(in_channels=self.in_channels, base_channels=self.base_channels, n_iterations=self.n_iterations)
        self.extra_logger = logger
        # self.automatic_optimization = False

        # loss functions
        self.loss_spa = SpatialConsistencyLoss()
        self.loss_exp = ExposureControlLoss(patch_size=16, mean_val=0.6)
        self.loss_col = ColorConsistencyLoss()
        self.loss_tv = IlluminationSmoothnessLoss()
        self.lambda_spa = model_config.get('lambda_spa', 0.1)
        self.lambda_exp = model_config.get('lambda_exp', 1.0)
        self.lambda_col = model_config.get('lambda_col', 0.5)
        self.lambda_tv = model_config.get('lambda_tv', 20.0)

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.mae = MeanAbsoluteError()

    def forward(self, x: torch.Tensor, return_a: bool = False):
        return self.dce_net(x, return_a)

    def configure_optimizers(self):
        train_config = self.config["train"]
        optimizer = get_optimizer(train_config, self.dce_net, self.extra_logger)
        scheduler = get_scheduler(train_config, optimizer, self.extra_logger)
        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val/loss",
            "strict": True,
            "name": None,
        }
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler_config,
        }

    def _compute_loss(self, x: torch.Tensor, enhanced: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        self.loss_spa_val = self.lambda_spa * self.loss_spa(x, enhanced)
        self.loss_exp_val = self.lambda_exp * self.loss_exp(enhanced)
        self.loss_col_val = self.lambda_col * self.loss_col(enhanced)
        self.loss_tv_val = self.lambda_tv * self.loss_tv(a)
        return self.loss_spa_val + self.loss_exp_val + self.loss_col_val + self.loss_tv_val

    def on_train_epoch_start(self):
        self.extra_logger.info(f"Epoch {self.current_epoch} starts.")
        self.extra_logger.info(f"Start training stage.")

    def training_step(self, batch, batch_idx):
        image = batch["low"]
        enhanced_image, a = self.forward(image, return_a=True)
        loss = self._compute_loss(image, enhanced_image, a)
        self.log_dict({
            "train/loss_spa": self.loss_spa_val,
            "train/loss_exp": self.loss_exp_val,
            "train/loss_col": self.loss_col_val,
            "train/loss_tv": self.loss_tv_val,
            "train/loss": loss,
        }, on_step=False, on_epoch=True, batch_size=image.shape[0])
        return loss

    def on_validation_epoch_start(self):
        self.extra_logger.info(f"Start validation stage.")

    def validation_step(self, batch, batch_idx):
        image = batch["low"]
        enhanced_image, a = self.forward(image, return_a=True)
        loss = self._compute_loss(image, enhanced_image, a)
        self.log_dict({
            "val/loss_spa": self.loss_spa_val,
            "val/loss_exp": self.loss_exp_val,
            "val/loss_col": self.loss_col_val,
            "val/loss_tv": self.loss_tv_val,
            "val/loss": loss,
        }, on_step=False, on_epoch=True, batch_size=image.shape[0])

        if batch_idx == 0:
            image = image[0].detach().cpu()
            enhanced_image = enhanced_image[0].detach().cpu()
            out = torch.cat([image, enhanced_image], dim=2)
            out = torch.clip(out * 255, 0, 255).to(torch.uint8)
            self.logger.experiment.add_image(f"val/image", out, self.current_epoch)

        return loss

    def on_validation_epoch_end(self):
        scheduler = self.lr_schedulers()
        self.log("lr", scheduler.get_last_lr()[0], on_step=False, on_epoch=True)

    def on_fit_start(self):
        self.extra_logger.info(f"Start training on {self.device}.")

    def on_fit_end(self):
        self.extra_logger.info(f"Training finished.")

    def test_step(self, batch, batch_idx):
        image = batch["low"]
        enhanced_image, a = self.forward(image, return_a=True)
        loss = self._compute_loss(image, enhanced_image, a)
        self.log_dict({
            "test/loss_spa": self.loss_spa_val,
            "test/loss_exp": self.loss_exp_val,
            "test/loss_col": self.loss_col_val,
            "test/loss_tv": self.loss_tv_val,
            "test/loss": loss,
        }, on_step=False, on_epoch=True, batch_size=image.shape[0])

        if batch_idx == 0:
            image = image[0].detach().cpu()
            enhanced_image = enhanced_image[0].detach().cpu()
            out = torch.cat([image, enhanced_image], dim=2)
            out = torch.clip(out * 255, 0, 255).to(torch.uint8)
            self.logger.experiment.add_image(f"test/image", out, self.current_epoch)

        return loss

    def on_test_start(self):
        self.extra_logger.info("Start testing.")

    def on_test_end(self):
        self.extra_logger.info(f"Testing finished.")

    def predict_step(self, batch, batch_idx):
        return self.forward(batch["low"])
from __future__ import annotations

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Grayscale
from einops import rearrange
import lightning as pl
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from typing import Optional

from src.llie.utils.logger import default_logger as extra_logger
from src.llie.utils.config import get_optimizer, get_scheduler
from src.llie.utils.loss import PerceptualLoss
from src.llie.models.utils import save_batch_tensor


# region Basic Blocks

class ResidualBlock(nn.Module):
    def __init__(self, base_channels: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        self._init_weights()

    def _init_weights(self, scale: float = 1.0):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = out + residual
        return out


class ResidualBlocks(nn.Module):
    def __init__(self, num_blocks: int, base_channels: int = 64):
        super().__init__()
        self.blocks = nn.Sequential(*[ResidualBlock(base_channels) for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)

# endregion

# region Transformer

class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, k_dim: int, v_dim: int, scale: Optional[float] = None,
                 dropout: float = 0.1, attn_dropout: float = 0.0):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.scale = scale or (k_dim ** -0.5)

        self.q_proj = nn.Linear(dim, num_heads * k_dim, bias=False)
        self.k_proj = nn.Linear(dim, num_heads * k_dim, bias=False)
        self.v_proj = nn.Linear(dim, num_heads * v_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * v_dim, dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x

        x = self.layer_norm(x)
        q = rearrange(self.q_proj(x), 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(self.k_proj(x), 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(self.v_proj(x), 'b n (h d) -> b h n d', h=self.num_heads)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        score = self.attn_dropout(self.softmax(attn))
        attn = score @ v

        attn = rearrange(attn, 'b h n d -> b n (h d)')
        attn = self.dropout(self.out_proj(attn))
        attn = attn + residual
        return attn, score


class FeedForward(nn.Module):
    def __init__(self, dim: int, mlp_ratio: int = 2, dropout: float = 0.1):
        super().__init__()

        self.layers = nn.Sequential(
            nn.LayerNorm(dim, eps=1e-6),
            nn.Linear(dim, dim * mlp_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.layers(x)
        x = x + residual
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, k_dim: int, v_dim: int, scale: Optional[float] = None,
                 dropout: float = 0.0, attn_dropout: float = 0.0, mlp_ratio: int = 2):
        super().__init__()
        self.attention = MultiHeadAttention(dim, num_heads, k_dim, v_dim, scale, dropout, attn_dropout)
        self.feed_forward = FeedForward(dim, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x, score = self.attention(x, mask)
        x = self.feed_forward(x)
        return x, score


class Transformer(nn.Module):
    def __init__(self, dim: int, num_layers: int, num_heads: int, k_dim: int = 64, v_dim: int = 64,
                 scale: Optional[float] = None, dropout: float = 0.0, attn_dropout: float = 0.0, mlp_ratio: int = 2):
        super().__init__()

        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, k_dim, v_dim, scale, dropout, attn_dropout, mlp_ratio)
            for _ in range(num_layers)
        ])

        self.unfold = nn.Unfold(kernel_size=5, dilation=1, stride=5, padding=0)

    def partition(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.unfold(x).permute(0, 2, 1)
        if mask is not None:
            mask = self.unfold(mask).permute(0, 2, 1)
            mask = mask.mean(2).unsqueeze(-2)
            mask[mask <= 0.5] = 0.0
        return x, mask

    def merge(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = F.fold(x, output_size=(h, w), kernel_size=5, dilation=1, stride=5, padding=0)
        return x

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h, w = x.shape[-2:]
        x, mask = self.partition(x, mask)
        for block in self.blocks:
            x, score = block(x, mask)
        x = self.merge(x, h, w)
        return x

# endregion

# region Loss Functions

class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.sum(torch.sqrt(diff * diff + self.eps))

# endregion

# region Main Model

class SNRModel(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 64, encoder_blocks: int = 5, decoder_blocks: int = 10,
                 res_blocks: int = 6, dim: int = 1024, num_layers: int = 6, num_heads: int = 8):
        super().__init__()

        # encoder
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=2, padding=1)
        self.feature_extraction = ResidualBlocks(encoder_blocks, base_channels)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.residual_branch = ResidualBlocks(res_blocks, base_channels)
        self.transformer = Transformer(dim, num_layers, num_heads)

        # decoder
        self.reconstruction = ResidualBlocks(decoder_blocks, base_channels)
        self.up_conv1 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=1, padding=1)
        self.up_conv2 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.hr_conv = nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, stride=1, padding=1)
        self.out_conv = nn.Conv2d(base_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # encoder
        x_center = x
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(x1))
        x3 = self.lrelu(self.conv3(x2))
        features = self.feature_extraction(x3)

        # residual branch
        fea_short = self.residual_branch(features)

        # transformer branch
        h_fea, w_fea = features.shape[-2:]
        mask = F.interpolate(mask, size=(h_fea, w_fea), mode='nearest') if mask is not None else None
        fea_long = self.transformer(features, mask)

        # fusion
        features = fea_long * (1 - mask) + fea_short * mask

        # decoder
        x = self.reconstruction(features)
        x = self.lrelu(self.pixel_shuffle(self.up_conv1(torch.cat([x, x3], dim=1))))
        x = self.lrelu(self.pixel_shuffle(self.up_conv2(torch.cat([x, x2], dim=1))))
        x = self.lrelu(self.hr_conv(torch.cat([x, x1], dim=1)))
        x = self.out_conv(x)
        x = x + x_center

        return x


class SNR(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config
        model_config = config["model"]
        self.snr_model = SNRModel(
            in_channels=model_config["in_channels"],
            base_channels=model_config["base_channels"],
            encoder_blocks=model_config["encoder_blocks"],
            decoder_blocks=model_config["decoder_blocks"],
            res_blocks=model_config["res_blocks"],
            dim=model_config["dim"],
            num_layers=model_config["num_layers"],
            num_heads=model_config["num_heads"]
        )

        self.save_path = config["data"].get("save_path", "")
        if self.save_path:
            os.makedirs(self.save_path, exist_ok=True)

        # loss functions
        self.lambda_vgg = model_config["lambda_vgg"]
        self.charbonnier_loss = CharbonnierLoss()
        self.perceptual_loss = PerceptualLoss(weights=[1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0], criterion="l1")

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

    @staticmethod
    def get_mask(x: torch.Tensor):
        rgb_to_gray = Grayscale()
        blur = F.conv2d(x, weight=torch.ones((1, 3, 5, 5)).to(x) / 25, padding=2)
        blur = rgb_to_gray(blur)
        gray_x = rgb_to_gray(x)
        noise = torch.abs(gray_x - blur)

        mask = blur / (noise + 1e-4)
        b, _, h, w = mask.shape
        mask_max = torch.max(mask.view(b, -1), dim=1)[0]
        mask_max = mask_max.view(b, 1, 1, 1)
        mask_max = mask_max.repeat(1, 1, h, w)
        mask = mask / (mask_max + 1e-4)
        mask = torch.clip(mask, 0, 1).float()

        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = self.get_mask(x)
        return self.snr_model(x, mask)

    def configure_optimizers(self):
        train_config = self.config["train"]
        optimizer = get_optimizer(train_config, self.snr_model)
        scheduler = get_scheduler(train_config, optimizer)
        return [optimizer], [scheduler]

    def _compute_loss(self, high_pred: torch.Tensor, high: torch.Tensor):
        self.loss = self.charbonnier_loss(high_pred, high) + self.lambda_vgg * self.perceptual_loss(high_pred, high)

    def _compute_metrics(self, high_pred: torch.Tensor, high: torch.Tensor):
        high_pred = torch.clip(high_pred, 0, 1)
        self.psnr_val = self.psnr(high_pred, high)
        self.ssim_val = self.ssim(high_pred, high)

    def on_train_epoch_start(self):
        extra_logger.info(f"Epoch {self.current_epoch} starts.")
        extra_logger.info(f"Start training stage.")

    def training_step(self, batch, batch_idx):
        low, high = batch["low"], batch["high"]
        high_pred = self.forward(low)
        self._compute_loss(high_pred, high)

        self.log("train/loss", self.loss, on_step=True, on_epoch=True, batch_size=high.shape[0])
        return self.loss

    def on_validation_epoch_start(self):
        extra_logger.info(f"Start validation stage.")

    def validation_step(self, batch, batch_idx):
        low, high = batch["low"], batch["high"]
        high_pred = self.forward(low)
        self._compute_loss(high_pred, high)
        self._compute_metrics(high_pred, high)

        self.log_dict({
            "val/loss": self.loss,
            "val/psnr": self.psnr_val,
            "val/ssim": self.ssim_val
        }, on_step=False, on_epoch=True, batch_size=high.shape[0])

        if batch_idx == 0:
            low = low[0].detach().cpu()
            high = high[0].detach().cpu()
            high_pred = high_pred[0].detach().cpu()
            image = torch.cat([low, high_pred, high], dim=2)
            image = torch.clip(image * 255, 0, 255).to(torch.uint8)
            self.logger.experiment.add_image("val/image", image, self.current_epoch)

    def on_validation_epoch_end(self):
        scheduler = self.lr_schedulers()
        self.log("lr", scheduler.get_last_lr()[0], on_step=False, on_epoch=True)

    def on_fit_start(self):
        extra_logger.info(f"Start training on {self.device}.")

    def on_fit_end(self):
        extra_logger.info(f"Training finished.")

    def test_step(self, batch, batch_idx):
        low, high = batch["low"], batch["high"]
        high_pred = self.forward(low)
        self._compute_loss(high_pred, high)
        self._compute_metrics(high_pred, high)

        self.log_dict({
            "test/loss": self.loss,
            "test/psnr": self.psnr_val,
            "test/ssim": self.ssim_val
        }, on_step=False, on_epoch=True, batch_size=high.shape[0])

        if batch_idx == 0:
            low = low[0].detach().cpu()
            high = high[0].detach().cpu()
            high_pred = high_pred[0].detach().cpu()
            image = torch.cat([low, high_pred, high], dim=2)
            image = torch.clip(image * 255, 0, 255).to(torch.uint8)
            self.logger.experiment.add_image("test/image", image, self.current_epoch)

    def on_test_start(self):
        extra_logger.info("Start testing.")

    def on_test_end(self):
        extra_logger.info(f"Testing finished.")

    def predict_step(self, batch, batch_idx):
        low = batch["low"]
        high_pred = self.forward(low)
        high_pred = torch.clip(high_pred * 255, 0, 255).to(torch.uint8)
        save_batch_tensor(high_pred, self.save_path, batch)

# endregion
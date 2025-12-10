import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
from einops import rearrange
from timm.models.layers import trunc_normal_
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from typing import Optional, List, Tuple

from src.llie.utils.logger import default_logger as extra_logger
from src.llie.utils.config import get_optimizer, get_scheduler
from src.llie.models.utils import save_batch_tensor
from src.llie.data.utils import MixUp


# region Illumination Estimator

class IlluminationEstimator(nn.Module):
    def __init__(self, in_channels: int, base_channels: int, out_channels: int):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=1)
        self.depth_conv = nn.Conv2d(base_channels, base_channels, kernel_size=5, padding=2, groups=in_channels)
        self.conv2 = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_mean = x.mean(dim=1, keepdim=True)
        x = torch.cat([x, x_mean], dim=1)
        x = self.conv1(x)
        illumination_feature = self.depth_conv(x)
        illumination_map = self.conv2(illumination_feature)

        return illumination_feature, illumination_map

# endregion

# region Denoiser

class PreNorm(nn.Module):
    def __init__(self, dim: int, module: nn.Module):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.module = module

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        x = self.norm(x)
        x = self.module(x, *args, **kwargs)
        return x


class FeedForward(nn.Module):
    def __init__(self, dim: int, scale: int = 4):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(dim, dim * scale, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim * scale, dim * scale, kernel_size=3, stride=1, padding=1, bias=False, groups=dim * scale),
            nn.GELU(),
            nn.Conv2d(dim * scale, dim, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "b h w c -> b c h w")
        x = self.layers(x)
        x = rearrange(x, "b c h w -> b h w c")
        return x


class IGMSA(nn.Module):
    """Illumination-Guided Multi-head Self-Attention"""
    def __init__(self, dim: int, dim_head: int = 64, heads: int = 8):
        super().__init__()

        self.dim = dim
        self.num_heads = heads
        self.dim_head = dim_head
        self.q_proj = nn.Linear(dim, dim_head * heads, bias=False)
        self.k_proj = nn.Linear(dim, dim_head * heads, bias=False)
        self.v_proj = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False, groups=dim),
        )

    def forward(self, x: torch.Tensor, illumination_feature: torch.Tensor) -> torch.Tensor:
        _, h, w, _ = x.shape
        x = rearrange(x, 'b h w c -> b (h w) c')
        illumination_feature = rearrange(illumination_feature, 'b h w c -> b (h w) c')
        q_inp = self.q_proj(x)
        k_inp = self.k_proj(x)
        v_inp = self.v_proj(x)
        q, k, v, illumination_attn = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                         (q_inp, k_inp, v_inp, illumination_feature))

        v = v * illumination_attn
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1)) * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v
        x = rearrange(x, 'b h d n -> b n (h d)')

        x1 = self.proj(x)
        x1 = rearrange(x1, 'b (h w) c -> b h w c', h=h, w=w)
        x2 = self.pos_emb(rearrange(v_inp, 'b (h w) c -> b c h w', h=h, w=w))
        x2 = rearrange(x2, 'b c h w -> b h w c')
        x = x1 + x2

        return x


class IGAB(nn.Module):
    """Illumination-Guided Attention Block"""
    def __init__(self, dim: int, dim_head: int = 64, heads: int = 8, num_blocks: int = 2):
        super().__init__()

        self.attns = nn.ModuleList([IGMSA(dim, dim_head, heads) for _ in range(num_blocks)])
        self.ffs = nn.ModuleList([PreNorm(dim, FeedForward(dim)) for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor, illumination_feature: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, 'b c h w -> b h w c')
        illumination_feature = rearrange(illumination_feature, 'b c h w -> b h w c')
        for attn, ff in zip(self.attns, self.ffs):
            x = attn(x, illumination_feature) + x
            x = ff(x) + x
        x = rearrange(x, 'b h w c -> b c h w')
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels: int = 3, dim: int = 31, level: int = 2, num_blocks: Optional[List[int]] = None):
        super().__init__()

        if num_blocks is None:
            num_blocks = [2, 4, 4]

        self.dim = dim
        self.level = level
        self.embedding = nn.Conv2d(in_channels, self.dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.layers = nn.ModuleList([
            nn.ModuleList([
                IGAB(dim=dim * 2 ** i, num_blocks=num_blocks[i], dim_head=dim, heads=2 ** i),
                nn.Conv2d(dim * 2 ** i, dim * 2 ** (i + 1), kernel_size=4, stride=2, padding=1, bias=False),
                nn.Conv2d(dim * 2 ** i, dim * 2 ** (i + 1), kernel_size=4, stride=2, padding=1, bias=False),
            ]) for i in range(level)
        ])

    def forward(self, x: torch.Tensor, illumination_feature: torch.Tensor):
        x = self.embedding(x)

        features = []
        illumination_features = []

        for (igab, x_down, illumination_down) in self.layers:
            x = igab(x, illumination_feature)
            features.append(x)
            illumination_features.append(illumination_feature)
            x = x_down(x)
            illumination_feature = illumination_down(illumination_feature)

        return x, illumination_feature, features, illumination_features


class Decoder(nn.Module):
    def __init__(self, dim: int = 31, level: int = 2, num_blocks: Optional[List[int]] = None):
        super().__init__()

        if num_blocks is None:
            num_blocks = [2, 4, 4]

        self.level = level
        self.layers = nn.ModuleList([
            nn.ModuleList([
                nn.ConvTranspose2d(dim * 2 ** (level - i), dim * 2 ** (level - i - 1), kernel_size=2, stride=2),
                nn.Conv2d(dim * 2 ** (level - i), dim * 2 ** (level - i - 1), kernel_size=1, stride=1, bias=False),
                IGAB(dim=dim * 2 ** (level - i - 1), num_blocks=num_blocks[level - i - 1], dim_head=dim, heads=2 ** (level - i - 1))
            ]) for i in range(level)
        ])

    def forward(self, x: torch.Tensor, features: List[torch.Tensor], illumination_features: List[torch.Tensor]) -> torch.Tensor:
        for i, (x_up, fusion, igab) in enumerate(self.layers):
            x = x_up(x)
            x = fusion(torch.cat([x, features[self.level - i - 1]], dim=1))
            illumination_feature = illumination_features[self.level - i - 1]
            x = igab(x, illumination_feature)

        return x


class Denoiser(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dim: int = 31,
                 level: int = 2, num_blocks: Optional[List[int]] = None):
        super().__init__()

        if num_blocks is None:
            num_blocks = [2, 4, 4]

        self.encoder = Encoder(in_channels, dim, level, num_blocks)
        self.bottle_neck = IGAB(dim=dim * 2 ** level, dim_head=dim, heads=2 ** level, num_blocks=num_blocks[-1])
        self.decoder = Decoder(dim, level, num_blocks)
        self.out_conv = nn.Conv2d(dim, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        # self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor, illumination_feature: torch.Tensor) -> torch.Tensor:
        feature, illumination_feature, features, illumination_features = self.encoder(x, illumination_feature)
        feature = self.bottle_neck(feature, illumination_feature)
        out = self.decoder(feature, features, illumination_features)
        out = self.out_conv(out) + x
        return out

# endregion

# region RetinexFormer

class RetinexFormerBlock(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, dim: int = 31,
                 level: int = 2, num_blocks: Optional[List[int]] = None):
        super().__init__()

        if num_blocks is None:
            num_blocks = [1, 1, 1]

        self.estimator = IlluminationEstimator(in_channels=in_channels + 1, base_channels=dim, out_channels=out_channels)
        self.denoiser = Denoiser(in_channels, out_channels, dim, level, num_blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        illumination_feature, illumination_map = self.estimator(x)
        x = x * illumination_map + x
        x = self.denoiser(x, illumination_feature)
        return x


class RetinexFormerModel(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, dim: int = 31,
                 stage: int = 3, level: int = 2, num_blocks: Optional[List[int]] = None):
        super().__init__()

        if num_blocks is None:
            num_blocks = [1, 1, 1]

        self.blocks = nn.Sequential(*[
            RetinexFormerBlock(in_channels, out_channels, dim, level, num_blocks)
            for _ in range(stage)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)

# endregion

# region Main Model

class RetinexFormer(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config
        model_config = config['model']
        self.model = RetinexFormerModel(
            in_channels=model_config['in_channels'],
            out_channels=model_config['out_channels'],
            dim=model_config['dim'],
            stage=model_config['stage'],
            level=model_config['level'],
            num_blocks=model_config['num_blocks']
        )

        self.mix_up_aug = MixUp(config["data"]["mix_beta"])
        self.save_path = config["data"].get("save_path", "")
        if self.save_path:
            os.makedirs(self.save_path, exist_ok=True)

        self.criterion = nn.L1Loss()
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def configure_optimizers(self):
        train_config = self.config["train"]
        optimizer = get_optimizer(train_config, self.model)
        scheduler = get_scheduler(train_config, optimizer)
        return [optimizer], [scheduler]

    def _compute_loss(self, y_pred, y):
        self.loss = self.criterion(y_pred, y)

    def _compute_metrics(self, y_pred, y):
        self.psnr_val = self.psnr(y_pred, y)
        self.ssim_val = self.ssim(y_pred, y)

    def on_train_epoch_start(self):
        extra_logger.info(f"Epoch {self.current_epoch} starts.")
        extra_logger.info(f"Start training stage.")

    def training_step(self, batch, batch_idx):
        low, high = batch["low"], batch["high"]
        low, high = self.mix_up_aug(low, high)
        high_pred = self.forward(low)
        self._compute_loss(high_pred, high)
        self.log("train/loss", self.loss, on_step=True, on_epoch=True, batch_size=len(low))
        return self.loss

    def on_train_epoch_end(self):
        scheduler = self.lr_schedulers()
        self.log("lr", scheduler.get_last_lr()[0], on_step=False, on_epoch=True)

    def on_validation_epoch_start(self):
        extra_logger.info(f"Start validation stage.")

    def validation_step(self, batch, batch_idx):
        low, high = batch["low"], batch["high"]
        low, high = self.mix_up_aug(low, high)
        high_pred = self.forward(low)
        self._compute_loss(high_pred, high)
        self._compute_metrics(high_pred, high)
        self.log("val/loss", self.loss, on_step=True, on_epoch=True, batch_size=len(low))
        self.log_dict({
            "val/psnr": self.psnr_val,
            "val/ssim": self.ssim_val
        }, on_step=False, on_epoch=True, batch_size=len(low))

        if batch_idx == 0:
            low = low[0].detach().cpu()
            high = high[0].detach().cpu()
            high_pred = high_pred[0].detach().cpu()
            image = torch.cat([low, high_pred, high], dim=2)
            image = torch.clip(image * 255.0, 0.0, 255.0).to(torch.uint8)

            self.logger.experiment.add_image(f"val/image", image, self.current_epoch)

        return self.loss

    def on_fit_start(self):
        extra_logger.info(f"Start training on {self.device}.")

    def on_fit_end(self):
        extra_logger.info(f"All training finished.")

    def test_step(self, batch, batch_idx):
        low, high = batch["low"], batch["high"]
        low, high = self.mix_up_aug(low, high)
        high_pred = self.forward(low)
        self._compute_loss(high_pred, high)
        self._compute_metrics(high_pred, high)
        self.log("test/loss", self.loss, on_step=True, on_epoch=True, batch_size=len(low))
        self.log_dict({
            "test/psnr": self.psnr_val,
            "test/ssim": self.ssim_val
        }, on_step=False, on_epoch=True, batch_size=len(low))

        if batch_idx == 0:
            low = low[0].detach().cpu()
            high = high[0].detach().cpu()
            high_pred = high_pred[0].detach().cpu()
            image = torch.cat([low, high_pred, high], dim=2)
            image = torch.clip(image * 255.0, 0.0, 255.0).to(torch.uint8)

            self.logger.experiment.add_image(f"test/image", image, self.current_epoch)

        return self.loss

    def on_test_start(self):
        extra_logger.info("Start testing.")

    def on_test_end(self):
        extra_logger.info(f"Testing finished.")

    def predict_step(self, batch, batch_idx):
        low = batch["low"]
        high_pred = self.forward(low)
        high_pred = torch.clip(high_pred * 255.0, 0.0, 255.0).to(torch.uint8)
        save_batch_tensor(high_pred, self.save_path, batch)

# endregion
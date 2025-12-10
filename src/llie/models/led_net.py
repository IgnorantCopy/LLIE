import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import lightning as pl
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, LearnedPerceptualImagePatchSimilarity
from typing import Optional, List

from src.llie.utils.loss import PerceptualLoss
from src.llie.utils.logger import default_logger as extra_logger
from src.llie.utils.config import get_optimizer, get_scheduler
from src.llie.models.utils import save_batch_tensor


# region Encoder

class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int = 3, bias: bool = False):
        super().__init__()

        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, bias=bias),
            nn.PReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, bias=bias),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, bias=bias),
            nn.PReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.conv1(x)
        x = self.conv2(x)
        return x


class Downsample(nn.Module):
    def __init__(self, pad_type: str = "reflect", filter_size: int = 3, stride: int = 2,
                 channels: Optional[int] = None, pad_offset: int = 0):
        super().__init__()

        self.filter_size = filter_size
        self.stride = stride
        self.channels = channels
        self.pad_offset = pad_offset
        pad_sizes = [
            int((filter_size - 1) / 2),
            int(np.ceil((filter_size - 1) / 2)),
            int((filter_size - 1) / 2),
            int(np.ceil((filter_size - 1) / 2)),
        ]
        self.pad_sizes = [pad_size + pad_offset for pad_size in pad_sizes]
        self.offset = int((stride - 1) / 2)

        if filter_size == 1:
            _filter = torch.tensor([1.])
        elif filter_size == 2:
            _filter = torch.tensor([1., 1.])
        elif filter_size == 3:
            _filter = torch.tensor([1., 2., 1.])
        elif filter_size == 4:
            _filter = torch.tensor([1., 3., 3., 1.])
        elif filter_size == 5:
            _filter = torch.tensor([1., 4., 6., 4., 1.])
        elif filter_size == 6:
            _filter = torch.tensor([1., 5., 10., 10., 5., 1.])
        elif filter_size == 7:
            _filter = torch.tensor([1., 6., 15., 20., 15., 6., 1.])
        else:
            raise ValueError(f"Unsupported filter size: {filter_size}")

        _filter = _filter[:, None] * _filter[None, :]
        _filter = _filter / torch.sum(_filter)
        self.register_buffer("filter", _filter[None, None, :, :].repeat((channels, 1, 1, 1)))

        if pad_type == "reflect":
            self.pad = nn.ReflectionPad2d(self.pad_sizes)
        elif pad_type == "replicate":
            self.pad = nn.ReplicationPad2d(self.pad_sizes)
        elif pad_type == "zero":
            self.pad = nn.ZeroPad2d(self.pad_sizes)
        else:
            raise ValueError(f"Unsupported padding type: {pad_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.filter_size == 1:
            if self.pad_offset == 0:
                return x[:, :, ::self.stride, ::self.stride]
            return self.pad(x)[:, :, ::self.stride, ::self.stride]
        return F.conv2d(self.pad(x), self.filter, stride=self.stride, groups=x.shape[1])


class ResidualDownsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super().__init__()

        self.top = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.PReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.PReLU(),
            Downsample(channels=in_channels, filter_size=3, stride=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
        )
        self.bottom = nn.Sequential(
            Downsample(channels=in_channels, filter_size=3, stride=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        top = self.top(x)
        bottom = self.bottom(x)
        return top + bottom


class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.residual_block = ResidualBlock(in_channels, kernel_size, bias)
        self.reshape_conv = ResidualDownsample(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.residual_block(x)
        x = self.reshape_conv(x)
        return x


class PPM(nn.Module):
    def __init__(self, in_channels: int, reduction_channels: int, bins: List[int]):
        super().__init__()

        self.features = nn.ModuleList()
        for b in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(b),
                nn.Conv2d(in_channels, reduction_channels, kernel_size=1, bias=False),
                nn.PReLU(),
            ))

        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels + reduction_channels * len(bins), in_channels, kernel_size=3, padding=1, bias=False),
            nn.PReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2:]
        features = [x]
        for f in self.features:
            features.append(F.interpolate(f(x), size=(h, w), mode="bilinear", align_corners=True))
        features = torch.cat(features, dim=1)
        return self.fuse(features)


class CurveNLU(nn.Module):
    def __init__(self, channels: int, n_curve: int):
        super().__init__()

        self.n_curve = n_curve
        self.relu = nn.ReLU()
        self.a_predictor = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, n_curve, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.a_predictor(x)
        x = self.relu(x) - self.relu(x - 1)
        for i in range(self.n_curve):
            x = x + a[:, i: i + 1] * x * (1 - x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 32):
        super().__init__()

        self.encoder_block1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            EncoderBlock(base_channels, base_channels * 2),
        )
        self.encoder_block2 = EncoderBlock(base_channels * 2, base_channels * 4)
        self.encoder_block3 = EncoderBlock(base_channels * 4, base_channels * 4)

        bins = [1, 2, 3, 6]
        self.ppm1 = PPM(base_channels * 2, base_channels * 2 // 4, bins)
        self.ppm2 = PPM(base_channels * 4, base_channels * 4 // 4, bins)
        self.ppm3 = PPM(base_channels * 4, base_channels * 4 // 4, bins)

        self.n_curve = 3
        self.curve_nlu1 = CurveNLU(base_channels * 2, self.n_curve)
        self.curve_nlu2 = CurveNLU(base_channels * 4, self.n_curve)
        self.curve_nlu3 = CurveNLU(base_channels * 4, self.n_curve)

        self.loss_conv = nn.Conv2d(base_channels * 4, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor, side_loss: bool = False) -> torch.Tensor:
        x2 = self.curve_nlu1(self.ppm1(self.encoder_block1(x)))
        x4 = self.curve_nlu2(self.ppm2(self.encoder_block2(x2)))
        x8 = self.curve_nlu3(self.ppm3(self.encoder_block3(x4)))

        result = {
            "down2": x2,
            "down4": x4,
            "down8": x8,
        }
        if side_loss:
            result["side_loss"] = self.loss_conv(x8)
        return result

# endregion

# region Decoder

class ResidualUpsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super().__init__()

        self.top = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.PReLU(),
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=bias),
            nn.PReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
        )
        self.bottom = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=bias),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        top = self.top(x)
        bottom = self.bottom(x)
        return top + bottom


class ResidualBlock2(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int = 3, bias: bool = False):
        super().__init__()

        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, bias=bias),
            nn.PReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, bias=bias),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, bias=bias),
            nn.PReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, bias=bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.conv1(x)
        x = x + self.conv2(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.reshape_conv = ResidualUpsample(in_channels, out_channels)
        self.residual_block = ResidualBlock2(out_channels, kernel_size, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.reshape_conv(x)
        x = self.residual_block(x)
        return x


class FAC(nn.Module):
    def __init__(self, channels: int, kernel_size: int):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels * kernel_size ** 2, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class KernelConv2d(nn.Module):
    def __init__(self, kernel_size: int = 5, activation: bool = True):
        super().__init__()
        self.kernel_size = kernel_size
        self.activation = activation

    def forward(self, x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        c = x.shape[1]
        padding = (self.kernel_size - 1) // 2
        x = F.pad(x, (padding, padding, padding, padding), mode="replicate")
        x = x.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1)
        x = rearrange(x, "n c h w k1 k2 -> n h w c (k1 k2)")

        kernel = rearrange(kernel, "n (c k) h w -> n h w c k", c=c)
        x = torch.sum(x * kernel, dim=-1)
        x = rearrange(x, "n h w c -> n c h w")
        if self.activation:
            x = F.leaky_relu(x, 0.2, inplace=True)
        return x


class Decoder(nn.Module):
    def __init__(self, base_channels: int = 32, connection: bool = False):
        super().__init__()

        self.connection = connection
        kernel_size = 5
        self.fac3 = FAC(base_channels * 4, kernel_size)
        self.fac2 = FAC(base_channels * 4, kernel_size)
        self.fac1 = FAC(base_channels * 2, kernel_size)

        self.decoder_block3 = DecoderBlock(base_channels * 4, base_channels * 4)
        self.decoder_block2 = DecoderBlock(base_channels * 4, base_channels * 2)
        self.decoder_block1 = nn.Sequential(
            DecoderBlock(base_channels * 2, base_channels),
            nn.Conv2d(base_channels, 3, kernel_size=3, padding=1),
        )

        self.kernel_conv_deblur = KernelConv2d(kernel_size=kernel_size, activation=True)

    def forward(self, x: torch.Tensor, encoder_results: dict) -> torch.Tensor:
        x8 = encoder_results["down8"]
        x4 = encoder_results["down4"]
        x2 = encoder_results["down2"]

        kernel3 = self.fac3(x8)
        de_x8 = self.kernel_conv_deblur(x, kernel3)
        if self.connection:
            de_x8 = de_x8 + x8

        de_x4 = self.decoder_block3(de_x8)
        kernel2 = self.fac2(x4)
        de_x4 = self.kernel_conv_deblur(de_x4, kernel2)
        if self.connection:
            de_x4 = de_x4 + x4

        de_x2 = self.decoder_block2(de_x4)
        kernel1 = self.fac1(x2)
        de_x2 = self.kernel_conv_deblur(de_x2, kernel1)
        if self.connection:
            de_x2 = de_x2 + x2

        de_x = self.decoder_block1(de_x2)
        return de_x

# endregion

# region Loss Functions

class Loss(nn.Module):
    def __init__(self, lambda_perceptual: float):
        super().__init__()
        self.lambda_perceptual = lambda_perceptual
        self.perceptual_loss = PerceptualLoss(weights=[1.0, 1.0, 1.0], criterion="l1")
        self.l1_loss = nn.L1Loss()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        perceptual_loss = self.lambda_perceptual * self.perceptual_loss(x, y)
        l1_loss = self.l1_loss(x, y)
        return perceptual_loss + l1_loss

# endregion

# region LEDNet

class LEDNetModel(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 32, connection: bool = False):
        super().__init__()

        self.encoder = Encoder(in_channels, base_channels)
        self.middle_blocks = nn.Sequential(
            ResidualBlock(base_channels * 4),
            ResidualBlock(base_channels * 4),
            ResidualBlock2(base_channels * 4),
        )
        self.decoder = Decoder(base_channels, connection)

    def forward(self, x: torch.Tensor, side_loss: bool = False) -> torch.Tensor:
        encoder_results = self.encoder(x, side_loss)
        x = encoder_results["down8"]
        x = self.middle_blocks(x)
        de_x = self.decoder(x, encoder_results)
        if side_loss:
            return de_x, encoder_results["side_loss"]
        else:
            return de_x


class LEDNet(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config
        model_config = config["model"]
        self.model = LEDNetModel(
            in_channels=model_config["in_channels"],
            base_channels=model_config["base_channels"],
            connection=model_config["connection"],
        )

        self.save_path = config["data"].get("save_path", "")
        if self.save_path:
            os.makedirs(self.save_path, exist_ok=True)

        # loss functions
        self.lambda_perceptual = model_config["lambda_perceptual"]
        self.lambda_encoder = model_config["lambda_encoder"]
        self.lambda_deblur = model_config["lambda_deblur"]
        self.loss_fn = Loss(self.lambda_perceptual)

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg")

    def forward(self, x: torch.Tensor, side_loss: bool = True) -> torch.Tensor:
        return self.model(x, side_loss)

    def configure_optimizers(self):
        train_config = self.config["train"]
        optimizer = get_optimizer(train_config, self.model)
        scheduler = get_scheduler(train_config, optimizer)
        return [optimizer], [scheduler]

    def _compute_loss(self, x8: torch.Tensor, y8: torch.Tensor, x: torch.Tensor, y: torch.Tensor):
        encoder_loss = self.lambda_encoder * self.loss_fn(x8, y8)
        deblur_loss = self.lambda_deblur * self.loss_fn(x, y)
        return encoder_loss + deblur_loss

    def _compute_metrics(self, x: torch.Tensor, y: torch.Tensor):
        x = torch.clip(x, 0.0, 1.0)
        self.psnr_val = self.psnr(x, y)
        self.ssim_val = self.ssim(x, y)
        self.lpips_val = self.lpips(x, y)

    def on_train_epoch_start(self):
        extra_logger.info(f"Epoch {self.current_epoch} starts.")
        extra_logger.info(f"Start training stage.")

    def training_step(self, batch, batch_idx):
        low, high = batch["low"], batch["high"]
        high_pred, x8 = self.forward(low, side_loss=True)
        h, w = x8.shape[-2:]
        y8 = F.interpolate(high, size=(h, w), mode="bicubic", align_corners=False)
        loss = self._compute_loss(x8, y8, high_pred, high)

        self.log("train/loss", loss, on_step=True, on_epoch=True, batch_size=low.shape[0])

        return loss

    def on_validation_epoch_start(self):
        extra_logger.info(f"Start validation stage.")

    def validation_step(self, batch, batch_idx):
        low, high = batch["low"], batch["high"]
        high_pred, x8 = self.forward(low, side_loss=True)
        h, w = x8.shape[-2:]
        y8 = F.interpolate(high, size=(h, w), mode="bicubic", align_corners=False)
        loss = self._compute_loss(x8, y8, high_pred, high)
        self._compute_metrics(high_pred, high)

        self.log("val/loss", loss, on_step=True, on_epoch=True, batch_size=low.shape[0])
        self.log_dict({
            "val/psnr": self.psnr_val,
            "val/ssim": self.ssim_val,
            "val/lpips": self.lpips_val,
        }, on_step=False, on_epoch=True, batch_size=low.shape[0])

        if batch_idx == 0:
            low = low[0].detach().cpu()
            high = high[0].detach().cpu()
            high_pred = high_pred[0].detach().cpu()
            image = torch.cat([low, high_pred, high], dim=2)
            image = torch.clip(image * 255.0, 0.0, 255.0).to(torch.uint8)

            self.logger.experiment.add_image(f"val/image", image, self.current_epoch)

        return loss

    def on_validation_epoch_end(self):
        scheduler = self.lr_schedulers()
        self.log("lr", scheduler.get_last_lr()[0], on_step=False, on_epoch=True)

    def on_fit_start(self):
        extra_logger.info(f"Start training on {self.device}.")

    def on_fit_end(self):
        extra_logger.info(f"All training finished.")

    def test_step(self, batch, batch_idx):
        low, high = batch["low"], batch["high"]
        high_pred = self.forward(low, side_loss=False)
        self._compute_metrics(high_pred, high)
        self.log_dict({
            "test/psnr": self.psnr_val,
            "test/ssim": self.ssim_val,
            "test/lpips": self.lpips_val,
        }, on_step=False, on_epoch=True, batch_size=low.shape[0])

        if batch_idx == 0:
            low = low[0].detach().cpu()
            high = high[0].detach().cpu()
            high_pred = high_pred[0].detach().cpu()
            image = torch.cat([low, high_pred, high], dim=2)
            image = torch.clip(image * 255.0, 0.0, 255.0).to(torch.uint8)
            self.logger.experiment.add_image(f"test/image", image, self.current_epoch)

    def on_test_start(self):
        extra_logger.info("Start testing.")

    def on_test_end(self):
        extra_logger.info(f"Testing finished.")

    def predict_step(self, batch, batch_idx):
        low = batch["low"]
        high_pred = self.forward(low, side_loss=False)
        high_pred = torch.clip(high_pred * 255.0, 0.0, 255.0).to(torch.uint8)
        save_batch_tensor(high_pred, self.save_path, batch)

# endregion
from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler
from torch.optim.lr_scheduler import _LRScheduler
from torchvision.transforms import Grayscale
from einops import rearrange
import lightning as pl
import loguru
import random
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, LearnedPerceptualImagePatchSimilarity
from typing import Optional, Union, List, Tuple
from collections import Counter

from src.llie.utils.config import get_optimizer
from src.llie.models.utils import save_batch_tensor


# ========================================================================================================
#                                          Conditional Encoder
# ========================================================================================================

class ResidualDenseBlock(nn.Module):
    def __init__(self, in_channels: int = 64, base_channels: int = 32, num_layers: int = 5, bias: bool = True):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                nn.Conv2d(in_channels + i * base_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=bias)
            )
        self.out_conv = nn.Conv2d(in_channels + (num_layers - 1) * base_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        self._init_weights(0.1)

    def _init_weights(self, scale):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        for conv in self.convs:
            x = torch.cat((x, self.lrelu(conv(x))), dim=1)
        x = self.out_conv(x)
        return 0.2 * x + identity


class RRDB(nn.Module):
    """Residual in Residual Dense Block"""
    def __init__(self, in_channels: int, base_channels: int = 32, num_layers: int = 5):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(in_channels, base_channels, num_layers)
        self.rdb2 = ResidualDenseBlock(in_channels, base_channels, num_layers)
        self.rdb3 = ResidualDenseBlock(in_channels, base_channels, num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.rdb1(x)
        x = self.rdb2(x)
        x = self.rdb3(x)
        return 0.2 * x + identity


class ConditionalEncoder(nn.Module):
    def __init__(self, in_channels: int, num_blocks: int,
                 num_features: int, base_channels: int = 32, num_layers: int = 5):
        super().__init__()

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, num_features, kernel_size=3, stride=1, padding=1, bias=True),
            self.lrelu,
            nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1, bias=True)
        )
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.rrdb_blocks = nn.Sequential(*[
            RRDB(num_features, base_channels, num_layers) for _ in range(num_blocks)
        ])
        self.rrdb_conv = nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1, bias=True)

        self.down_conv1 = nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1, bias=True)
        self.down_conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1, bias=True)

        self.color_map_conv = nn.Sequential(
            nn.Conv2d(num_features, 3, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, get_steps: bool = False, blocks: Optional[List[int]] = None) -> torch.Tensor:
        x = self.in_conv(x)
        x_down2 = self.max_pool(x)

        x_head = x_down2
        block_results = {}
        for i, block in enumerate(self.rrdb_blocks.children()):
            x_down2 = block(x_down2)
            if i in blocks:
                block_results[f"block_{i}"] = x_down2
        x_down2 = self.rrdb_conv(x_down2) + x_head

        x_down4 = self.down_conv1(F.interpolate(x_down2, scale_factor=0.5, mode='bilinear', align_corners=False, recompute_scale_factor=True))
        x = self.lrelu(x_down4)
        x_down8 = self.down_conv2(F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False, recompute_scale_factor=True))

        color_map = self.color_map_conv(F.interpolate(x_down2, scale_factor=2))

        results = {
            'features_down0': x_head,
            'features_down2': x_down2,
            'features_down4': x_down4,
            'features_down8': x_down8,
            'color_map': color_map
        }
        if get_steps:
            return {**results, **block_results}
        else:
            return results

# ========================================================================================================
#                                             Flow Network
# ========================================================================================================

class SqueezeLayer(nn.Module):
    def __init__(self, factor: int = 2):
        super().__init__()
        assert factor >= 1
        self.factor = factor

    @staticmethod
    def squeeze2d(x: torch.Tensor, factor: int):
        if factor == 1:
            return x
        b, c, h, w = x.shape
        assert h % factor == 0 and w % factor == 0
        x = rearrange(x, "b c (ph fh) (pw fw) -> b (c fh fw) ph pw", fh=factor, fw=factor)
        return x

    @staticmethod
    def unsqueeze2d(x: torch.Tensor, factor: int):
        if factor == 1:
            return x
        b, c, h, w = x.shape
        assert c % factor == 0
        x = rearrange(x, "b (c fh fw) ph pw -> b c (ph fh) (pw fw)", fh=factor, fw=factor)
        return x

    def forward(self, x: torch.Tensor, reverse: bool = False) -> torch.Tensor:
        if not reverse:
            return self.squeeze2d(x, self.factor)
        else:
            return self.unsqueeze2d(x, self.factor)


class ActNorm(nn.Module):
    """Activation Normalization Layer"""
    def __init__(self, num_features: int, scale: float = 1.0):
        super().__init__()
        self.num_features = num_features
        self.scale = scale
        self.register_parameter('bias', nn.Parameter(torch.zeros((1, num_features, 1, 1))))
        self.register_parameter('logs', nn.Parameter(torch.zeros((1, num_features, 1, 1))))
        self.initialized = False

    def _initialize(self, x: torch.Tensor):
        if not self.training:
            return
        if (self.bias != 0).any():
            self.initialized = True
            return
        with torch.no_grad():
            bias = -torch.mean(x.clone(), dim=[0, 2, 3], keepdim=True)
            vars = torch.mean((x.clone() + bias) ** 2, dim=[0, 2, 3], keepdim=True)
            logs = torch.log(self.scale / (torch.sqrt(vars) + 1e-6))
            self.bias.data.copy_(bias.data)
            self.logs.data.copy_(logs.data)
            self.initialized = True

    def _center(self, x: torch.Tensor, reverse: bool = False, offset: Optional[torch.Tensor] = None):
        bias = self.bias
        if offset is not None:
            bias = bias + offset
        if not reverse:
            return x + bias
        else:
            return x - bias

    def _scale(self, x: torch.Tensor, logdet: Optional[torch.Tensor] = None,
               reverse: bool = False, offset: Optional[torch.Tensor] = None):
        logs = self.logs
        if offset is not None:
            logs = logs + offset
        if not reverse:
            x = x * torch.exp(logs)
        else:
            x = x * torch.exp(-logs)
        if logdet is not None:
            h, w = x.shape[-2:]
            d_logdet = torch.sum(logs) * h * w
            if not reverse:
                logdet = logdet + d_logdet
            else:
                logdet = logdet - d_logdet
        return x, logdet

    def forward(self, x: torch.Tensor, logdet: Optional[torch.Tensor] = None, reverse: bool = False,
                offset_mask: Optional[torch.Tensor] = None, bias_offset: Optional[torch.Tensor] = None,
                logs_offset: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not self.initialized:
            self._initialize(x)

        if offset_mask is not None:
            bias_offset *= offset_mask
            logs_offset *= offset_mask

        if not reverse:
            x = self._center(x, reverse, bias_offset)
            x, logdet = self._scale(x, logdet, reverse, logs_offset)
        else:
            x, logdet = self._scale(x, logdet, reverse, logs_offset)
            x = self._center(x, reverse, bias_offset)
        return x, logdet


class InvConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.w_shape = (out_channels, in_channels)
        w_init, _ = torch.linalg.qr(torch.randn(self.w_shape))
        self.register_parameter("weight", nn.Parameter(w_init))

    def get_weight(self, x: torch.Tensor, reverse: bool = False):
        pixels = x.shape[2] * x.shape[3]
        d_logdet = torch.tensor(float('inf'))

        while torch.isinf(d_logdet):
            try:
                d_logdet = pixels * torch.slogdet(self.weight)[1]   #  sign, logdet
            except:
                d_logdet = pixels * torch.slogdet(
                    self.weight + self.weight.mean() * torch.randn(self.w_shape).to(x) * 0.001
                )[1]

        if not reverse:
            weight = self.weight
        else:
            try:
                weight = torch.inverse(self.weight.double()).float()
            except:
                weight = torch.inverse(
                    self.weight.double() + self.weight.mean() * torch.randn(self.w_shape).to(x) * 0.001
                ).float()

        weight = weight.unsqueeze(-1).unsqueeze(-1)
        return weight, d_logdet

    def forward(self, x: torch.Tensor, logdet: Optional[torch.Tensor] = None, reverse: bool = False) -> torch.Tensor:
        weight, d_logdet = self.get_weight(x, reverse)
        z = F.conv2d(x, weight)
        if logdet is not None:
            if not reverse:
                logdet = logdet + d_logdet
            else:
                logdet = logdet - d_logdet
        return z, logdet


class CAR(nn.Module):
    """Conv-ActNorm-ReLU"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        padding = ((kernel_size - 1) * stride + 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.act_norm = ActNorm(out_channels)
        self.relu = nn.ReLU()
        self._init_weights()

    def _init_weights(self):
        self.conv.weight.data.normal_(mean=0.0, std=0.05)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x, _ = self.act_norm(x)
        x = self.relu(x)
        return x


class Conv2dZero(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, log_scale: float = 3.0):
        super().__init__()
        padding = ((kernel_size - 1) * stride + 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.log_scale = log_scale
        self.register_parameter("logs", nn.Parameter(torch.zeros(out_channels, 1, 1)))
        self._init_weights()

    def _init_weights(self):
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x * torch.exp(self.logs * self.log_scale)
        return x


class CondAffineSeperatedAndCond(nn.Module):
    def __init__(self, in_channels: int, in_channels_rrdb: int = 320, kernel_hidden: int = 1,
                 affine_eps: float = 0.0001, num_hidden_layers: int = 1, hidden_channels: int = 64):
        super().__init__()

        self.in_channels = in_channels
        self.in_channels_rrdb = in_channels_rrdb
        self.kernel_hidden = kernel_hidden
        self.affine_eps = affine_eps
        self.num_hidden_layers = num_hidden_layers
        self.hidden_channels = hidden_channels
        self.channels_nn = self.in_channels // 2
        self.channels_co = self.in_channels - self.channels_nn

        self.affine_nn = self._make_layer(
            in_channels=self.channels_nn + self.in_channels_rrdb,
            out_channels=self.channels_co * 2,
        )
        self.features_nn = self._make_layer(
            in_channels=self.in_channels_rrdb,
            out_channels=self.in_channels * 2,
        )

    def _make_layer(self, in_channels: int, out_channels: int):
        return nn.Sequential(
            CAR(in_channels, self.hidden_channels),
            *[CAR(self.hidden_channels, self.hidden_channels, kernel_size=self.kernel_hidden)
              for _ in range(self.num_hidden_layers)],
            Conv2dZero(self.hidden_channels, out_channels)
        )

    def forward(self, x: torch.Tensor, ft: torch.Tensor, logdet: torch.Tensor, reverse: bool = False):
        z = x
        if not reverse:
            # feature conditional
            shift_ft, scale_ft = self.feature_extraction(self.features_nn(ft))
            z = z + shift_ft
            z = z * scale_ft
            logdet = logdet + torch.sum(torch.log(scale_ft), dim=[1, 2, 3])

            # self conditional
            z1, z2 = z[:, :self.channels_nn, ...], z[:, self.channels_nn:, ...]
            shift, scale = self.feature_extraction(self.affine_nn(torch.cat([z1, ft], dim=1)))
            z2 = z2 + shift
            z2 = z2 * scale
            logdet = logdet + torch.sum(torch.log(scale), dim=[1, 2, 3])
            z = torch.cat([z1, z2], dim=1)
        else:
            # self conditional
            z1, z2 = z[:, :self.channels_nn, ...], z[:, self.channels_nn:, ...]
            shift, scale = self.feature_extraction(self.affine_nn(torch.cat([z1, ft], dim=1)))
            z2 = z2 / scale
            z2 = z2 - shift
            logdet = logdet - torch.sum(torch.log(scale), dim=[1, 2, 3])
            z = torch.cat([z1, z2], dim=1)

            # feature conditional
            shift_ft, scale_ft = self.feature_extraction(self.features_nn(ft))
            z = z / scale_ft
            z = z - shift_ft
            logdet = logdet - torch.sum(torch.log(scale_ft), dim=[1, 2, 3])

        output = z
        return output, logdet

    def feature_extraction(self, h: torch.Tensor):
        shift, scale = h[:, 0::2, ...], h[:, 1::2, ...]
        scale = torch.sigmoid(scale + 2.) + self.affine_eps
        return shift, scale


class FlowStep(nn.Module):
    def __init__(self, in_channels: int, actnorm_scale: float = 1.0, flow_coupling: bool = True, position: Optional[str] = None):
        super().__init__()

        self.position = position
        self.flow_coupling = flow_coupling
        self.act_norm = ActNorm(in_channels, actnorm_scale)
        self.flow_permutation = InvConv2d(in_channels, in_channels)
        self.affine = CondAffineSeperatedAndCond(in_channels) if flow_coupling else None

    def forward(self, z: torch.Tensor, logdet: Optional[torch.Tensor] = None, reverse: bool = False,
                rrdb_results: Optional[Union[dict, torch.Tensor]] = None):
        if not reverse:
            z, logdet = self.act_norm(z, logdet, reverse)
            z, logdet = self.flow_permutation(z, logdet, reverse)
            if self.flow_coupling:
                image_ft = rrdb_results[self.position] if isinstance(rrdb_results, dict) else rrdb_results
                z, logdet = self.affine(z, image_ft, logdet, reverse)
        else:
            if self.flow_coupling:
                image_ft = rrdb_results[self.position] if isinstance(rrdb_results, dict) else rrdb_results
                z, logdet = self.affine(z, image_ft, logdet, reverse)
            z, logdet = self.flow_permutation(z, logdet, reverse)
            z, logdet = self.act_norm(z, logdet, reverse)
        return z, logdet


class FlowBlock(nn.Module):
    def __init__(self, in_channels: int, num_steps: int, additional_steps_without_coupling: int,
                 level: int, actnorm_scale: float = 1.0):
        super().__init__()
        level_to_name = {
            1: "features_down2",
            2: "features_down4",
            3: "features_down8",
        }
        self.level_name = level_to_name[level]
        self.layers = nn.ModuleList([
            SqueezeLayer(2),
            *[
                FlowStep(in_channels, actnorm_scale, flow_coupling=False)
                for _ in range(additional_steps_without_coupling)
            ],
            *[
                FlowStep(in_channels, actnorm_scale, flow_coupling=True, position=self.level_name)
                for _ in range(num_steps)
            ]
        ])

    def forward(self, z: Optional[torch.Tensor] = None, logdet: Optional[torch.Tensor] = None, reverse: bool = False,
                rrdb_results: Optional[dict] = None, epses: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None):
        if not reverse:
            z, logdet = self.encode(z, rrdb_results, logdet, epses)
        else:
            z, logdet = self.decode(z, rrdb_results, logdet, epses)
        return z, logdet

    def encode(self, z: torch.Tensor, rrdb_results: Optional[dict], logdet: torch.Tensor,
               epses: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None):
        reverse = False

        for layer in self.layers:
            if isinstance(layer, FlowStep):
                z, logdet = layer(z, logdet, reverse, rrdb_results[self.level_name])
            else:
                z = layer(z, reverse)

        if not isinstance(epses, list):
            return z, logdet
        epses.append(z)
        return epses, logdet

    def decode(self, z: torch.Tensor, rrdb_results: Optional[dict], logdet: torch.Tensor,
               epses: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None):
        reverse = True
        z = epses.pop() if isinstance(epses, list) else z

        for layer in reversed(self.layers):
            if isinstance(layer, FlowStep):
                z, logdet = layer(z, logdet, reverse, rrdb_results[self.level_name])
            else:
                z = layer(z, reverse)

        return z, logdet


class FlowUpsamplerNet(nn.Module):
    def __init__(self, in_channels: int, num_steps: int, additional_steps_without_coupling: int,
                 num_blocks: int, actnorm_scale: float = 1.0):
        super().__init__()

        self.in_channels = in_channels
        self.num_steps = num_steps
        self.num_blocks = num_blocks

        self.layers = nn.ModuleList()
        # Upsampler
        for level in range(1, num_blocks + 1):
            self.in_channels = self.in_channels * 4 # squeeze
            self.layers.append(
                FlowBlock(self.in_channels, self.num_steps, additional_steps_without_coupling, level, actnorm_scale)
            )

    def forward(self, z: Optional[torch.Tensor] = None, logdet: Optional[torch.Tensor] = None, reverse: bool = False,
                gt: Optional[torch.Tensor] = None, rrdb_results: Optional[dict] = None,
                epses: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if not reverse:
            z = gt
            for layer in self.layers:
                z, logdet = layer(z, logdet, reverse, rrdb_results, epses)
        else:
            epses_copy = [eps for eps in epses] if isinstance(epses, list) else epses
            for layer in reversed(self.layers):
                z, logdet = layer(z, logdet, reverse, rrdb_results, epses_copy)
        return z, logdet

# ========================================================================================================
#                                               Main Model
# ========================================================================================================

class GaussianDiag(object):
    @staticmethod
    def _likelihood(mean, logs, x):
        log2pi = np.log(2 * np.pi)
        if mean is None and logs is None:
            return -0.5 * (x ** 2 + log2pi)
        else:
            # logs = log(|vars|)
            return -0.5 * (logs * 2. + ((x - mean) ** 2) / torch.exp(logs * 2.) + log2pi)

    @staticmethod
    def logp(mean, logs, x):
        if isinstance(x, (list, tuple)):
            likelihood = 0
            for i in x:
                likelihood += torch.sum(GaussianDiag._likelihood(mean, logs, i), dim=[1, 2, 3])
        else:
            likelihood = torch.sum(GaussianDiag._likelihood(mean, logs, x), dim=[1, 2, 3])
        return likelihood


class MultiStepLR_Restart(_LRScheduler):
    def __init__(self, optimizer, milestones: List[int], gamma: float = 0.1, last_epoch: int = -1, warmup_epochs: int = 0,
                 restart: Optional[List[int]] = None, restart_weights: Optional[List[float]] = None):
        self.milestones = Counter(milestones)
        self.gamma = gamma
        self.warmup_epochs = warmup_epochs
        self.restart = restart if restart is not None else [0]
        self.restart_weights = restart_weights if restart_weights is not None else [1]
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if self.last_epoch in self.restart:
            weight = self.restart_weights[self.restart.index(self.last_epoch)]
            lr = [group['initial_lr'] * weight for group in self.optimizer.param_groups]
        elif self.last_epoch not in self.milestones:
            lr = [group['lr'] for group in self.optimizer.param_groups]
        else:
            lr =[
                group['lr'] * self.gamma ** self.milestones[self.last_epoch]
                for group in self.optimizer.param_groups
            ]
        if self.last_epoch < self.warmup_epochs:
            init_lr = [group['initial_lr'] for group in self.optimizer.param_groups]
            lr = [v / self.warmup_epochs * self.last_epoch for v in init_lr]
        return lr


class LLFlow(pl.LightningModule):
    def __init__(self, config, logger: "loguru.Logger"):
        super().__init__()

        self.config = config
        self.quant = config["data"]["quant"]
        self.train_gt_ratio = config["train"]["gt_ratio"]
        self.rrdb_start_epoch = config["train"]["rrdb_start_epoch"]
        model_config = config["model"]
        conditional_encoder_config = model_config["ConditionalEncoder"]
        self.rrdb = ConditionalEncoder(
            in_channels=conditional_encoder_config["in_channels"],
            num_blocks=conditional_encoder_config["num_blocks"],
            num_features=conditional_encoder_config["num_features"],
            base_channels=conditional_encoder_config["base_channels"],
            num_layers=conditional_encoder_config["num_layers"],
        )
        self.blocks = conditional_encoder_config["blocks"]
        flow_upsampler_config = model_config["FlowUpsampler"]
        self.flow_upsampler = FlowUpsamplerNet(
            in_channels=flow_upsampler_config["in_channels"],
            num_steps=flow_upsampler_config["num_steps"],
            additional_steps_without_coupling=flow_upsampler_config["additional_steps"],
            num_blocks=flow_upsampler_config["num_blocks"],
            actnorm_scale=flow_upsampler_config["actnorm_scale"],
        )
        self.scaler = GradScaler()
        self.extra_logger = logger
        self.automatic_optimization = False

        self.save_path = config["data"].get("save_path", "")
        if self.save_path:
            os.makedirs(self.save_path, exist_ok=True)

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze')

    def rrdb_preprocess(self, low: torch.Tensor):
        rrdb_results = self.rrdb(low, get_steps=True, blocks=self.blocks)
        if len(self.blocks) > 0:
            concat = torch.concat([
                rrdb_results[f"block_{i}"] for i in self.blocks
            ], dim=1)
            for key in rrdb_results.keys():
                if key.startswith("features"):
                    h, w = rrdb_results[key].shape[-2:]
                    rrdb_results[key] = torch.cat([rrdb_results[key], F.interpolate(concat, (h, w))], dim=1)
        return rrdb_results

    def normal_flow(self, gt: torch.Tensor, lr: torch.Tensor, lr_enc: Optional[dict] = None,
                    epses: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None, add_gt_noise: bool = True):
        logdet = torch.zeros_like(gt[:, 0, 0, 0])
        pixels = gt.shape[2] * gt.shape[3]

        z = gt
        if add_gt_noise:
            z = z + (torch.randn_like(z) - 0.5) / self.quant
            logdet = logdet - torch.log(self.quant) * pixels

        if lr_enc is None:
            lr_enc = self.rrdb_preprocess(lr)

        epses, logdet = self.flow_upsampler(rrdb_results=lr_enc, gt=z, logdet=logdet, epses=epses, reverse=False)

        z = epses
        color_map = SqueezeLayer.squeeze2d(lr_enc["color_map"], 8)
        gt_color = SqueezeLayer.squeeze2d(gt / (gt.sum(dim=1, keepdim=True) + 1e-4), 8)
        mean = color_map if random.random() > self.train_gt_ratio else gt_color
        objective = logdet.clone()
        objective = objective + GaussianDiag.logp(mean, torch.tensor(0.).to(z), z)

        nll = -objective / (np.log(2.) * pixels)

        if isinstance(epses, list):
            return epses, nll, logdet
        return z, nll, logdet

    def reverse_flow(self, lr: torch.Tensor, lr_enc: Optional[dict] = None,
                     epses: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None, add_gt_noise: bool = True):
        logdet = torch.zeros_like(lr[:, 0, 0, 0])
        pixels = lr.shape[2] * lr.shape[3]

        if add_gt_noise:
            logdet = logdet + torch.log(self.quant) * pixels

        if lr_enc is None:
            lr_enc = self.rrdb_preprocess(lr)

        color_map = SqueezeLayer.squeeze2d(lr_enc["color_map"], 8)
        x, logdet = self.flow_upsampler(rrdb_results=lr_enc, z=color_map, epses=epses, logdet=logdet, reverse=True)
        x = torch.masked_fill(x, torch.isnan(x), 1.0)
        return x, logdet

    def forward(self, lr: Optional[torch.Tensor], gt: Optional[torch.Tensor] = None, reverse: bool = False,
                lr_enc: Optional[dict] = None, epses: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
                add_gt_noise: bool = False):
        if not reverse:
            return self.normal_flow(gt, lr, lr_enc, epses, add_gt_noise)
        else:
            return self.reverse_flow(lr, lr_enc, epses, add_gt_noise)

    @staticmethod
    def finetune_brightness(images: torch.Tensor, gt: torch.Tensor):
        rgb_to_gray = Grayscale()
        mean_gray_image = rgb_to_gray(images).mean(dim=[1, 2, 3], keepdim=True)
        mean_gray_gt = rgb_to_gray(gt).mean(dim=[1, 2, 3], keepdim=True)
        return torch.clip(images * (mean_gray_gt / mean_gray_image), 0, 1)

    def configure_optimizers(self):
        train_config = self.config["train"]
        scheduler_config = train_config["scheduler"]
        optimizer = get_optimizer(train_config, self, self.extra_logger)
        scheduler = MultiStepLR_Restart(optimizer, milestones=scheduler_config["milestones"],
                                        gamma=scheduler_config["gamma"], warmup_epochs=scheduler_config["warmup_steps"])
        return [optimizer], [scheduler]

    def _compute_metrics(self, fake_hr: torch.Tensor, gt: torch.Tensor):
        fake_hr = torch.clip(fake_hr, 0, 1)
        gt = torch.clip(gt, 0, 1)
        self.psnr_score = self.psnr(gt, fake_hr)
        self.ssim_score = self.ssim(gt, fake_hr)
        self.lpips_score = self.lpips(gt * 2 - 1, fake_hr * 2 - 1)

    def on_train_epoch_start(self):
        self.extra_logger.info(f"Epoch {self.current_epoch} starts.")
        self.extra_logger.info(f"Start training stage.")

    def training_step(self, batch, batch_idx):
        lr, gt = batch["low"], batch["high"]
        z, nll, _ = self.forward(lr, gt=gt, reverse=False)
        nll_loss = torch.mean(nll)

        optimizer = self.optimizers()
        lr_scheduler = self.lr_schedulers()
        lr_scheduler.step()
        self.manual_backward(self.scaler.scale(nll_loss))
        self.scaler.step(optimizer)
        self.scaler.update()

        self.log("train/loss", nll_loss, on_step=True, on_epoch=True, batch_size=lr.shape[0])
        self.log("lr", lr_scheduler.get_last_lr()[0], on_step=True, on_epoch=False, prog_bar=True)

        return nll_loss

    def on_validation_epoch_start(self):
        self.extra_logger.info(f"Start validation stage.")

    def validation_step(self, batch, batch_idx):
        lr, gt = batch["low"], batch["high"]
        fake_hr, _ = self.forward(lr, reverse=True)
        # fake_hr = self.finetune_brightness(fake_hr, gt)
        self._compute_metrics(fake_hr, gt)

        self.log_dict({
            "val/psnr": self.psnr_score,
            "val/ssim": self.ssim_score,
            "val/lpips": self.lpips_score
        }, on_step=False, on_epoch=True, batch_size=lr.shape[0])

        # log images
        if batch_idx == 0:
            lr = lr[0].detach().cpu()
            gt = gt[0].detach().cpu()
            fake_hr = fake_hr[0].detach().cpu()
            image = torch.cat([lr[:3].exp(), fake_hr, gt], dim=2)
            image = torch.clip(image * 255, 0, 255).to(torch.uint8)
            self.logger.experiment.add_image("val/images", image, self.current_epoch)

    def on_fit_start(self):
        self.extra_logger.info(f"Start training on {self.device}.")

    def on_fit_end(self):
        self.extra_logger.info(f"Training finished.")

    def test_step(self, batch, batch_idx):
        lr, gt = batch["low"], batch["high"]
        fake_hr, _ = self.forward(lr, reverse=True)
        fake_hr = self.finetune_brightness(fake_hr, gt)
        self._compute_metrics(fake_hr, gt)

        self.log_dict({
            "test/psnr": self.psnr_score,
            "test/ssim": self.ssim_score,
            "test/lpips": self.lpips_score
        }, on_step=False, on_epoch=True, batch_size=lr.shape[0])

        # log images
        if batch_idx == 0:
            lr = lr[0].detach().cpu()
            gt = gt[0].detach().cpu()
            fake_hr = fake_hr[0].detach().cpu()
            image = torch.cat([lr[:3].exp(), fake_hr, gt], dim=2)
            image = torch.clip(image * 255, 0, 255).to(torch.uint8)
            self.logger.experiment.add_image("test/images", image, self.current_epoch)

    def on_test_start(self):
        self.extra_logger.info("Start testing.")

    def on_test_end(self):
        self.extra_logger.info(f"Testing finished.")

    def predict_step(self, batch, batch_idx):
        lr, gt = batch["low"], batch["high"]
        fake_hr, _ = self.forward(lr, reverse=True)
        fake_hr = self.finetune_brightness(fake_hr, gt)
        self._compute_metrics(fake_hr, gt)
        self.extra_logger.info(f"PSNR: {self.psnr_score:.2f}, SSIM: {self.ssim_score:.4f}, LPIPS: {self.lpips_score:.4f}")

        fake_hr = fake_hr.detach().cpu()
        fake_hr = torch.clip(fake_hr * 255, 0, 255).to(torch.uint8)
        save_batch_tensor(fake_hr, self.save_path, batch)
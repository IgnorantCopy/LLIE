import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import lightning as pl
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, LearnedPerceptualImagePatchSimilarity
from typing import Optional, List

from src.llie.metrics.ssim import SSIM
from src.llie.utils.logger import default_logger as extra_logger
from src.llie.utils.config import get_optimizer, get_scheduler
from src.llie.models.utils import save_batch_tensor


# region Wavelet Transform

class DWT(nn.Module):
    def __init__(self):
        super().__init__()
        self.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2,]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]
        x_LL = x1 + x2 + x3 + x4
        x_HL = -x1 - x2 + x3 + x4
        x_LH = -x1 + x2 - x3 + x4
        x_HH = x1 - x2 - x3 + x4
        return torch.cat([x_LL, x_HL, x_LH, x_HH], dim=0)


class IDWT(nn.Module):
    def __init__(self):
        super().__init__()
        self.requires_grad_(False)

    def forword(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        out_b, out_c, out_h, out_w = b // 4, c, h * 2, w * 2
        x1 = x[:out_b, :, :, :] / 2
        x2 = x[out_b:out_b * 2, :, :, :] / 2
        x3 = x[out_b * 2:out_b * 3, :, :, :] / 2
        x4 = x[out_b * 3:out_b * 4, :, :, :] / 2

        output = torch.zeros((out_b, out_c, out_h, out_w)).to(x)
        output[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
        output[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
        output[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
        output[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

        return output

# endregion

# region UNet

class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.dense = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.in_channels // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb).to(t.device)
        emb = t.float()[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        if self.in_channels % 2 == 1:
            emb = F.pad(emb, (0, 1, 0, 0))
        return self.dense(emb)


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: Optional[int] = None, conv_shortcut: bool = False,
                 dropout: float = 0.0, time_emb_channels: int = 512):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=self.in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.time_emd_proj = nn.Linear(time_emb_channels, out_channels)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=self.out_channels, eps=1e-6, affine=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.silu = nn.SiLU()

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = x
        h = self.conv1(self.silu(self.norm1(h)))
        h = h + self.time_emd_proj(self.silu(t_emb))[:, :, None, None]
        h = self.conv2(self.dropout(self.silu(self.norm2(h))))

        if self.in_channels != self.out_channels:
            x = self.shortcut(x)

        return x + h


class AttentionBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()

        self.in_channels = in_channels

        self.norm = nn.GroupNorm(num_groups=32, num_channels=self.in_channels, eps=1e-6, affine=True)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        h = self.norm(h)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)

        q = rearrange(q, "b c h w -> b (h w) c")
        k = rearrange(k, "b c h w -> b c (h w)")
        v = rearrange(v, "b c h w -> b c (h w)")
        scale = q.shape[-1] ** -0.5
        attn = q @ k * scale
        attn = attn.softmax(dim=-1)

        h = v @ attn.T
        h = rearrange(h, "b c (h w) -> b c h w", h=x.shape[2])
        h = self.proj_out(h)

        return h + x


class Downsample(nn.Module):
    def __init__(self, in_channels: int, use_conv: bool = True):
        super().__init__()

        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)
        else:
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_conv:
            x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0)
            x = self.conv(x)
        else:
            x = self.pool(x)

        return x


class Bottleneck(nn.Module):
    def __init__(self, in_channels: int, time_emb_channels: int = 512, dropout: float = 0.0):
        super().__init__()

        self.block1 = ResnetBlock(in_channels, in_channels, time_emb_channels=time_emb_channels, dropout=dropout)
        self.attn = AttentionBlock(in_channels)
        self.block2 = ResnetBlock(in_channels, in_channels, time_emb_channels=time_emb_channels, dropout=dropout)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.block1(x, t_emb)
        h = self.attn(h)
        h = self.block2(h, t_emb)
        return h


class Upsample(nn.Module):
    def __init__(self, in_channels: int, use_conv: bool = True):
        super().__init__()

        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int, channels_ratio: List[int],
                 num_blocks: int, resample_with_conv: bool = True, dropout: float = 0.0):
        super().__init__()

        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.time_emb_channels = self.mid_channels * 4
        self.levels = len(channels_ratio)
        self.num_blocks = num_blocks

        # timestep embedding
        self.time_embedding = TimestepEmbedding(self.mid_channels, self.time_emb_channels)

        # downsampling
        self.in_conv = torch.nn.Conv2d(in_channels, self.mid_channels, kernel_size=3, stride=1, padding=1)

        in_channels_ratio = [1] + channels_ratio
        self.down = nn.ModuleList()
        block_in = None
        for level in range(self.levels):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = mid_channels * in_channels_ratio[level]
            block_out = mid_channels * channels_ratio[level]
            for _ in range(self.num_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, time_emb_channels=self.time_emb_channels, dropout=dropout))
                block_in = block_out
                if level == 2:
                    attn.append(AttentionBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if level != self.levels - 1:
                down.downsample = Downsample(block_in, resample_with_conv)
            self.down.append(down)

        # middle
        self.mid = Bottleneck(block_in, time_emb_channels=self.time_emb_channels, dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for level in reversed(range(self.levels)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = mid_channels * channels_ratio[level]
            skip_in = mid_channels * channels_ratio[level]
            for i in range(self.num_blocks + 1):
                if i == self.num_blocks:
                    skip_in = mid_channels * in_channels_ratio[level]
                block.append(ResnetBlock(in_channels=block_in + skip_in, out_channels=block_out, time_emb_channels=self.time_emb_channels, dropout=dropout))
                block_in = block_out
                if level == 2:
                    attn.append(AttentionBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if level != 0:
                up.upsample = Upsample(block_in, resample_with_conv)
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.out = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True),
            nn.SiLU(),
            nn.Conv2d(block_in, out_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x, t):
        # timestep embedding
        t_emb = self.time_embedding(t)

        # downsampling
        hs = [self.in_conv(x)]
        for level in range(self.levels):
            for i in range(self.num_blocks):
                h = self.down[level].block[i](hs[-1], t_emb)
                if len(self.down[level].attn) > 0:
                    h = self.down[level].attn[i](h)
                hs.append(h)
            if level != self.levels - 1:
                hs.append(self.down[level].downsample(hs[-1]))

        # middle
        h = self.mid(hs[-1], t_emb)

        # upsampling
        for level in reversed(range(self.levels)):
            for i in range(self.num_blocks + 1):
                h = self.up[level].block[i](torch.cat([h, hs.pop()], dim=1), t_emb)
                if len(self.up[level].attn) > 0:
                    h = self.up[level].attn[i](h)
            if level != 0:
                h = self.up[level].upsample(h)

        # end
        h = self.out(h)

        return h

# endregion

# region High Frequency Restoration Module

class DepthConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.depth_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels)
        self.point_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.q = DepthConv(dim, dim)
        self.k = DepthConv(dim, dim)
        self.v = DepthConv(dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        q = self.q(hidden_states).permute(0, 2, 1, 3)
        k = self.k(context).permute(0, 2, 1, 3)
        v = self.v(context).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) / np.sqrt(int(self.dim / self.num_heads))
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        context = (attn @ v).permute(0, 2, 1, 3).contiguous()
        return context


class DilatedResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, in_channels, kernel_size=3, stride=1, padding=3, dilation=3),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, in_channels, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, in_channels, kernel_size=3, stride=1, padding=1, dilation=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x) + x
        return x


class HFRM(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv_head = DepthConv(in_channels, out_channels)

        self.dilated_LH = DilatedResBlock(out_channels, out_channels)
        self.dilated_HL = DilatedResBlock(out_channels, out_channels)
        self.dilated_HH = DilatedResBlock(out_channels, out_channels)
        self.cross_attn0 = CrossAttention(out_channels, num_heads=8)
        self.cross_attn1 = CrossAttention(out_channels, num_heads=8)
        self.conv_HH = nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, stride=1, padding=1)

        self.conv_tail = DepthConv(out_channels, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0] // 3
        residual = x

        x = self.conv_head(x)

        x_HL = x[:b, ...]
        x_LH = x[b:2 * b, ...]
        x_HH = x[2 * b:, ...]

        x_HH_LH = self.cross_attn0(x_LH, x_HH)
        x_HH_HL = self.cross_attn1(x_HL, x_HH)

        x_HL = self.dilated_HL(x_HL)
        x_LH = self.dilated_LH(x_LH)
        x_HH = self.dilated_HH(self.conv_HH(torch.cat([x_HH_LH, x_HH_HL], dim=1)))

        x = torch.cat([x_HL, x_LH, x_HH], dim=0)
        x = self.conv_tail(x)

        return x + residual

# endregion

# region Low Light Diffusion

class Diffusion(nn.Module):
    def __init__(self, diffusion_timesteps: int, sampling_timesteps: int,
                 beta_schedule: str, beta_start: float, beta_end: float,
                 in_channels: int, mid_channels: int, out_channels: int, channels_ratio: List[int],
                 num_blocks: int, resample_with_conv: bool = True, dropout: float = 0.0):
        super().__init__()

        self.diffusion_timesteps = diffusion_timesteps
        self.sampling_timesteps = sampling_timesteps
        self.beta_schedule = beta_schedule
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.enhance0 = HFRM(3, 64)
        self.enhance1 = HFRM(3, 64)
        self.unet = UNet(in_channels, mid_channels, out_channels, channels_ratio, num_blocks, resample_with_conv, dropout)

        self.betas = self.get_beta_schedule()
        self.num_timesteps = self.betas.shape[0]

        self.dwt = DWT()
        self.idwt = IDWT()

    def get_beta_schedule(self):
        if self.beta_schedule == "quad":
            betas = torch.linspace(self.beta_start ** 0.5, self.beta_end ** 0.5, self.diffusion_timesteps, dtype=torch.float32) ** 2
        elif self.beta_schedule == "linear":
            betas = torch.linspace(self.beta_start, self.beta_end, self.diffusion_timesteps, dtype=torch.float32)
        elif self.beta_schedule == "const":
            betas = self.beta_end * torch.ones(self.diffusion_timesteps, dtype=torch.float32)
        elif self.beta_schedule == "jsd":
            betas = 1.0 / torch.linspace(self.diffusion_timesteps, 1, self.diffusion_timesteps, dtype=torch.float32)
        elif self.beta_schedule == "sigmoid":
            betas = torch.linspace(-6, 6, self.diffusion_timesteps, dtype=torch.float32)
            betas = betas.sigmoid() * (self.beta_end - self.beta_start) + self.beta_start
        else:
            raise ValueError(f"Invalid beta schedule: {self.beta_schedule}")
        return betas

    @staticmethod
    def compute_alpha(beta, t):
        beta = torch.cat([torch.zeros(1).to(beta), beta], dim=0)
        a = (1 - beta).cumprod(0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return a

    def sample_training(self, x_cond: torch.Tensor, beta: torch.Tensor, eta=0.):
        step = self.diffusion_timesteps // self.sampling_timesteps
        seq = range(0, self.diffusion_timesteps, step)
        seq_next = [-1] + list(seq[:-1])
        b = x_cond.shape[0]
        x = torch.randn_like(x_cond)
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(b) * i).to(x_cond.device)
            next_t = (torch.ones(b) * j).to(x_cond.device)
            alpha_t = self.compute_alpha(beta, t.long())
            alpha_t_next = self.compute_alpha(beta, next_t.long())
            x_t = xs[-1].to(x.device)
            eps_t = self.unet(torch.cat([x_cond, x_t], dim=1), t)
            x0_t = (x_t - eps_t * (1 - alpha_t).sqrt()) / alpha_t.sqrt()

            c1 = eta * ((1 - alpha_t / alpha_t_next) * (1 - alpha_t_next) / (1 - alpha_t)).sqrt()
            c2 = ((1 - alpha_t_next) - c1 ** 2).sqrt()

            x_t_next = alpha_t_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * eps_t
            xs.append(x_t_next.to(x.device))

        return xs[-1]

    @staticmethod
    def transform(x: torch.Tensor):
        return 2 * x - 1.0

    @staticmethod
    def inverse_transform(x: torch.Tensor):
        return torch.clip((x + 1.0) / 2.0, 0.0, 1.0)

    def forward(self, low: torch.Tensor, high: Optional[torch.Tensor] = None):
        b = low.shape[0]

        low = self.transform(low)
        low_dwt = self.dwt(low)
        low_LL, low_high_freq0 = low_dwt[:b, ...], low_dwt[b:, ...]

        low_high_freq0 = self.enhance0(low_high_freq0)

        low_LL_dwt = self.dwt(low_LL)
        low_LL_LL, low_high_freq1 = low_LL_dwt[:b, ...], low_LL_dwt[b:, ...]
        low_high_freq1 = self.enhance1(low_high_freq1)

        beta = self.betas.to(low.device)
        t = torch.randint(0, self.num_timesteps, (low_LL_LL.shape[0] // 2 + 1,)).to(low.device)
        t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:low_LL_LL.shape[0]].to(low.device)
        alpha = (1 - beta).cumprod(0).index_select(0, t).view(-1, 1, 1, 1)
        eps = torch.randn_like(low_LL_LL)

        if high is not None:
            high = self.transform(high)
            high_dwt = self.dwt(high)
            high_LL, high_high_freq0 = high_dwt[:b, ...], high_dwt[b:, ...]

            high_LL_dwt = self.dwt(high_LL)
            high_LL_LL, high_high_freq1 = high_LL_dwt[:b, ...], high_LL_dwt[b:, ...]

            x = high_LL_LL * alpha.sqrt() + eps * (1.0 - alpha).sqrt()
            noise = self.unet(torch.cat([low_LL_LL, x], dim=1), t.float())
            denoise_LL_LL = self.sample_training(low_LL_LL, beta)

            pred_high_LL = self.idwt(torch.cat([denoise_LL_LL, low_high_freq1], dim=0))
            pred_high = self.idwt(torch.cat([pred_high_LL, low_high_freq0], dim=0))
            pred_high = self.inverse_transform(pred_high)

            return {
                "low_high_freq0": low_high_freq0,
                "low_high_freq1": low_high_freq1,
                "high_high_freq0": high_high_freq0,
                "high_high_freq1": high_high_freq1,
                "pred_high_LL": pred_high_LL,
                "high_LL": high_LL,
                "pred_high": pred_high,
                "noise": noise,
                "eps": eps,
            }

        denoise_LL_LL = self.sample_training(low_LL_LL, beta)
        pred_high_LL = self.idwt(torch.cat([denoise_LL_LL, low_high_freq1], dim=0))
        pred_high = self.idwt(torch.cat([pred_high_LL, low_high_freq0], dim=0))
        pred_high = self.inverse_transform(pred_high)

        return {
            "pred_high": pred_high,
        }


class TVLoss(nn.Module):
    def __init__(self, lambda_tv: float = 1.0):
        super().__init__()
        self.lambda_tv = lambda_tv

    def _tensor_size(self, tensor: torch.Tensor):
        return tensor.shape[1] * tensor.shape[2] * tensor.shape[3]

    def forward(self, x: torch.Tensor):
        b, _, h, w = x.shape
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w - 1]), 2).sum()
        return self.lambda_tv * 2 * (h_tv / count_h + w_tv / count_w) / b


class DiffLL(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config
        model_config = config["model"]
        model_config.pop("name", None)
        self.diffusion = Diffusion(**model_config)

        self.save_path = config["data"].get("save_path", "")
        if self.save_path:
            os.makedirs(self.save_path, exist_ok=True)

        # losses
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        self.tv_loss = TVLoss()
        self.ssim_loss = SSIM()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze')

    def forward(self, low: torch.Tensor, high: Optional[torch.Tensor] = None):
        return self.diffusion(low, high)

    def configure_optimizers(self):
        train_config = self.config["train"]
        optimizer = get_optimizer(train_config, self.diffusion)
        scheduler = get_scheduler(train_config, optimizer)
        return [optimizer], [scheduler]

    def _compute_loss(self, output, high):
        low_high_freq0 = output["low_high_freq0"]
        low_high_freq1 = output["low_high_freq1"]
        high_high_freq0 = output["high_high_freq0"]
        high_high_freq1 = output["high_high_freq1"]
        pred_high_LL = output["pred_high_LL"]
        high_LL = output["high_LL"]
        pred_high = output["pred_high"]
        noise = output["noise"]
        eps = output["eps"]

        self.diffusion_loss = self.l2_loss(noise, eps)
        self.detail_loss = 0.1 * (self.l2_loss(low_high_freq0, high_high_freq0) +
                                  self.l2_loss(low_high_freq1, high_high_freq1) +
                                  self.l2_loss(pred_high_LL, high_LL)) + \
                           0.01 * (self.tv_loss(low_high_freq0) +
                                   self.tv_loss(low_high_freq1) +
                                   self.tv_loss(pred_high_LL))
        self.content_loss = self.l1_loss(pred_high, high) + (1 - self.ssim_loss(pred_high, high))

        return self.diffusion_loss + self.detail_loss + self.content_loss

    def _compute_metrics(self, output, high):
        pred_high = output["pred_high"]
        self.psnr_val = self.psnr(pred_high, high)
        self.ssim_val = self.ssim(pred_high, high)
        self.lpips_val = self.lpips(pred_high, high)

    def on_train_epoch_start(self):
        extra_logger.info(f"Epoch {self.current_epoch} starts.")
        extra_logger.info(f"Start training stage.")

    def training_step(self, batch, batch_idx):
        low, high = batch["low"], batch["high"]
        output = self.forward(low, high)
        total_loss = self._compute_loss(output, high)
        self.log_dict({
            "train/total_loss": total_loss,
            "train/diffusion_loss": self.diffusion_loss,
            "train/detail_loss": self.detail_loss,
            "train/content_loss": self.content_loss,
        }, on_step=True, on_epoch=True, batch_size=low.shape[0])
        return total_loss

    def on_train_epoch_end(self):
        scheduler = self.lr_schedulers()
        self.log("lr", scheduler.get_last_lr()[0], on_step=False, on_epoch=True)

    def on_validation_epoch_start(self):
        extra_logger.info(f"Start validation stage.")

    def validation_step(self, batch, batch_idx):
        low, high = batch["low"], batch["high"]
        output = self.forward(low, high)
        total_loss = self._compute_loss(output, high)
        self._compute_metrics(output, high)
        self.log_dict({
            "val/total_loss": total_loss,
            "val/diffusion_loss": self.diffusion_loss,
            "val/detail_loss": self.detail_loss,
            "val/content_loss": self.content_loss,
        }, on_step=True, on_epoch=True, batch_size=low.shape[0])
        self.log_dict({
            "val/psnr": self.psnr_val,
            "val/ssim": self.ssim_val,
            "val/lpips": self.lpips_val,
        }, on_step=False, on_epoch=True, batch_size=low.shape[0])

        if batch_idx == 0:
            low = low[0].detach().cpu()
            high = high[0].detach().cpu()
            pred_high = output["pred_high"][0].detach().cpu()
            image = torch.cat([low, high, pred_high], dim=2)
            image = torch.clip(image * 255.0, 0.0, 255.0).to(torch.uint8)

            self.logger.experiment.add_image(f"val/image", image, self.current_epoch)

        return total_loss

    def on_fit_start(self):
        extra_logger.info(f"Start training on {self.device}.")

    def on_fit_end(self):
        extra_logger.info(f"All training finished.")

    def test_step(self, batch, batch_idx):
        low, high = batch["low"], batch["high"]
        output = self.forward(low, high)
        total_loss = self._compute_loss(output, high)
        self._compute_metrics(output, high)
        self.log_dict({
            "test/total_loss": total_loss,
            "test/diffusion_loss": self.diffusion_loss,
            "test/detail_loss": self.detail_loss,
            "test/content_loss": self.content_loss,
        }, on_step=True, on_epoch=True, batch_size=low.shape[0])
        self.log_dict({
            "test/psnr": self.psnr_val,
            "test/ssim": self.ssim_val,
            "test/lpips": self.lpips_val,
        }, on_step=False, on_epoch=True, batch_size=low.shape[0])

        if batch_idx == 0:
            low = low[0].detach().cpu()
            high = high[0].detach().cpu()
            pred_high = output["pred_high"][0].detach().cpu()
            image = torch.cat([low, high, pred_high], dim=2)
            image = torch.clip(image * 255.0, 0.0, 255.0).to(torch.uint8)

            self.logger.experiment.add_image(f"test/image", image, self.current_epoch)

        return total_loss

    def on_test_start(self):
        extra_logger.info("Start testing.")

    def on_test_end(self):
        extra_logger.info(f"Testing finished.")

    def predict_step(self, batch, batch_idx):
        low = batch["low"]
        output = self.forward(low)
        pred_high = output["pred_high"]
        pred_high = torch.clip(pred_high * 255.0, 0.0, 255.0).to(torch.uint8)
        save_batch_tensor(pred_high, self.save_path, batch)

# endregion
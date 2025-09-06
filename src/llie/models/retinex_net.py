import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class DecomNet(nn.Module):
    def __init__(self, in_channels, num_layers, hidden_dim: int = 64, kernel_size: int = 3):
        super().__init__()

        self.in_conv = nn.Conv2d(in_channels + 1, hidden_dim, kernel_size=kernel_size * 3, padding=4, padding_mode='replicate')
        self.conv_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.conv_layers.append(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=1, padding_mode='replicate')
            )
            self.conv_layers.append(nn.ReLU())
        self.out_conv = nn.Conv2d(hidden_dim, in_channels + 1, kernel_size=kernel_size, padding=1, padding_mode='replicate')
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_max = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([x_max, x], dim=1)

        x = self.in_conv(x)
        for layer in self.conv_layers:
            x = layer(x)
        x = self.out_conv(x)

        R = self.sigmoid(x[:, 0:3, :, :])
        L = self.sigmoid(x[:, 3:4, :, :])

        return R, L


class RelightNet(nn.Module):
    def __init__(self,in_channels: int = 4, hidden_dim: int = 64, kernel_size: int = 3):
        super().__init__()

        self.in_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=kernel_size, padding=1, padding_mode='replicate')

        self.en_conv1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=2, padding=1, padding_mode='replicate')
        self.en_conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=2, padding=1, padding_mode='replicate')
        self.en_conv3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=2, padding=1, padding_mode='replicate')

        self.de_conv1 = nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, kernel_size=kernel_size, padding=1, padding_mode='replicate')
        self.de_conv2 = nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, kernel_size=kernel_size, padding=1, padding_mode='replicate')
        self.de_conv3 = nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, kernel_size=kernel_size, padding=1, padding_mode='replicate')

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


class RetinexNet(nn.Module):
    def __init__(self, in_channels: int = 4, hidden_dim: int = 64, kernel_size: int = 3, num_layers: int = 5):
        super().__init__()

        self.decom_net = DecomNet(in_channels, num_layers, hidden_dim, kernel_size)
        self.relight_net = RelightNet(in_channels, hidden_dim, kernel_size)

    def forward(self, low: torch.Tensor, high: Optional[torch.Tensor] = None):
        self.R_low, I_low = self.decom_net(low)
        self.I_low = torch.cat([I_low, I_low, I_low], dim=1)

        I_delta = self.relight_net(I_low, self.R_low)
        self.I_delta = torch.cat([I_delta, I_delta, I_delta], dim=1)

        self.S = self.R_low * self.I_delta

        if high is not None:
            self.R_high, I_high = self.decom_net(high)
            self.I_high = torch.cat([I_high, I_high, I_high], dim=1)

        self._compute_loss(low, high)

    def _compute_loss(self, low: torch.Tensor, high: Optional[torch.Tensor] = None):
        self.recon_loss_low = F.l1_loss(self.R_low * self.I_low, low)
        if high is not None:
            self.recon_loss_high = F.l1_loss(self.R_high * self.I_high, high)
            self.recon_loss_mutal_low = F.l1_loss(self.R_high * self.I_low, low)
            self.recon_loss_mutal_high = F.l1_loss(self.R_low * self.I_high, high)
            self.equal_R_loss = F.l1_loss(self.R_low, self.R_high)
            self.relight_loss = F.l1_loss(self.R_low * self.I_delta, high)

            self.smooth_loss_low = self._smooth_loss(self.I_low, self.R_low)
            self.smooth_loss_high = self._smooth_loss(self.I_high, self.R_high)
            self.smooth_loss_delta = self._smooth_loss(self.I_delta, self.R_low)

            self.decom_loss = self.recon_loss_low + self.recon_loss_high + \
                              0.001 * self.recon_loss_mutal_low + 0.001 * self.recon_loss_mutal_high + \
                              0.1 * self.smooth_loss_low + 0.1 * self.smooth_loss_high + \
                              0.01 * self.equal_R_loss
            self.relight_loss = self.relight_loss + 3 * self.smooth_loss_delta

    def _gradient(self, x: torch.Tensor, direction: str):
        self.smooth_kernel_x = torch.FloatTensor([[0, 0], [-1, 1]]).view((1, 1, 2, 2)).to(x.device)
        self.smooth_kernel_y = self.smooth_kernel_x.transpose(2, 3)

        if direction == 'x':
            kernel = self.smooth_kernel_x
        elif direction == 'y':
            kernel = self.smooth_kernel_y
        else:
            raise ValueError('direction should be x or y')

        grad = torch.abs(F.conv2d(x, kernel, stride=1, padding=1))
        return grad

    def _avg_gradient(self, x: torch.Tensor, direction: str):
        return F.avg_pool2d(self._gradient(x, direction), kernel_size=3, stride=1, padding=1)

    def _smooth_loss(self, I: torch.Tensor, R: torch.Tensor):
        R = 0.299 * R[:, 0, :, :] + 0.587 * R[:, 1, :, :] + 0.114 * R[:, 2, :, :]
        R = R.unsqueeze(1)
        return torch.mean(
            self._gradient(I, 'x') * torch.exp(-10 * self._avg_gradient(R, 'x')) +
            self._gradient(I, 'y') * torch.exp(-10 * self._avg_gradient(R, 'y'))
        )

import torch
import torch.nn as nn
from torchvision import models
from typing import List, Optional


class VGG19(nn.Module):
    def __init__(self):
        super().__init__()
        vgg_pretrained_features = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class PerceptualLoss(nn.Module):
    def __init__(self, weights: List[float], criterion: str = "l1", norm: bool = False):
        super().__init__()
        self.vgg = VGG19()
        self.vgg.requires_grad_(False)
        if criterion == "l1":
            self.criterion = nn.L1Loss()
        elif criterion == "l2":
            self.criterion = nn.MSELoss()
        elif criterion == "mse":
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Invalid criterion: {criterion}")

        self.weights = weights
        self.norm = norm

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.norm:
            x = (x + 1) / 2
            y = (y + 1) / 2
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(self.weights)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss
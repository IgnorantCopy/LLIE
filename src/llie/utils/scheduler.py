import numpy as np
import torch
from torch.optim.lr_scheduler import _LRScheduler
from collections import Counter
from typing import List, Optional


class MultiStepLRWarmup(_LRScheduler):
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


class CosineAnnealingRestartCyclicLR(_LRScheduler):
    def __init__(self, optimizer, periods: List[int], restart_weights: List[float],
                 eta_mins: List[float], last_epoch: int = -1):
        self.periods = periods
        self.restart_weights = restart_weights
        self.eta_mins = eta_mins
        self.cum_periods = torch.cumsum(torch.tensor(periods), 0)
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        index = 0
        for i, period in enumerate(self.cum_periods):
            if self.last_epoch < period:
                index = i
                break
        weight = self.restart_weights[index]
        nearset_restart = 0 if index == 0 else self.cum_periods[index - 1]
        period = self.periods[index]
        eta_min = self.eta_mins[index]

        return [
            eta_min + weight * 0.5 * (base_lr - eta_min) *
            (1 + np.cos(np.pi * ((self.last_epoch - nearset_restart) / period)))
            for base_lr in self.base_lrs
        ]

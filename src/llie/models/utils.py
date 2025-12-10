import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


def pad_tensor(tensor: torch.Tensor, div: int = 16):
    height_ori, width_ori = tensor.shape[-2:]
    height_res = height_ori % div
    width_res = width_ori % div
    if height_res == 0 and width_res == 0:
        pad_left = pad_right = pad_top = pad_bottom = 0
    else:
        if height_res == 0:
            pad_top = pad_bottom = 0
        else:
            height_diff = div - height_res
            pad_top = height_diff // 2
            pad_bottom = height_diff - pad_top
        if width_res == 0:
            pad_left = pad_right = 0
        else:
            width_diff = div - width_res
            pad_left = width_diff // 2
            pad_right = width_diff - pad_left

        tensor = nn.ReflectionPad2d((pad_left, pad_right, pad_top, pad_bottom))(tensor)

    return tensor, pad_left, pad_right, pad_top, pad_bottom


def pad_tensor_back(tensor: torch.Tensor, pad_left: int, pad_right: int, pad_top: int, pad_bottom: int):
    h, w = tensor.shape[-2:]
    return tensor[..., pad_top: h - pad_bottom, pad_left: w - pad_right]


def gradient(tensor: torch.Tensor, direction: str, no_abs=False):
    smooth_kernel_x = torch.FloatTensor([[0, 0], [-1, 1]]).view((1, 1, 2, 2)).to(tensor)
    smooth_kernel_y = smooth_kernel_x.transpose(2, 3)

    if direction == 'x':
        kernel = smooth_kernel_x
    elif direction == 'y':
        kernel = smooth_kernel_y
    else:
        raise ValueError('direction should be x or y')

    grad = F.conv2d(tensor, kernel, stride=1, padding=1)
    if not no_abs:
        grad = torch.abs(grad)

    return grad


def save_batch_tensor(tensor: torch.Tensor, path: str, batch: dict):
    low_path = batch["low_path"]
    height, width = batch["height"], batch["width"]
    for i in range(tensor.shape[0]):
        save_path = os.path.join(path, os.path.basename(low_path[i]))
        Image.fromarray(tensor[i].permute(1, 2, 0).numpy()).resize((width[i], height[i]), Image.Resampling.BICUBIC).save(save_path)

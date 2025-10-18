import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.llie.data.utils import DataModuleFromConfig, Compose, RandomCrop, ToTensor, Resize, RandomHorizontalFlip, RandomVerticalFlip


class FlowDataset(Dataset):
    def __init__(self, father_dataset: Dataset):
        super().__init__()
        self.father_dataset = father_dataset
        self.low_image_paths = self.father_dataset.low_image_paths
        self.high_image_paths = self.father_dataset.high_image_paths if hasattr(self.father_dataset, 'high_image_paths') else None
        self.transform = self.father_dataset.transform

    def __len__(self):
        return len(self.father_dataset)

    def __getitem__(self, index):
        low_image_path = self.low_image_paths[index]
        high_image_path = self.high_image_paths[index] if self.high_image_paths is not None else None
        low_image = cv2.cvtColor(cv2.imread(low_image_path), cv2.COLOR_BGR2RGB)
        if high_image_path is not None:
            high_image = cv2.cvtColor(cv2.imread(high_image_path), cv2.COLOR_BGR2RGB)
        else:
            high_image = None

        hist_equalized_image = self._get_hist_equalized_image(low_image)
        original_size = (low_image.shape[0], low_image.shape[1])

        if self.transform is not None:
            if high_image is not None:
                low_image, hist_equalized_image, high_image = self.transform(
                    low_image,
                    hist_equalized_image,
                    high_image,
                )
            else:
                low_image, hist_equalized_image = self.transform(
                    low_image,
                    hist_equalized_image,
                )
            color_map = self._get_color_map(low_image)
            noise_map = self._get_noise_map(color_map)
            low_image = torch.log(torch.clip(low_image + 1e-3, min=1e-3))
            low_image = torch.concat([low_image, hist_equalized_image, color_map, noise_map], dim=0).float()

        if high_image is not None:
            return {
                "low": low_image,
                "high": high_image,
                "low_path": low_image_path,
                "high_path": high_image_path,
                "height": original_size[0],
                "width": original_size[1],
            }
        return {
            "low": low_image,
            "low_path": low_image_path,
            "height": original_size[0],
            "width": original_size[1],
        }

    @staticmethod
    def _get_hist_equalized_image(image: np.ndarray):
        r, g, b = cv2.split(image)
        r_hist = cv2.equalizeHist(r)
        g_hist = cv2.equalizeHist(g)
        b_hist = cv2.equalizeHist(b)
        hist_equalized_image = cv2.merge((r_hist, g_hist, b_hist))
        return hist_equalized_image

    @staticmethod
    def _get_color_map(image: torch.Tensor):
        image = image / (image.sum(dim=0, keepdim=True) + 1e-4)
        return image

    @staticmethod
    def _get_noise_map(image: torch.Tensor):
        def gradient(x: torch.Tensor):
            left_shift_x, right_shift_x = torch.zeros_like(x), torch.zeros_like(x)
            left_shift_x[:, :-1] = x[:, 1:]
            right_shift_x[:, 1:] = x[:, :-1]
            return torch.abs(0.5 * (left_shift_x - right_shift_x))

        dx, dy = gradient(image), gradient(image.transpose(1, 2)).transpose(1, 2)
        noise_map = torch.max(torch.stack([dx, dy], dim=0), dim=0)[0]
        return noise_map


class FlowDataModule(DataModuleFromConfig):
    def __init__(self, father_datamodule: DataModuleFromConfig):
        super().__init__(father_datamodule.data_config)
        self.father_datamodule = father_datamodule
        crop_size = self.father_datamodule.data_config["crop_size"]
        self.father_datamodule.train_transform = Compose([
            ToTensor(),
            Resize((int(self.image_height), int(self.image_width))),
            RandomCrop(crop_size),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
        ])

    def setup(self, stage=None):
        self.father_datamodule.setup(stage)
        self.train_dataset = FlowDataset(self.father_datamodule.train_dataset.dataset)
        self.val_dataset = FlowDataset(self.father_datamodule.val_dataset.dataset)
        self.test_dataset = FlowDataset(self.father_datamodule.test_dataset)
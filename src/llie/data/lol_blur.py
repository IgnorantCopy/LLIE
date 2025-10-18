import os
import glob
import random
from torch.utils.data import Dataset
from PIL import Image
from typing import Tuple, List

from src.llie.data.utils import split_train_val, DataModuleFromConfig


class LOLBlurDataset(Dataset):
    def __init__(self, root_dir: str, split: str, transform=None):
        super().__init__()
        assert split in ['train', 'test']

        self.data_root = os.path.join(root_dir, split)
        self.transform = transform
        self.high_image_paths, self.low_image_paths = self._load_data()

    def _load_data(self) -> Tuple[List[str], List[str]]:
        high_image_dirs = sorted(glob.glob(os.path.join(self.data_root, 'high_sharp_scaled', '*')))
        low_image_dirs = sorted(glob.glob(os.path.join(self.data_root, 'low_blur', '*')))

        high_image_paths = []
        low_image_paths = []
        for high_image_dir, low_image_dir in zip(high_image_dirs, low_image_dirs):
            high_images = glob.glob(os.path.join(high_image_dir, '*.png'))
            low_images = glob.glob(os.path.join(low_image_dir, '*.png'))
            index = random.randint(0, len(high_images) - 1)
            high_image_path = high_images[index]
            low_image_path = low_images[index]
            high_image_paths.append(high_image_path)
            low_image_paths.append(low_image_path)

        assert len(high_image_paths) == len(low_image_paths)

        return high_image_paths, low_image_paths

    def __len__(self) -> int:
        return len(self.high_image_paths)

    def __getitem__(self, idx: int):
        high_image_path = self.high_image_paths[idx]
        low_image_path = self.low_image_paths[idx]
        high_image = Image.open(high_image_path).convert('RGB')
        low_image = Image.open(low_image_path).convert('RGB')

        original_size = high_image.size

        if self.transform is not None:
            low_image, high_image = self.transform(low_image, high_image)

        return {
            "high": high_image,
            "low": low_image,
            "high_path": high_image_path,
            "low_path": low_image_path,
            "height": original_size[0],
            "width": original_size[1],
        }


class LOLBlurDataModule(DataModuleFromConfig):
    def __init__(self, data_config):
        super().__init__(data_config)

    def setup(self, stage: str):
        train_dataset = LOLBlurDataset(root_dir=self.root, split="train", transform=self.train_transform)
        self.train_dataset, self.val_dataset = split_train_val(self.data_config, train_dataset)
        self.val_dataset.transform = self.test_transform
        self.test_dataset = LOLBlurDataset(root_dir=self.root, split="test", transform=self.test_transform)

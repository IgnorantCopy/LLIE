import os
import glob
import random
from torch.utils.data import Dataset
from PIL import Image
from typing import Tuple, List

from src.llie.data.utils import DataModuleFromConfig


class SIDDataset(Dataset):
    def __init__(self, root_dir: str, split: str, transform=None):
        super().__init__()
        assert split in ['train', 'eval', 'test']

        self.data_root = os.path.join(root_dir, split)
        self.split = split
        self.transform = transform
        self.high_image_paths, self.low_image_paths = self._load_data()

    def _load_data(self) -> Tuple[List[str], List[str]]:
        if self.split == "eval":
            high_image_paths = sorted(glob.glob(os.path.join(self.data_root, 'long', '*.png')))
            low_image_paths = sorted(glob.glob(os.path.join(self.data_root, 'short', '*.png')))
        else:
            high_image_dirs = sorted(glob.glob(os.path.join(self.data_root, 'long', '*')))
            low_image_dirs = sorted(glob.glob(os.path.join(self.data_root, 'short', '*')))

            high_image_paths = []
            low_image_paths = []
            for high_image_dir, low_image_dir in zip(high_image_dirs, low_image_dirs):
                high_image_path = glob.glob(os.path.join(high_image_dir, '*.png'))[0]
                low_images = glob.glob(os.path.join(low_image_dir, '*.png'))
                low_image_path = random.choice(low_images)
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

        if self.transform is not None:
            high_image = self.transform(high_image)
            low_image = self.transform(low_image)

        return {
            "high": high_image,
            "low": low_image,
            "high_path": high_image_path,
            "low_path": low_image_path
        }


class SIDDataModule(DataModuleFromConfig):
    def __init__(self, data_config):
        super().__init__(data_config)

    def setup(self, stage: str):
        self.train_dataset = SIDDataset(root_dir=self.root, split='train', transform=self.train_transform)
        self.val_dataset = SIDDataset(root_dir=self.root, split='eval', transform=self.test_transform)
        self.test_dataset = SIDDataset(root_dir=self.root, split='test', transform=self.test_transform)
import os
import glob
from torch.utils.data import Dataset
from PIL import Image
from typing import Tuple, List

from src.llie.data.utils import split_train_val, DataModuleFromConfig


_metadata = {
    "LOLv1": {
        "train": "our485",
        "test": "eval15",
        "high": "high",
        "low": "low",
    },
    "Real_captured": {
        "train": "Train",
        "test": "Test",
        "high": "Normal",
        "low": "Low",
    },
    "Synthetic": {
        "train": "Train",
        "test": "Test",
        "high": "Normal",
        "low": "Low",
    },
}


class LOLDataset(Dataset):
    def __init__(self, root_dir: str, split: str, transform=None):
        super().__init__()
        assert split in ['train', 'test']

        self.metadata = _metadata[os.path.basename(root_dir)]
        self.data_root = os.path.join(root_dir, self.metadata[split])
        self.split = split
        self.transform = transform
        self.high_image_paths, self.low_image_paths = self._load_data()

    def _load_data(self) -> Tuple[List[str], List[str]]:
        high_image_paths = sorted(glob.glob(os.path.join(self.data_root, self.metadata['high'], "*.png")))
        low_image_paths = sorted(glob.glob(os.path.join(self.data_root, self.metadata['low'], "*.png")))

        assert len(high_image_paths) == len(low_image_paths)

        return high_image_paths, low_image_paths

    def __len__(self):
        return len(self.low_image_paths)

    def __getitem__(self, idx):
        high_image_path = self.high_image_paths[idx]
        low_image_path = self.low_image_paths[idx]

        assert os.path.basename(high_image_path) == os.path.basename(low_image_path)

        high_image = Image.open(high_image_path).convert('RGB')
        low_image = Image.open(low_image_path).convert('RGB')

        original_size = high_image.size

        if self.transform:
            high_image = self.transform(high_image)
            low_image = self.transform(low_image)

        return {
            "high": high_image,
            "low": low_image,
            "high_path": high_image_path,
            "low_path": low_image_path,
            "height": original_size[0],
            "width": original_size[1],
        }


class LOLDataModule(DataModuleFromConfig):
    def __init__(self, data_config):
        super().__init__(data_config)

    def setup(self, stage: str):
        train_dataset = LOLDataset(root_dir=self.root, split="train", transform=self.train_transform)
        self.train_dataset, self.val_dataset = split_train_val(self.data_config, train_dataset)
        self.val_dataset.transform = self.test_transform
        self.test_dataset = LOLDataset(root_dir=self.root, split="test", transform=self.test_transform)
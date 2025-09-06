import os
import glob
import random
from torch.utils.data import Dataset
from PIL import Image
from typing import Tuple, List


class SICEDataset(Dataset):
    def __init__(self, root_dir: str, split: str, transform=None):
        super().__init__()
        assert split in ['train', 'eval']

        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.high_image_paths, self.low_image_paths = self._load_data()

    def _load_data(self) -> Tuple[List[str], List[str]]:
        if self.split == 'train':
            high_image_dir = os.path.join(self.root_dir, "label")
            low_image_dir = os.path.join(self.root_dir, "train")
        elif self.split == 'eval':
            high_image_dir = os.path.join(self.root_dir, "eval", "target")
            low_image_dir = os.path.join(self.root_dir, "eval", "test")
        else:
            raise ValueError("Invalid split: {}".format(self.split))

        high_image_paths = sorted(glob.glob(os.path.join(high_image_dir, "*.JPG")))
        low_image_paths = []
        for high_image_path in high_image_paths:
            image_name = os.path.basename(high_image_path)
            if self.split == 'train':
                low_images = glob.glob(os.path.join(low_image_dir, image_name.split('.')[0], "*.JPG"))
                low_image_path = random.choice(low_images)
            else:
                low_image_path = os.path.join(low_image_dir, image_name)
            if os.path.exists(low_image_path):
                low_image_paths.append(low_image_path)
            else:
                high_image_paths.remove(high_image_path)

        assert len(high_image_paths) == len(low_image_paths)

        return high_image_paths, low_image_paths

    def __len__(self):
        return len(self.high_image_paths)

    def __getitem__(self, idx):
        high_image_path = self.high_image_paths[idx]
        low_image_path = self.low_image_paths[idx]
        high_image = Image.open(high_image_path).convert('RGB')
        low_image = Image.open(low_image_path).convert('RGB')

        if self.transform is not None:
            high_image = self.transform(high_image)
            low_image = self.transform(low_image)

        return high_image, low_image
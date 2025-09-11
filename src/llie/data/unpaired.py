import os
import glob
from torch.utils.data import Dataset
from PIL import Image

from src.llie.data.utils import DataModuleFromConfig


class UnpairedDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        super().__init__()

        self.root = root_dir
        self.transform = transform
        self.image_paths = self._load_data()

    def _load_data(self):
        image_dirs = glob.glob(os.path.join(self.root, '*'))
        image_paths = []
        for image_dir in image_dirs:
            image_paths.extend(glob.glob(os.path.join(image_dir, '*')))
        return sorted(image_paths)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image


class UnpairedDataModule(DataModuleFromConfig):
    def __init__(self, data_config):
        super().__init__(data_config)

    def setup(self, stage: str):
        self.test_dataset = UnpairedDataset(root_dir=self.root, transform=self.test_transform)
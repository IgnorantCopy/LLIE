import os
import glob
from torch.utils.data import Dataset
from PIL import Image
from PIL import ImageFile

from src.llie.data.utils import split_train_val
from src.llie.data.utils import DataModuleFromConfig

ImageFile.LOAD_TRUNCATED_IMAGES = True


class UnpairedDataset(Dataset):
    def __init__(self, root_dir: str, transform=None, sub_folder: bool = False):
        super().__init__()

        self.root = root_dir
        self.transform = transform
        self.sub_folder = sub_folder
        self.image_paths = self._load_data()

    def _load_data(self):
        if not self.sub_folder:
            image_paths = sorted(glob.glob(os.path.join(self.root, '*')))
        else:
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

        original_size = image.size

        if self.transform is not None:
            image = self.transform(image)

        return {
            "low": image,
            "low_path": image_path,
            "height": original_size[0],
            "width": original_size[1],
        }


class UnpairedDataModule(DataModuleFromConfig):
    def __init__(self, data_config):
        super().__init__(data_config)
        self.sub_folder = data_config.get('sub_folder', False)

    def setup(self, stage: str):
        train_dataset = UnpairedDataset(root_dir=self.root, transform=self.train_transform, sub_folder=self.sub_folder)
        self.train_dataset, self.val_dataset = split_train_val(self.data_config, train_dataset)
        self.val_dataset.transform = self.test_transform
        self.test_dataset = UnpairedDataset(root_dir=self.root, transform=self.test_transform, sub_folder=self.sub_folder)


class UnpairedGANDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        super().__init__()
        self.dirA = os.path.join(root_dir, "trainA")    # low images
        self.dirB = os.path.join(root_dir, "trainB")    # high images
        self.pathsA, self.pathsB = self._load_data()
        self.transform = transform

    def _load_data(self):
        paths_a = sorted(glob.glob(os.path.join(self.dirA, "*.png")))
        paths_b = sorted(glob.glob(os.path.join(self.dirB, "*.png")))
        return paths_a, paths_b

    def __len__(self) -> int:
        return max(len(self.pathsA), len(self.pathsB))

    def __getitem__(self, index: int):
        path_a = self.pathsA[index % len(self.pathsA)]
        path_b = self.pathsB[index % len(self.pathsB)]
        img_a = Image.open(path_a).convert('RGB')
        img_b = Image.open(path_b).convert('RGB')
        if self.transform is not None:
            img_a = self.transform(img_a)
            img_b = self.transform(img_b)
        return {
            "low": img_a,
            "high": img_b,
            "low_path": path_a,
            "high_path": path_b,
        }


class UnpairedGANDataModule(DataModuleFromConfig):
    def __init__(self, data_config):
        super().__init__(data_config)

    def setup(self, stage: str):
        train_dataset = UnpairedGANDataset(root_dir=self.root, transform=self.train_transform)
        self.train_dataset, self.val_dataset = split_train_val(self.data_config, train_dataset)
        self.val_dataset.transform = self.test_transform
        self.test_dataset = UnpairedGANDataset(root_dir=self.root, transform=self.test_transform)

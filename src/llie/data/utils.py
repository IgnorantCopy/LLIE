import random
import torch
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as F
import lightning as pl
from typing import Union


def split_train_val(data_config, dataset):
    val_ratio = data_config.get("val_ratio", 0.2)
    seed = data_config.get("seed", 42)
    train_size = int(len(dataset) * (1 - val_ratio))
    val_size = len(dataset) - train_size
    seed = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=seed)


class Resize(object):
    def __init__(self, size: Union[int, tuple]):
        self.resize = transforms.Resize(size)

    def __call__(self, *images):
        result = [self.resize(image) for image in images]
        return *result,


class ToTensor(object):
    def __init__(self):
        self.to_tensor = transforms.ToTensor()

    def __call__(self, *images):
        result = [image.float() if isinstance(image, torch.Tensor) else self.to_tensor(image) for image in images]
        return *result,


class Normalize(object):
    def __init__(self, mean, std):
        self.normalize = transforms.Normalize(mean, std)

    def __call__(self, *images):
        result = [self.normalize(image) for image in images]
        return *result,


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, *images):
        flip = torch.rand(1) < self.p
        result = [F.hflip(image) if flip else image for image in images]
        return *result,


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, *images):
        flip = torch.rand(1) < self.p
        result = [F.vflip(image) if flip else image for image in images]
        return *result,


class RandomCrop(object):
    def __init__(self, size: Union[int, tuple]):
        self.size = size if isinstance(size, tuple) else (size, size)

    def __call__(self, *images):
        _, size_x, size_y = images[0].size()
        start_x = random.randint(0, size_x - self.size[0] + 1) if size_x > self.size[0] else 0
        start_y = random.randint(0, size_y - self.size[1] + 1) if size_y > self.size[1] else 0
        result = [F.crop(image, start_x, start_y, self.size[0], self.size[1]) for image in images]
        return *result,

class Compose(object):
    def __init__(self, transform):
        self.transforms = transform

    def __call__(self, *images):
        for t in self.transforms:
            if t is not None:
                images = t(*images)
        return images


class MixUp(object):
    def __init__(self, beta):
        self.dist = torch.distributions.beta.Beta(torch.tensor([beta]), torch.tensor([beta]))

    def __call__(self, x, target):
        lam = self.dist.rsample((1, 1)).item()
        r_index = torch.randperm(x.size(0))

        target = lam * target + (1 - lam) * target[r_index]
        x = lam * x + (1 - lam) * x[r_index]

        return x, target


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, data_config):
        super().__init__()
        self.data_config = data_config
        self.root = data_config["root"]
        self.batch_size = data_config["batch_size"]
        self.image_height = data_config["height"]
        self.image_width = data_config["width"]
        self.crop_size = data_config.get("crop_size", None)
        self.num_workers = data_config.get("num_workers", 4)
        self.pin_memory = data_config.get("pin_memory", True)

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_transform = Compose([
            ToTensor(),
            Resize((int(self.image_height), int(self.image_width))),
            RandomCrop(self.crop_size) if self.crop_size is not None else None,
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
        ])
        self.test_transform = Compose([
            ToTensor(),
            Resize((int(self.image_height), int(self.image_width))),
        ])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory)
import torch
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
import lightning as pl


def split_train_val(data_config, dataset):
    val_ratio = getattr(data_config, "val_ratio", 0.2)
    seed = getattr(data_config, "seed", 42)
    train_size = int(len(dataset) * (1 - val_ratio))
    val_size = len(dataset) - train_size
    seed = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=seed)


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, data_config):
        super().__init__()
        self.data_config = data_config
        self.root = data_config["root"]
        self.batch_size = data_config["batch_size"]
        self.image_height = data_config["height"]
        self.image_width = data_config["width"]
        self.num_workers = getattr(data_config, "num_workers", 4)
        self.pin_memory = getattr(data_config, "pin_memory", True)

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_transform = transforms.Compose([
            transforms.Resize((int(self.image_height), int(self.image_width))),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize((int(self.image_height), int(self.image_width))),
            transforms.ToTensor(),
        ])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory)
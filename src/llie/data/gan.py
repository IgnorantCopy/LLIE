from torch.utils.data import Dataset
from torchvision import transforms

from src.llie.data.utils import DataModuleFromConfig


class GANDataset(Dataset):
    def __init__(self, father_dataset: Dataset):
        super().__init__()
        self.father_dataset = father_dataset

    def __len__(self):
        return len(self.father_dataset)

    def __getitem__(self, index):
        res = self.father_dataset[index]
        low = res["low"]
        r, g, b = (low[0] + 1) / 2, (low[1] + 1) / 2, (low[2] + 1) / 2
        attn_map = 1. - (0.299 * r + 0.587 * g + 0.114 * b).unsqueeze(0)
        res["attn_map"] = attn_map
        return res


class GANDataModule(DataModuleFromConfig):
    def __init__(self, father_datamodule: DataModuleFromConfig):
        super().__init__(father_datamodule.data_config)
        self.father_datamodule = father_datamodule
        self.father_datamodule.train_transform = transforms.Compose([
            self.father_datamodule.train_transform,
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])
        self.father_datamodule.test_transform = transforms.Compose([
            self.father_datamodule.test_transform,
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])

    def setup(self, stage: str):
        self.father_datamodule.setup(stage)
        self.train_dataset = GANDataset(self.father_datamodule.train_dataset)
        self.val_dataset = GANDataset(self.father_datamodule.val_dataset)
        self.test_dataset = GANDataset(self.father_datamodule.test_dataset)
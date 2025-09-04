import os
import glob
from torch.utils.data import Dataset
from PIL import Image


class UnpairedDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        super().__init__()

        self.transform = transform
        self.image_paths = sorted(glob.glob(os.path.join(root_dir, '*')))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image
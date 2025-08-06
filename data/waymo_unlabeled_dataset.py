from torch.utils.data import Dataset
import torch
import os

class WaymoUnlabeledDataset(Dataset):
    def __init__(self, root_dir, file_list, transform=None):
        self.root_dir = root_dir.rstrip('/')
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.root_dir, file_name)

        with open(file_path, 'rb') as f:
            sample = torch.load(f)

        image = sample['image']

        if self.transform:
            image = self.transform(image)

        return image, file_name

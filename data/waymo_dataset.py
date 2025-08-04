import torch
from torch.utils.data import Dataset
import os

class WaymoDataset(Dataset):
    def __init__(self, gcs_prefix, file_list, transform=None, label_map=None):
        self.gcs_prefix = gcs_prefix.rstrip('/')
        self.file_list = file_list
        self.transform = transform
        self.label_map = label_map

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        for _ in range(len(self.file_list)):
            file_name = self.file_list[idx]
            file_path = os.path.join(self.gcs_prefix, file_name)

            try:
                with open(file_path, 'rb') as f:
                    sample = torch.load(f)
            except Exception as e:
                raise RuntimeError(f"Failed to load file: {file_path}, error: {e}")

            image = sample['image']
            boxes = sample['boxes']
            labels = sample['labels']

            # Skip frames with no boxes
            if boxes.shape[0] == 0:
                idx = (idx + 1) % len(self.file_list)
                continue

            if self.label_map is not None:
                labels = torch.tensor([self.label_map.get(int(lbl), 0) for lbl in labels], dtype=torch.int64)

            target = {
                'boxes': boxes,
                'labels': labels
            }

            if self.transform:
                image = self.transform(image)

            return image, target

        raise RuntimeError("All samples in dataset have no boxes.")

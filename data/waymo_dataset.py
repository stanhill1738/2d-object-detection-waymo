import torch
from torch.utils.data import Dataset
import gcsfs

class WaymoDataset(Dataset):
    def __init__(self, gcs_prefix, file_list, transform=None, label_map=None):
        self.fs = gcsfs.GCSFileSystem(token='google_default')
        self.gcs_prefix = gcs_prefix.rstrip('/')  # remove trailing slash if present
        self.file_list = file_list
        self.transform = transform
        self.label_map = label_map

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = f"{self.gcs_prefix}/{file_name}"

        with self.fs.open(file_path, 'rb') as f:
            sample = torch.load(f)

        image = sample['image']
        boxes = sample['boxes']
        labels = sample['labels']

        # Skip frames with no boxes
        if boxes.shape[0] == 0:
            # Try next index (wrap around if at end)
            return self.__getitem__((idx + 1) % len(self))

        # Apply label map if defined
        if self.label_map is not None:
            labels = torch.tensor([self.label_map.get(int(lbl), 0) for lbl in labels], dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels
        }

        if self.transform:
            image = self.transform(image)

        return image, target

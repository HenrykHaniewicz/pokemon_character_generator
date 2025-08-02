import os
import json
from PIL import Image
from torch.utils.data import Dataset
import random

class SpriteDataset(Dataset):
    def __init__(self, image_dir, label_file, transform=None, encoder=None):
        with open(label_file) as f:
            self.labels = json.load(f)
        self.image_dir = image_dir
        self.transform = transform
        self.filenames = list(self.labels.keys())
        self.encoder = encoder

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        meta = self.labels[img_name]
        meta_vec = self.encoder.encode(meta)

        return image, meta_vec

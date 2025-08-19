import os
import json
from PIL import Image
from torch.utils.data import Dataset

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

class GradedSpriteDataset(Dataset):
    def __init__(self, graded_file, transform=None, encoder=None, include_original=None, min_quality=None):
        with open(graded_file) as f:
            graded_data = json.load(f)

        labels = {}
        for img_path, data in graded_data.items():
            if (min_quality is None) or (data.get("quality_rating", 0) >= min_quality):
                img_name = os.path.basename(img_path)
                labels[img_name] = {
                    "meta": data.get("meta_corrected", data.get("meta", {})),
                    "quality": data.get("quality_rating", 10),
                    "image_path": img_path
                }

        # Merge original dataset (assumed to all have quality of 10) if specified
        if include_original:
            image_dir, labels_file = include_original
            with open(labels_file) as f:
                original_labels = json.load(f)
            for img_name, meta in original_labels.items():
                labels[img_name] = {
                    "meta": meta,
                    "quality": 10,
                    "image_path": os.path.join(image_dir, img_name)
                }

        self.labels = labels
        self.filenames = list(labels.keys())
        self.transform = transform
        self.encoder = encoder

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        entry = self.labels[img_name]
        image = Image.open(entry["image_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        meta_vec = self.encoder.encode(entry["meta"])
        quality = entry["quality"]
        return image, meta_vec, quality

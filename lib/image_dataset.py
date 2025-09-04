import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import cv2 as cv

class ImageDataset(Dataset):

    def __init__(self, csv_path, image_dir, transform=None):
        df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform

        # Convert one-hot to class ID
        self.samples = [
            (row['filename'], 0 if row['healthy'] == 1 else 1)
            for _, row in df.iterrows()
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, label = self.samples[idx]
        path = os.path.join(self.image_dir, filename)
        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
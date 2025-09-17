import timm
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score
import torch.nn as nn
import torch.optim as optim
from ultralytics import YOLO
import os
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import argparse
from ultralytics import YOLO
# ========================================================= #
import os
import pandas as pd
from PIL import Image
from torchvision import transforms as T
from torch.utils.data import Dataset
import cv2 as cv

class ImageDataset(Dataset):

    def __init__(self, csv_path, image_dir, base_transform, aug_transform=None, copies=3):
        df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.base_transform = base_transform
        self.aug_transform = aug_transform
        self.copies = copies

        # Convert one-hot to class ID
        self.samples = [
            (row['filename'], 0 if row['healthy'] == 1 else 1)
            for _, row in df.iterrows()
        ]

    def __len__(self):

        if self.aug_transform is None:
            return len(self.samples)

        return len(self.samples) * (1+self.copies)

    def __getitem__(self, idx):

        n = len(self.samples)
        orig_idx = idx % n # seleziona immagine originale
        copy_id  = idx // n # 0 = versione “base”, >0 = augmented

        filename, label = self.samples[orig_idx]
        path = os.path.join(self.image_dir, filename)
        image = Image.open(path).convert("RGB")

        if copy_id == 0 or self.aug_transform is None:
            # versione base (senza augment)
            return self.base_transform(image), label
        else:
            # ogni accesso genera una augment diversa (random)
            return self.aug_transform(image), label

# ========================================================= #
def yolo():
    # Load a COCO-pretrained YOLO11n model
    models_path = Path('models')
    dataset_path = Path('dataset/potatoes-v11')
    model = YOLO(models_path / "yolo11n.pt")
    results = model.train(data=dataset_path / "data.yaml", epochs=20, imgsz=640)
    print(results)

def resnet50():

    data_dir = Path('dataset/rotten_healthy')
    export_dir = Path('models')

    base_transform = T.Compose([
        T.Resize((224, 224)),
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        T.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ])

    aug_transform = T.Compose([
        T.Resize((232, 232)),
        T.RandomCrop(224, padding=2, padding_mode="reflect"),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10, expand=False),
        T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        T.ToTensor(),
        T.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ])

    train_dataset = ImageDataset(
        csv_path=data_dir/'train/_classes.csv',
        image_dir=data_dir/'train',
        base_transform=base_transform,
        aug_transform=aug_transform,
        copies=3
    )

    origin_dataset = ImageDataset(
        csv_path=data_dir/'train/_classes.csv',
        image_dir=data_dir/'train',
        base_transform=base_transform
    )

    print(f"Original train samples: {len(origin_dataset)}")
    print(f"Augmented train samples: {len(train_dataset)}")

    test_dataset = ImageDataset(
        csv_path=data_dir/'test/_classes.csv',
        image_dir=data_dir/'test',
        base_transform=base_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = timm.create_model('resnet50', pretrained=True)

    for param in model.parameters():
        param.requires_grad = True

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 1)
    model.to(device)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    num_epochs = 20

    model.train()
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs} ..")
        for imgs, labels in train_loader:

            imgs = imgs.to(device)
            labels = labels.to(device).view(-1,1).float()
            pred = model(imgs)
            loss = loss_fn(pred, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    model.eval()

    y_true = []
    y_preds = []

    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        preds = (torch.sigmoid(outputs) >= 0.5).float()

        y_true.extend(labels.cpu().numpy())
        y_preds.extend(preds.cpu().numpy())

    accuracy = accuracy_score(y_true, y_preds)
    precision = precision_score(y_true, y_preds)
    recall = recall_score(y_true, y_preds)

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")

    model = model.cpu()
    torch.save(model, export_dir/"classifier.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='Model Trainer',
                    description='Train YOLO and ResNet50 models for potato classification')
    parser.add_argument('--model', type=str, choices=['yolo', 'resnet50', 'all'], default='all',)
    args = parser.parse_args()
    if args.model == 'yolo':
        yolo()
    elif args.model == 'resnet50':
        resnet50()
    else:
        resnet50()
        yolo()
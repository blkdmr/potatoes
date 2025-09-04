import timm
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score
import torch.nn as nn
import torch.optim as optim
from ultralytics import YOLO
import os
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import joblib
# ========================================================= #
from lib.image_dataset import ImageDataset
# ========================================================= #

def resnet50():

    data_dir = Path('dataset/rotten_healthy')
    export_dir = Path('models')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = ImageDataset(
        csv_path=data_dir/'train/_classes.csv',
        image_dir=data_dir/'train',
        transform=transform
    )

    test_dataset = ImageDataset(
        csv_path=data_dir/'test/_classes.csv',
        image_dir=data_dir/'test',
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = timm.create_model('resnet50', pretrained=True)

    for param in model.parameters():
        param.requires_grad = True

    num_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_features, 1))
    model.to(device)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    num_epochs = 50

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

def yolo():
    # Load a COCO-pretrained YOLO11n model
    model = YOLO("models/yolo11n.pt")
    results = model.train(data="dataset/potatoes-v11/data.yaml", epochs=100, imgsz=640)

def main():
    resnet50()

if __name__ == "__main__":
    main()
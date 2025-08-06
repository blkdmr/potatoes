import timm
import torch
import joblib
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ========================================================= #
# Dataset for loading potato images
import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

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

# ========================================================= #

def load_and_preprocess_data(data_dir, backbone, device):

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_dataset = ImageDataset(
        csv_path=os.path.join(data_dir, 'train/_classes.csv'),
        image_dir=os.path.join(data_dir, 'train'),
        transform=transform
    )

    test_dataset = ImageDataset(
        csv_path=os.path.join(data_dir, 'test/_classes.csv'),
        image_dir=os.path.join(data_dir, 'test'),
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    train_features = []
    train_labels = []

    with torch.no_grad():
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            feats = backbone.forward_features(imgs)
            pooled = feats.mean(dim=[2, 3])  # global average pooling to [B, 2048]
            train_features.append(pooled.cpu())
            train_labels.append(labels)

    X_train = torch.cat(train_features).numpy()
    y_train = torch.cat(train_labels).numpy()

    test_features = []
    test_labels = []

    with torch.no_grad():

        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            feats = backbone.forward_features(imgs)
            pooled = feats.mean(dim=[2, 3])

            test_features.append(pooled.cpu())
            test_labels.append(labels)

    X_test = torch.cat(test_features).numpy()
    y_test = torch.cat(test_labels).numpy()

    return X_train, y_train, X_test, y_test

def load_backbone(backbone_name='resnet50'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone = timm.create_model(backbone_name, pretrained=True)

    # Freeze the model
    backbone.eval()  # disables dropout, batchnorm updates
    for param in backbone.parameters():
        param.requires_grad = False

    backbone.to(device)
    return backbone, device

def main():
    backbone, device = load_backbone('resnet50')
    data_dir = '../dataset/rotten_healthy'
    export_dir = '../models'
    X_train, y_train, X_test, y_test = load_and_preprocess_data(data_dir, backbone, device)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    test_preds = clf.predict(X_test)
    print(f"Test Accuracy: {accuracy_score(y_test, test_preds):.2%}")
    print(classification_report(y_test, test_preds, target_names=["Healthy", "Rotten"]))

    # Save the model
    os.makedirs(export_dir, exist_ok=True)
    joblib.dump(clf, f'{export_dir}/rotten_healthy_classifier.pkl')

if __name__ == "__main__":
    main()

import numpy as np
from PIL import Image
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models, transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score


class BinaryImageFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.samples = []
        self.class_to_idx = {
            "benign": 0,
            "malignant": 1
        }

        for class_name, class_idx in self.class_to_idx.items():
            class_dir = os.path.join(root_dir, class_name)
            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                    self.samples.append(
                        (os.path.join(class_dir, fname), class_idx)
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
    

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return accuracy_score(all_labels, all_preds)


def k_fold_train(
    data_root,
    k=10,
    num_epochs=10,
    batch_size=32,
    lr=1e-4
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = BinaryImageFolderDataset(data_root, transform=transform)
    kfold = KFold(n_splits=k, shuffle=True, random_state=69420)

    fold_accuracies = []

    for fold, (train_idx, test_idx) in enumerate(kfold.split(dataset)):
        print(f"\n===== Fold {fold + 1}/{k} =====")

        train_subset = Subset(dataset, train_idx)
        test_subset = Subset(dataset, test_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

        # Load ResNet
        model = models.resnet18(pretrained=True)

        # Replace final layer for binary classification
        model.fc = nn.Linear(model.fc.in_features, 2)
        model = model.to(device)

        # Train ALL weights
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")

        acc = evaluate(model, test_loader, device)
        print(f"Fold {fold + 1} Test Accuracy: {acc:.4f}")

        fold_accuracies.append(acc)

    avg_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)

    print("\n===== Cross-Validation Result =====")
    print(f"Average Accuracy: {avg_acc:.4f}")
    print(f"Std Accuracy: {std_acc:.4f}")

    return avg_acc, std_acc


if __name__ == "__main__":
    data_root = "./hist_matched_dataset"
    k_fold_train(
        data_root=data_root,
        k=5,
        num_epochs=10,
        batch_size=32,
        lr=1e-4
    )

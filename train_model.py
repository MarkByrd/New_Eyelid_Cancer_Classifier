from cross_validate_test import BinaryImageFolderDataset, train_one_epoch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms

def train_and_save_model(
    data_root,
    num_epochs=10,
    batch_size=32,
    lr=1e-4,
    save_path= None
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #Uses Nvidia GPU if you have one

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = BinaryImageFolderDataset(data_root, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = models.resnet152(pretrained=True) #The base network

    # Replace final layer for binary classification
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr) #No regularisation
    criterion = nn.CrossEntropyLoss()

    print("\n===== Training on entire dataset =====")
    for epoch in range(num_epochs):
        loss = train_one_epoch(model, data_loader, optimizer, criterion, device)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train_and_save_model(
        data_root="./hist_match_dataset", #Change this if you don't wont to override current model.
        num_epochs=10, #The number of times the model sees each datapoint
        batch_size=32, #The number of datapoints processed in paralell
        lr=1e-4,
        save_path = "./model/final_model_2.pth"
    )

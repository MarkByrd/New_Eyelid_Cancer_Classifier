import torch
import torch.nn as nn
from torchvision import models


def load_model(weights_path="model/final_model.pth", device="cpu"):
    # Correct model architecture
    model = models.resnet152(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)  # match training
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

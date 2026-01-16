import gradio as gr
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn as nn
from model import model

def load_model(weights_path="model/final_model.pth", device="cpu"):
    # Correct model architecture
    model = models.resnet152(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)  # match training
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model


# Load model
model = model.load_model()

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

classes = ["Benign", "Malignant"]

def predict(image):
    image = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
    classes = ["Benign", "Malignant"]
    return classes[pred.item()], f"{conf.item() * 100:.2f}% confidence"

# Gradio interface
title = "Neural Net Based Eyelid Cancer Detector"
description = "Upload an image of a eye lesion and get a prediction. Predictions based on statistical patterns. Mistakes can be made and creators are not liable."

interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Label(num_top_classes=1), gr.Textbox(label="Confidence")],
    title=title,
    description=description
)

if __name__ == "__main__":
    interface.launch()

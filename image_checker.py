import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os


# -----------------------------
# Configuration
# -----------------------------
DATASET_FOLDER_1 = "Dataset/benign"
DATASET_FOLDER_2 = "Dataset/malignant"   # folder containing dataset images
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Load Pretrained ResNet Model
# -----------------------------
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # remove final classifier
model.eval()
model.to(DEVICE)

# -----------------------------
# Image Transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -----------------------------
# Feature Extraction Function
# -----------------------------
def extract_features(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        features = model(image)
    
    features = features.cpu().numpy().flatten()
    return features

# -----------------------------
# Build Dataset Feature Index
# -----------------------------
print("Extracting dataset features...")
image_paths = []
image_features = []

for filename in os.listdir(DATASET_FOLDER_1):
    path = os.path.join(DATASET_FOLDER_1, filename)
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        image_paths.append(path)
        image_features.append(extract_features(path))

for filename in os.listdir(DATASET_FOLDER_2):
    path = os.path.join(DATASET_FOLDER_2, filename)
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        image_paths.append(path)
        image_features.append(extract_features(path))

image_features = np.array(image_features)

print("Feature extraction complete.")

# -----------------------------
# Find Nearest Image
# -----------------------------
def find_nearest_image(query_image_path):
    query_features = extract_features(query_image_path)
    
    similarities = cosine_similarity(
        [query_features], image_features
    )[0]
    
    best_index = np.argmax(similarities)
    return image_paths[best_index], similarities[best_index]

if __name__ == "__main__":
    threshold = .93
    matches = 0
    names = []
    copies = []
    scores = []
    for filename in os.listdir("Testing_dataset"):
        path = os.path.join("Testing_dataset", filename)
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            nearest_image, score = find_nearest_image(path)
            if score > threshold:
                matches +=1
                names.append(path)
                copies.append(nearest_image)
                scores.append(score)
    rows = len(names)
    cols = 2

    fig, axes = plt.subplots(rows, cols, figsize=(8, 4 * rows))

    # If only 1 row, axes won't be 2D — fix that
    if rows == 1:
        axes = [axes]

    for i in range(rows):
        # Left column
        img1 = mpimg.imread(names[i])
        axes[i][0].imshow(img1)
        axes[i][0].axis("off")
    
        # Right column
        img2 = mpimg.imread(copies[i])
        axes[i][1].imshow(img2)
        axes[i][1].axis("off")
        fig.text(
        0.02,                     # horizontal position (left margin)
        1 - (i + 0.5) / rows,   # vertical position (row center)
        scores[i],
        va='center'
        )
    plt.tight_layout()
    plt.show()
    print(f"There are {matches} duplicate images {names} with {scores}")
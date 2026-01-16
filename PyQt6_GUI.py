import sys
import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QLabel, 
                             QPushButton, QFrame, QFileDialog, QStackedWidget,
                             QScrollArea, QHBoxLayout)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap

# --- MODEL LOADING ---
def load_model(weights_path="model/final_model.pth", device="cpu"):
    model_arch = models.resnet152(weights=None)
    model_arch.fc = nn.Linear(model_arch.fc.in_features, 2)
    if os.path.exists(weights_path):
        model_arch.load_state_dict(torch.load(weights_path, map_location=device))
    model_arch.to(device)
    model_arch.eval()
    return model_arch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = load_model(device=DEVICE)
PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
CLASSES = ["Benign", "Malignant"]

# Style and Text Constants
STYLE_NORMAL = "border: 3px dashed #3498db; border-radius: 20px; background: #ffffff; color: #3498db; font-size: 18px;"
STYLE_HOVER = "border: 3px solid #2ecc71; background: #e8f8f5; border-radius: 20px; color: #2ecc71; font-size: 18px;"
TEXT_PROMPT = "\n\n<b>DRAG AND DROP IMAGES HERE</b><br><br><small>OR CLICK TO BROWSE FILES</small>"

# --- CLICKABLE LABEL COMPONENT ---
class ClickableDropZone(QLabel):
    clicked = pyqtSignal()

    def __init__(self, text):
        super().__init__(text)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(STYLE_NORMAL)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()

# --- RESULT ROW COMPONENT ---
class ResultRow(QFrame):
    def __init__(self, image_path, prediction, confidence):
        super().__init__()
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet("background-color: white; border-radius: 10px; border: 1px solid #ddd; margin-bottom: 5px;")
        
        layout = QHBoxLayout(self)
        img_label = QLabel()
        pixmap = QPixmap(image_path).scaled(120, 120, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        img_label.setPixmap(pixmap)
        img_label.setFixedWidth(120)
        
        info_layout = QVBoxLayout()
        name_label = QLabel(os.path.basename(image_path))
        name_label.setStyleSheet("font-weight: bold; border: none;")
        
        color = "#e74c3c" if prediction == "Malignant" else "#27ae60"
        pred_label = QLabel(f"Result: {prediction} ({confidence})")
        pred_label.setStyleSheet(f"color: {color}; font-size: 15px; font-weight: bold; border: none;")
        
        info_layout.addWidget(name_label)
        info_layout.addWidget(pred_label)
        
        layout.addWidget(img_label)
        layout.addLayout(info_layout)
        layout.addStretch()

# --- MAIN APP ---
class CancerDetectorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neural Net Cancer Detector")
        self.setMinimumSize(800, 700)
        self.setAcceptDrops(True)

        self.main_stack = QStackedWidget(self)
        self.setup_upload_page()
        self.setup_results_page()
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.addWidget(self.main_stack)

    def setup_upload_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        
        title = QLabel("Eyelid Cancer Detector")
        title.setStyleSheet("font-size: 26px; font-weight: bold; color: #2c3e50; margin-bottom: 5px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        sub_title = QLabel("Clinical Prediction Tool")
        sub_title.setStyleSheet("color: #7f8c8d; margin-bottom: 20px;")
        sub_title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Integrated Drop/Click Zone
        self.drop_zone = ClickableDropZone(TEXT_PROMPT)
        self.drop_zone.clicked.connect(self.open_file_dialog)
        self.drop_zone.setMinimumHeight(500)

        layout.addWidget(title)
        layout.addWidget(sub_title)
        layout.addWidget(self.drop_zone)
        layout.addStretch()
        
        self.main_stack.addWidget(page)

    def setup_results_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        
        nav_bar = QHBoxLayout()
        back_btn = QPushButton("← New Batch Analysis")
        back_btn.setFixedWidth(180)
        back_btn.setMinimumHeight(40)
        back_btn.setStyleSheet("background: #34495e; color: white; border-radius: 5px; font-weight: bold;")
        back_btn.clicked.connect(self.go_back_to_upload)
        
        header_label = QLabel("Analysis Results")
        header_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        
        nav_bar.addWidget(back_btn)
        nav_bar.addStretch()
        nav_bar.addWidget(header_label)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.scroll.setWidget(self.scroll_content)

        layout.addLayout(nav_bar)
        layout.addWidget(self.scroll)
        self.main_stack.addWidget(page)

    def go_back_to_upload(self):
        self.drop_zone.setText(TEXT_PROMPT)
        self.drop_zone.setStyleSheet(STYLE_NORMAL)
        self.main_stack.setCurrentIndex(0)

    def open_file_dialog(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select Images", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp)")
        if files:
            self.process_multiple_files(files)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
            self.drop_zone.setStyleSheet(STYLE_HOVER)
            self.drop_zone.setText("\n\nReady to analyze...")

    def dragLeaveEvent(self, event):
        self.drop_zone.setStyleSheet(STYLE_NORMAL)
        self.drop_zone.setText(TEXT_PROMPT)

    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        if files:
            self.process_multiple_files(files)

    def process_multiple_files(self, paths):
        while self.scroll_layout.count():
            item = self.scroll_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for path in paths:
            try:
                img = Image.open(path).convert("RGB")
                tensor_img = PREPROCESS(img).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    outputs = MODEL(tensor_img)
                    probs = torch.softmax(outputs, dim=1)
                    conf, pred = torch.max(probs, 1)
                row = ResultRow(path, CLASSES[pred.item()], f"{conf.item()*100:.2f}%")
                self.scroll_layout.addWidget(row)
            except Exception as e:
                print(f"Skipping {path}: {e}")

        self.main_stack.setCurrentIndex(1)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = CancerDetectorApp()
    window.show()
    sys.exit(app.exec())
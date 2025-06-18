import os
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import numpy as np

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset path
dataset_path = Path(r"C:\Users\HP\.cache\kagglehub\datasets\paultimothymooney\chest-xray-pneumonia\versions\2")
train_dir = dataset_path / "chest_xray" / "chest_xray" / "train"
val_dir = dataset_path / "chest_xray" / "chest_xray" / "val"
test_dir = dataset_path / "chest_xray" / "chest_xray" / "test"

if not train_dir.exists():
    train_dir = dataset_path / "chest_xray" / "train"
    val_dir = dataset_path / "chest_xray" / "val"
    test_dir = dataset_path / "chest_xray" / "test"

if not test_dir.exists():
    print(f"Test directory not found: {test_dir}")
    exit(1)

# Define transforms
IMG_SIZE = (224, 224)
BATCH_SIZE = 32  # Match training script

test_transforms = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create test dataset and loader
test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print("Class indices:", test_dataset.class_to_idx)
print(f"Test samples: {len(test_dataset)}")

# Define Custom CNN
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Evaluation function
def evaluate_model(model, loader, name):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            predicted = (outputs >= 0.5).float().cpu().numpy()
            all_preds.extend(predicted.flatten())
            all_labels.extend(labels.numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    cm = confusion_matrix(all_labels, all_preds)  # Fixed: all_preds instead of all_labels
    
    print(f"\n{name} Evaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Plot confusion matrix
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['NORMAL', 'PNEUMONIA'], yticklabels=['NORMAL', 'PNEUMONIA'])
    plt.title(f'{name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f"F:/PneumoniaDetection/{name}_confusion_matrix.png")
    plt.close()
    
    return accuracy, precision, recall, f1, cm

# Load Custom CNN
custom_model = CustomCNN().to(device)
custom_model_path = "F:/PneumoniaDetection/CustomCNN_best.pth"
if os.path.exists(custom_model_path):
    custom_model.load_state_dict(torch.load(custom_model_path, map_location=device))
    custom_model.eval()
    print("Loaded CustomCNN weights")
else:
    print(f"CustomCNN weights not found at {custom_model_path}")
    exit(1)

# Evaluate Custom CNN
custom_metrics = evaluate_model(custom_model, test_loader, "CustomCNN")
custom_accuracy, custom_precision, custom_recall, custom_f1, custom_cm = custom_metrics

# Plot Custom CNN training curves
custom_history = {
    'train_loss': [0.8551, 0.5818, 0.4693, 0.4546, 0.4120, 0.3889, 0.3831, 0.3657, 0.4117, 0.3700],
    'val_loss': [1.8033, 2.6310, 2.1572, 2.9736, 3.9744, 1.4583, 1.6363, 2.5588, 1.1203, 2.0090],
    'train_acc': [0.6898, 0.8232, 0.8666, 0.8673, 0.8804, 0.8940, 0.8944, 0.9009, 0.8855, 0.8932],
    'val_acc': [0.6250, 0.5625, 0.5000, 0.6250, 0.5625, 0.6875, 0.6250, 0.6250, 0.6875, 0.5625]
}
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(custom_history['train_loss'], label='Train Loss')
plt.plot(custom_history['val_loss'], label='Val Loss')
plt.title('CustomCNN Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(custom_history['train_acc'], label='Train Acc')
plt.plot(custom_history['val_acc'], label='Val Acc')
plt.title('CustomCNN Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("F:/PneumoniaDetection/CustomCNN_training_curves.png")
plt.close()

# Load ResNet18
resnet_model = models.resnet18(weights=None)
resnet_model.fc = nn.Sequential(
    nn.Linear(resnet_model.fc.in_features, 1),
    nn.Sigmoid()
)
resnet_model = resnet_model.to(device)
resnet_model_path = "F:/PneumoniaDetection/ResNet18FineTuned_best.pth"
if os.path.exists(resnet_model_path):
    resnet_model.load_state_dict(torch.load(resnet_model_path, map_location=device))
    resnet_model.eval()
    print("Loaded ResNet18FineTuned weights")
else:
    print(f"ResNet18FineTuned weights not found at {resnet_model_path}")
    exit(1)

# Evaluate ResNet18
resnet_metrics = evaluate_model(resnet_model, test_loader, "ResNet18FineTuned")
resnet_accuracy, resnet_precision, resnet_recall, resnet_f1, resnet_cm = resnet_metrics

# Plot ResNet18 training curves
resnet_history = {
    'train_loss': [0.6501, 0.4379, 0.3886, 0.3742, 0.3636, 0.2626],  # Initial 5 + Fine-tune E1
    'val_loss': [0.8188, 0.6470, 0.7019, 0.7980, 0.7013, 1.2675],
    'train_acc': [0.8194, 0.8871, 0.8961, 0.8974, 0.9032, 0.9331],
    'val_acc': [0.9375, 0.9375, 0.8750, 0.8750, 0.8750, 0.8750]
}
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(resnet_history['train_loss'], label='Train Loss')
plt.plot(resnet_history['val_loss'], label='Val Loss')
plt.title('ResNet18FineTuned Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(resnet_history['train_acc'], label='Train Acc')
plt.plot(resnet_history['val_acc'], label='Val Acc')
plt.title('ResNet18FineTuned Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("F:/PneumoniaDetection/ResNet18FineTuned_training_curves.png")
plt.close()

# Print model comparison
print("\nModel Comparison:")
print(f"CustomCNN: Acc={custom_accuracy:.4f}, F1={custom_f1:.4f}")
print(f"ResNet18FineTuned: Acc={resnet_accuracy:.4f}, F1={resnet_f1:.4f}")
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import kagglehub
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Set device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset paths
try:
    kagglehub.login()
    path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
    print("Path to dataset files:", path)
except Exception as e:
    print(f"Error with kagglehub: {e}")
    print("Using manual path: F:\\PneumoniaDetection\\chest_xray")
    path = "F:/PneumoniaDetection/chest_xray"

dataset_path = Path(path)
train_dir = dataset_path / "chest_xray" / "chest_xray" / "train"
val_dir = dataset_path / "chest_xray" / "chest_xray" / "val"
test_dir = dataset_path / "chest_xray" / "chest_xray" / "test"

if not train_dir.exists():
    print(f"Nested training directory not found: {train_dir}")
    train_dir = dataset_path / "chest_xray" / "train"
    val_dir = dataset_path / "chest_xray" / "val"
    test_dir = dataset_path / "chest_xray" / "test"

if not train_dir.exists():
    print(f"Training directory not found: {train_dir}")
    print("Listing dataset path contents:", os.listdir(dataset_path))
    exit(1)

# Define transforms
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_transforms = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transforms = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create datasets and loaders
train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
val_dataset = datasets.ImageFolder(val_dir, transform=val_test_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform=val_test_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("Class indices:", train_dataset.class_to_idx)
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# Define class weights for imbalance
normal_count, pneumonia_count = 1341, 3875
weight_for_normal = pneumonia_count / normal_count  # ~2.89
weight_for_pneumonia = 1.0
class_weights = torch.tensor([weight_for_normal, weight_for_pneumonia]).to(device)

# Define custom CNN
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

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, name):
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0
    best_model_path = f"{name}_best.pth"

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).float()
            # Compute per-sample weights
            weights = torch.where(labels == 0, class_weights[0], class_weights[1]).to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels)
            weighted_loss = (loss * weights).mean()
            weighted_loss.backward()
            optimizer.step()
            train_loss += weighted_loss.item() * images.size(0)
            predicted = (outputs >= 0.5).float()
            train_correct += (predicted.squeeze() == labels).sum().item()
            train_total += labels.size(0)
        
        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validation phase
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).float()
                weights = torch.where(labels == 0, class_weights[0], class_weights[1]).to(device)
                outputs = model(images)
                loss = criterion(outputs.squeeze(), labels)
                weighted_loss = (loss * weights).mean()
                val_loss += weighted_loss.item() * images.size(0)
                predicted = (outputs >= 0.5).float()
                val_correct += (predicted.squeeze() == labels).sum().item()
                val_total += labels.size(0)
        
        val_loss /= val_total
        val_acc = val_correct / val_total

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        print(f"{name} Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    return history, best_model_path

# Evaluation function
def evaluate_model(model, test_loader, name):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            predicted = (outputs >= 0.5).float().cpu().numpy()
            all_preds.extend(predicted.flatten())
            all_labels.extend(labels.numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    print(f"\n{name} Test Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    return accuracy, precision, recall, f1

# Plot training curves
def plot_training_curves(history, name):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title(f'{name} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title(f'{name} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()

# 1. Train Custom CNN
custom_model = CustomCNN().to(device)
criterion = nn.BCELoss(reduction='none')  # Compute loss per sample
optimizer = optim.Adam(custom_model.parameters(), lr=0.001)
custom_history, custom_model_path = train_model(custom_model, train_loader, val_loader, criterion, optimizer, num_epochs=10, name="CustomCNN")

# Evaluate Custom CNN
custom_metrics = evaluate_model(custom_model, test_loader, "CustomCNN")

# 2. Train ResNet18 (Transfer Learning)
resnet_model = models.resnet18(pretrained=True) #lading pre trained model
resnet_model.fc = nn.Sequential(
    nn.Linear(resnet_model.fc.in_features, 1),
    nn.Sigmoid()
)
resnet_model = resnet_model.to(device)

# Freeze all layers except the final layer
for param in resnet_model.parameters():
    param.requires_grad = False
for param in resnet_model.fc.parameters():
    param.requires_grad = True

criterion = nn.BCELoss(reduction='none')
optimizer = optim.Adam(resnet_model.fc.parameters(), lr=0.001)
resnet_history, resnet_model_path = train_model(resnet_model, train_loader, val_loader, criterion, optimizer, num_epochs=5, name="ResNet18")

# Fine-tune ResNet18 (unfreeze layer4)
for param in resnet_model.layer4.parameters():
    param.requires_grad = True
optimizer = optim.Adam([
    {'params': resnet_model.layer4.parameters(), 'lr': 0.0001},
    {'params': resnet_model.fc.parameters(), 'lr': 0.001}
])
resnet_finetune_history, resnet_finetune_path = train_model(resnet_model, train_loader, val_loader, criterion, optimizer, num_epochs=5, name="ResNet18FineTuned")

# Evaluate ResNet18
resnet_metrics = evaluate_model(resnet_model, test_loader, "ResNet18FineTuned")

# Plot training curves
plot_training_curves(custom_history, "CustomCNN")
plot_training_curves(resnet_history, "ResNet18")
plot_training_curves(resnet_finetune_history, "ResNet18FineTuned")

# Compare models
print("\nModel Comparison:")
print(f"CustomCNN: Acc={custom_metrics[0]:.4f}, F1={custom_metrics[3]:.4f}")
print(f"ResNet18FineTuned: Acc={resnet_metrics[0]:.4f}, F1={resnet_metrics[3]:.4f}")
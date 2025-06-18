import os
from pathlib import Path
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import kagglehub

# Authenticate with Kaggle (if not already authenticated)
try:
    kagglehub.login()
except Exception as e:
    print(f"Error logging into Kaggle: {e}")
    print("Falling back to manual path: F:\\PneumoniaDetection\\chest_xray")
    path = "F:/PneumoniaDetection/chest_xray"
else:
    # Download the dataset
    try:
        path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
        print("Path to dataset files:", path)
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Falling back to manual path: F:\\PneumoniaDetection\\chest_xray")
        path = "F:/PneumoniaDetection/chest_xray"

# Path to the dataset
dataset_path = Path(path)

# Define paths to training, validation, and test sets
train_dir = dataset_path / "chest_xray" / "chest_xray" / "train"
val_dir = dataset_path / "chest_xray" / "chest_xray" / "val"
test_dir = dataset_path / "chest_xray" / "chest_xray" / "test"

# Fallback to non-nested structure
if not train_dir.exists():
    print(f"Nested training directory not found: {train_dir}")
    train_dir = dataset_path / "chest_xray" / "train"
    val_dir = dataset_path / "chest_xray" / "val"
    test_dir = dataset_path / "chest_xray" / "test"

# Verify directories exist
if not train_dir.exists():
    print(f"Training directory not found: {train_dir}")
    print("Listing dataset path contents:", os.listdir(dataset_path))
    print("Ensure dataset is downloaded and unzipped correctly.")
    exit(1)

# Define image size and batch size
IMG_SIZE = (224, 224)  # Standard size for CNNs like ResNet, MobileNet
BATCH_SIZE = 32

# Define transforms for training (with augmentation)
train_transforms = transforms.Compose([
    transforms.Resize(IMG_SIZE),  # Resize to 224x224
    transforms.RandomRotation(20),  # Random rotation up to 20 degrees
    transforms.RandomHorizontalFlip(),  # Random horizontal flip
    transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),  # Random shift up to 20%
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),  # Random zoom
    transforms.ToTensor(),  # Convert to tensor (normalizes to [0, 1])
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

# Define transforms for validation and test (no augmentation)
val_test_transforms = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create datasets
train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
val_dataset = datasets.ImageFolder(val_dir, transform=val_test_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform=val_test_transforms)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Print class indices and dataset sizes
print("Class indices:", train_dataset.class_to_idx)
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# Visualize a few augmented training images
def visualize_augmented_images(loader, num_samples=3):
    data_iter = iter(loader)
    images, labels = next(data_iter)
    class_names = {v: k for k, v in loader.dataset.class_to_idx.items()}
    
    plt.figure(figsize=(15, 5))
    for i in range(min(num_samples, len(images))):
        img = images[i].permute(1, 2, 0).numpy()  # Convert from (C, H, W) to (H, W, C)
        img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]  # Denormalize
        img = img.clip(0, 1)  # Ensure values in [0, 1]
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(img)
        plt.title(f"Class: {class_names[labels[i].item()]}")
        plt.axis("off")
    plt.show()

print("Visualizing augmented training images:")
visualize_augmented_images(train_loader)
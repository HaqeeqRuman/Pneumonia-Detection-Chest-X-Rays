import os
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import kagglehub

# Authenticate with Kaggle (if not already authenticated)
try:
    kagglehub.login()
except Exception as e:
    print(f"Error logging into Kaggle: {e}")
    print("Ensure you have a Kaggle API token set up. See instructions below.")
    exit(1)

# Download the dataset
try:
    path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
    print("Path to dataset files:", path)
except Exception as e:
    print(f"Error downloading dataset: {e}")
    exit(1)

# Path to the dataset
dataset_path = Path(path)  # Use the path from kagglehub.dataset_download

# Define paths to training, validation, and test sets
train_dir = dataset_path / "chest_xray" / "train"
val_dir = dataset_path / "chest_xray" / "val"
test_dir = dataset_path / "chest_xray" / "test"

# Verify directories exist
if not train_dir.exists():
    print(f"Training directory not found: {train_dir}")
    print("Listing dataset path contents:", os.listdir(dataset_path))
    exit(1)

# Function to load and display sample images
def visualize_samples(data_dir, class_name, num_samples=3):
    class_path = data_dir / class_name
    image_files = list(class_path.glob("*.jpeg"))[:num_samples]  # Get first few images
    
    if not image_files:
        print(f"No images found in {class_path}")
        return
    
    plt.figure(figsize=(15, 5))
    for i, img_path in enumerate(image_files, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Failed to load image: {img_path}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        plt.subplot(1, num_samples, i)
        plt.imshow(img)
        plt.title(f"{class_name} Sample {i}")
        plt.axis("off")
    plt.show()

# Visualize samples from NORMAL and PNEUMONIA classes in the training set
print("Visualizing NORMAL samples:")
visualize_samples(train_dir, "NORMAL")
print("Visualizing PNEUMONIA samples:")
visualize_samples(train_dir, "PNEUMONIA")
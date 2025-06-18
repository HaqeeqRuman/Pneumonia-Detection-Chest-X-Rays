import os
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import seaborn as sns
import pandas as pd
import kagglehub

# Authenticate with Kaggle (if not already authenticated)
try:
    kagglehub.login()
except Exception as e:
    print(f"Error logging into Kaggle: {e}")
    print("Ensure you have a Kaggle API token set up.")
    print("1. Go to Kaggle > Account > Create New API Token to download kaggle.json")
    print("2. Place kaggle.json in C:\\Users\\<YourUsername>\\.kaggle\\")
    print("3. Or set environment variables: KAGGLE_USERNAME and KAGGLE_KEY")
    print("Falling back to manual path: F:\\PneumoniaDetection\\chest_xray")
    path = "F:/PneumoniaDetection/chest_xray"  # Manual fallback path
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
# Try nested structure first (kagglehub often nests chest_xray/chest_xray)
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

# Function to load and display sample images
def visualize_samples(data_dir, class_name, num_samples=3):
    class_path = data_dir / class_name
    image_files = list(class_path.glob("*.[jJ][pP][eE][gG]"))[:num_samples]  # Case-insensitive
    
    if not image_files:
        print(f"No images found in {class_path}")
        print(f"Contents of {class_path}:", os.listdir(class_path))
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

# Function to count images in each class
def analyze_class_distribution(data_dir, set_name):
    normal_count = len(list((data_dir / "NORMAL").glob("*.[jJ][pP][eE][gG]")))
    pneumonia_count = len(list((data_dir / "PNEUMONIA").glob("*.[jJ][pP][eE][gG]")))
    return {"Set": set_name, "NORMAL": normal_count, "PNEUMONIA": pneumonia_count}

# Collect distribution data
dist_data = [
    analyze_class_distribution(train_dir, "Training"),
    analyze_class_distribution(val_dir, "Validation"),
    analyze_class_distribution(test_dir, "Testing")
]

# Convert to DataFrame
dist_df = pd.DataFrame(dist_data)

# Print the distribution
print("\nClass Distribution:")
print(dist_df)

# Plot the distribution
plt.figure(figsize=(10, 6))
dist_df_melted = dist_df.melt(id_vars="Set", value_vars=["NORMAL", "PNEUMONIA"], 
                              var_name="Class", value_name="Count")
sns.barplot(x="Set", y="Count", hue="Class", data=dist_df_melted)
plt.title("Class Distribution Across Dataset Splits")
plt.show()
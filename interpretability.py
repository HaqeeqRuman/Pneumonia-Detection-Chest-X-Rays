import os
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset path
dataset_path = Path(r"C:\Users\HP\.cache\kagglehub\datasets\paultimothymooney\chest-xray-pneumonia\versions\2")
test_dir = dataset_path / "chest_xray" / "chest_xray" / "test"

if not test_dir.exists():
    test_dir = dataset_path / "chest_xray" / "test"

if not test_dir.exists():
    print(f"Test directory not found: {test_dir}")
    exit(1)

# Define transforms
IMG_SIZE = (224, 224)
BATCH_SIZE = 8

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
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(inplace=False),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Grad-CAM implementation
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self.hook_handles.append(self.target_layer.register_forward_hook(self.save_activations))
        self.hook_handles.append(self.target_layer.register_full_backward_hook(self.save_gradients))
    
    def save_activations(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
    
    def __call__(self, x, class_idx=None):
        self.model.eval()
        self.gradients = None
        self.activations = None
        
        # Forward pass
        output = self.model(x)
        if class_idx is None:
            class_idx = (output >= 0.5).float().item()
        
        # Backward pass
        self.model.zero_grad()
        output.backward(gradient=torch.ones_like(output), retain_graph=True)
        
        if self.gradients is None or self.activations is None:
            raise ValueError("Gradients or activations not captured. Check target layer.")
        
        # Debug shapes
        print(f"Activations shape: {self.activations.shape}")
        print(f"Gradients shape: {self.gradients.shape}")
        
        # Compute CAM
        pooled_grads = torch.mean(self.gradients, dim=[0, 2, 3], keepdim=True)
        cam = self.activations * pooled_grads
        cam = cam.sum(dim=1, keepdim=True)  # Sum over channels
        cam = torch.relu(cam)
        cam = cam / (torch.max(cam) + 1e-8)  # Normalize
        cam = cam.squeeze().cpu().numpy()  # (H, W)
        
        return cam

# Visualize Grad-CAM
def visualize_gradcam(model, loader, name, target_layer, num_samples=3):
    model.eval()
    data_iter = iter(loader)
    images, labels = next(data_iter)
    images, labels = images[:num_samples].to(device), labels[:num_samples]
    class_names = {0: 'NORMAL', 1: 'PNEUMONIA'}
    
    grad_cam = GradCAM(model, target_layer)
    
    plt.figure(figsize=(15, 5))
    for i in range(num_samples):
        img = images[i].unsqueeze(0)
        try:
            cam = grad_cam(img)
            print(f"CAM for image {i+1} min: {cam.min()}, max: {cam.max()}")
        except ValueError as e:
            print(f"Error computing Grad-CAM for image {i+1}: {e}")
            grad_cam.remove_hooks()
            return
        
        # Denormalize image for visualization
        img_np = images[i].cpu().numpy()
        img_np = img_np * np.array([0.229, 0.224, 0.225])[:, None, None] + np.array([0.485, 0.456, 0.406])[:, None, None]
        img_np = np.transpose(img_np, (1, 2, 0))  # (C, H, W) -> (H, W, C)
        img_np = img_np.clip(0, 1)
        
        # Resize CAM to match image size
        cam_resized = zoom(cam, (IMG_SIZE[0] / cam.shape[-2], IMG_SIZE[1] / cam.shape[-1]))
        
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(img_np)
        plt.imshow(cam_resized, cmap='jet', alpha=0.5)
        plt.title(f"{class_names[labels[i].item()]}")
        plt.axis('off')
    
    # Remove hooks after all images
    grad_cam.remove_hooks()
    plt.tight_layout()
    plt.savefig(f"F:/PneumoniaDetection/{name}_gradcam.png")
    plt.close()

# Load Custom CNN
custom_model = CustomCNN().to(device)
custom_model_path = "F:/PneumoniaDetection/CustomCNN_best.pth"
if os.path.exists(custom_model_path):
    try:
        custom_model.load_state_dict(torch.load(custom_model_path, map_location=device))
        custom_model.eval()
        print("Loaded CustomCNN weights")
        # Verify ReLU inplace settings
        for module in custom_model.modules():
            if isinstance(module, nn.ReLU):
                if module.inplace:
                    print("Warning: Found ReLU with inplace=True in CustomCNN")
    except Exception as e:
        print(f"Error loading CustomCNN weights: {e}")
        exit(1)
else:
    print(f"CustomCNN weights not found at {custom_model_path}")
    exit(1)

# Visualize Grad-CAM for CustomCNN
visualize_gradcam(custom_model, test_loader, "CustomCNN", custom_model.features[-3])  # Third conv layer

# Load ResNet18
resnet_model = models.resnet18(weights=None)
resnet_model.fc = nn.Sequential(
    nn.Linear(resnet_model.fc.in_features, 1),
    nn.Sigmoid()
)
resnet_model = resnet_model.to(device)
resnet_model_path = "F:/PneumoniaDetection/ResNet18FineTuned_best.pth"
if os.path.exists(resnet_model_path):
    try:
        resnet_model.load_state_dict(torch.load(resnet_model_path, map_location=device))
        # Disable inplace ReLU
        for module in resnet_model.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False
        resnet_model.eval()
        print("Loaded ResNet18FineTuned weights")
    except Exception as e:
        print(f"Error loading ResNet18FineTuned weights: {e}")
        exit(1)
else:
    print(f"ResNet18FineTuned weights not found at {resnet_model_path}")
    exit(1)

# Visualize Grad-CAM for ResNet18
visualize_gradcam(resnet_model, test_loader, "ResNet18FineTuned", resnet_model.layer4[-1].conv2)  # Last conv layer in layer4
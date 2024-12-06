import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

##############################
# Dataset and Preprocessing
##############################

# We define a custom transform for the target segmentation masks
# to merge classes 1 (border) and 2 (pet) into a single 'pet' class = 1.
def target_transform(mask):
    mask = transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST)(mask)
    mask = torch.tensor(np.array(mask), dtype=torch.long)
    mask[mask == 3] = 2
    mask = mask-1
    return mask

# Debugging: Check unique values in the target mask
def debug_target_transform(mask):
    mask = transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST)(mask)
    mask = torch.tensor(np.array(mask), dtype=torch.long)
    unique_values = torch.unique(mask)
    mask[mask == 3] = 2
    mask = mask-1
    print(f"Unique values in target mask: {unique_values}")
    return mask

# We'll also define image transforms.
# Typically, you might add data augmentation here.
img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load training split of Oxford-IIIT Pet with segmentation masks
train_dataset = torchvision.datasets.OxfordIIITPet(
    root="oxford_pet_dataset",
    split="trainval",
    download=True,
    target_types="segmentation",
    transform=img_transform,
    target_transform=target_transform
)

# For simplicity, use a subset or the full dataset
# Optionally, you can create a validation split
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)

# Load test dataset without transforms
test_dataset = torchvision.datasets.OxfordIIITPet(
    root="oxford_pet_dataset",
    split="test",
    download=True,
    target_types="segmentation",
    transform=None,  # No transform here
    target_transform=None  # No transform here
)

def process_image_for_model(image, device='cuda'):
    """
    Process a PIL image for model input.

    Args:
        image (PIL.Image): Original image
        device (str): Device to put tensor on

    Returns:
        torch.Tensor: Processed image tensor (1, C, H, W)
    """
    # Apply transforms for model input
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0).to(device)

def visualize_predictions(model, dataset, num_samples=5, device='cuda'):
    """
    Visualize model predictions with IoU scores using original image sizes.

    Args:
        model (nn.Module): The trained model
        dataset (Dataset): Dataset to sample from (without transforms)
        num_samples (int): Number of samples to visualize
        device (str): Device to run inference on
    """
    model.eval()
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))

    with torch.no_grad():
        for idx in range(num_samples):
            # Get original image and mask
            img, gt = dataset[idx]
            original_size = img.size[::-1]  # (W,H) -> (H,W)

            # Process image for model
            img_tensor = process_image_for_model(img, device)

            # Get prediction and resize back to original size
            pred = model(img_tensor)['out'].argmax(dim=1).cpu().squeeze()
            pred_resized = transforms.Resize(original_size,
                                          interpolation=transforms.InterpolationMode.NEAREST)(
                pred.unsqueeze(0).float()
            ).squeeze().bool()
            pred_resized = ~pred_resized

            # Convert ground truth mask to tensor and boolean
            gt_tensor = torch.tensor(np.array(gt))
            gt_tensor[gt_tensor == 3] = 2  # Adjust classes as before
            gt_bool = gt_tensor==1

            # Calculate IoU with original size masks
            iou = calculate_iou(pred_resized, gt_bool)

            # Plot
            axes[idx, 0].imshow(img)
            axes[idx, 0].set_title("Original Image")
            axes[idx, 0].axis('off')

            axes[idx, 1].imshow(gt_bool.numpy(), cmap='gray')
            axes[idx, 1].set_title("Ground Truth")
            axes[idx, 1].axis('off')

            axes[idx, 2].imshow(pred_resized.numpy(), cmap='gray')
            axes[idx, 2].set_title(f"Prediction (IoU: {iou:.3f})")
            axes[idx, 2].axis('off')

    plt.tight_layout()
    plt.savefig("deeplab_segmentation_results.png")
    plt.close()

@torch.no_grad()
def evaluate_model(model, dataset, device='cuda', num_samples=100):
    """
    Evaluate model on dataset using IoU metric.

    Args:
        model (nn.Module): The trained model
        dataset (Dataset): Dataset without transforms
        device (str): Device to run evaluation on

    Returns:
        float: Average IoU score
    """
    model.eval()
    total_iou = 0

    for idx in tqdm(range(num_samples), desc="Evaluating"):
        # Get original image and mask
        img, gt = dataset[idx]
        original_size = img.size[::-1]  # (W,H) -> (H,W)

        # Process image for model
        img_tensor = process_image_for_model(img, device)

        # Get prediction and resize back to original size
        pred = model(img_tensor)['out'].argmax(dim=1).cpu().squeeze()
        pred_resized = transforms.Resize(original_size,
                                      interpolation=transforms.InterpolationMode.NEAREST)(
            pred.unsqueeze(0).float()
        ).squeeze().bool()
        pred_resized = ~pred_resized
        # Process ground truth
        gt_tensor = torch.tensor(np.array(gt))
        gt_bool = gt_tensor==1

        # Calculate IoU
        iou = calculate_iou(pred_resized, gt_bool)
        total_iou += iou

        if idx % 10 == 0:
            print(f"Processed {idx+1}/{len(dataset)} images. Current avg IoU: {total_iou/(idx+1):.3f}")

    return total_iou / num_samples

##############################
# Model Setup
##############################

# Load a pretrained DeepLabv3-ResNet50 model
model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)

# We have 2 classes now: background (0) and pet (1)
# The classifier is a DeepLabHead, which ends in a Conv2d with output channels = 21 (for COCO)
# We replace it with a Conv2d layer that outputs 2 classes.
model.classifier[4] = nn.Conv2d(256, 2, kernel_size=1)
model.to('cuda')

# Set model to training mode
model.train()

##############################
# Loss and Optimizer
##############################

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

##############################
# Evaluation Functions
##############################

def calculate_iou(pred, target):
    """
    Calculate Intersection over Union (IoU) between prediction and target masks.

    Args:
        pred (torch.Tensor): Predicted binary mask
        target (torch.Tensor): Target binary mask

    Returns:
        float: IoU score
    """
    intersection = torch.logical_and(pred, target).sum()
    union = torch.logical_or(pred, target).sum()
    return (intersection / union).item()

##############################
# Training Loop Updates
##############################

num_epochs = 2  # Increase this for real training
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, targets in tqdm(train_loader):
        images = images.to('cuda')
        targets = targets.to('cuda')

        optimizer.zero_grad()
        outputs = model(images)['out']

        # outputs shape: [batch, 2, H, W]
        # targets shape: [batch, H, W] with values in {0,1}
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {running_loss/len(train_loader):.4f}")

print("Training finished.")

# After training, evaluate and visualize
average_iou = evaluate_model(model, test_dataset)
print(f"Average IoU on test set: {average_iou:.3f}")

# Visualize predictions
visualize_predictions(model, test_dataset, num_samples=5)

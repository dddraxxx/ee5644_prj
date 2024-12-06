import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from itertools import product
from functools import lru_cache
import argparse
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
import torch.optim as optim
import torchvision.transforms as transforms

def target_transform(mask):
    mask = transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST)(mask)
    mask = torch.tensor(np.array(mask), dtype=torch.long)
    mask = (mask==1).long()
    # Now we have {0:bg, 1:pet} which is what we want.
    return mask

img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
def parse_args():
    parser = argparse.ArgumentParser(description='Hyperparameter search for pet segmentation')
    parser.add_argument('--num-samples', '-n', type=int, default=100,
                       help='Number of test samples to evaluate (default: 100)')
    parser.add_argument('--batch-size', '-b', type=int, default=4,
                       help='Batch size for evaluation (default: 4)')
    return parser.parse_args()

@lru_cache(maxsize=None)
def get_test_dataset():
    """Get the test dataset with original image sizes (cached to avoid reloading)"""
    dataset = torchvision.datasets.OxfordIIITPet(
        root="oxford_pet_dataset",
        split="test",
        download=True,
        target_types="segmentation",
        transform=None,  # No transform initially to get original size
        target_transform=None
    )

    # Create a wrapper to store original sizes and apply transforms
    class DatasetWrapper:
        def __init__(self, dataset):
            self.dataset = dataset

        def __getitem__(self, idx):
            img, mask = self.dataset[idx]
            original_size = img.size  # Store original size

            # Apply transforms
            img = img_transform(img)
            mask = target_transform(mask)

            return img, mask, original_size

        def __len__(self):
            return len(self.dataset)

    return DatasetWrapper(dataset)

def calculate_iou(pred, target):
    """
    Calculate IoU (Intersection over Union) for the pet class (label 1)

    Args:
        pred: Binary prediction tensor
        target: Binary ground truth tensor

    Returns:
        float: IoU score for the pet class
    """
    intersection = torch.logical_and(pred, target).sum()
    union = torch.logical_or(pred, target).sum()
    return (intersection / union).item() if union > 0 else 0.0

def evaluate_model(model, dataset, device, num_samples):
    """
    Evaluate model on test set using IoU metric

    Args:
        model: FCN model
        dataset: Test dataset
        device: torch device
        num_samples: Number of samples to evaluate

    Returns:
        float: Average IoU score
    """
    model.eval()
    ious = []

    with torch.no_grad():
        for i in tqdm(range(min(num_samples, len(dataset))), desc="Evaluating"):
            # Load single image and target (ignore original_size for evaluation)
            image, target, _ = dataset[i]
            image = image.unsqueeze(0).to(device)  # Add batch dimension
            target = target.to(device)

            # Get prediction
            output = model(image)['out']
            pred = output.argmax(dim=1).squeeze()  # Remove batch dimension

            # Calculate IoU
            iou = calculate_iou(pred == 1, target == 1)
            ious.append(iou)

    return np.mean(ious)

def visualize_predictions(model, dataset, device, num_samples=3, params_str=''):
    """
    Visualize model predictions and save them with IoU scores.
    Predictions are resized back to original image dimensions.
    """
    model.eval()
    plt.figure(figsize=(5, num_samples * 5))

    with torch.no_grad():
        for i in range(num_samples):
            # Load image, ground truth, and original size
            image, gt, original_size = dataset[i]
            image = image.unsqueeze(0).to(device)

            # Get prediction
            output = model(image)['out']
            pred = output.argmax(dim=1).cpu().squeeze()

            # Resize prediction back to original dimensions
            pred = transforms.ToPILImage()(pred.unsqueeze(0).float())
            pred = transforms.Resize(original_size[::-1], interpolation=transforms.InterpolationMode.NEAREST)(pred)
            pred = torch.tensor(np.array(pred), dtype=torch.bool)

            # Resize ground truth back to original dimensions
            gt = transforms.ToPILImage()(gt.unsqueeze(0).float())
            gt = transforms.Resize(original_size[::-1], interpolation=transforms.InterpolationMode.NEAREST)(gt)
            gt = torch.tensor(np.array(gt), dtype=torch.bool)

            # Calculate IoU
            iou = calculate_iou(pred, gt)

            # Plot prediction
            plt.subplot(num_samples, 1, i + 1)
            plt.imshow(pred.numpy(), cmap='gray')
            plt.axis('off')
            plt.title(f"Model Prediction (Original Size: {original_size})\nIoU: {iou:.3f}")

    plt.tight_layout()
    # Create filename from hyperparameters
    os.makedirs("./fcn_scratch", exist_ok=True)
    plt.savefig(f"./fcn_scratch/segmentation_predictions_{params_str}.png")
    plt.close()

def train_model(model, dataset, device, params, num_epochs=2):
    """
    Train the model with given hyperparameters

    Args:
        model: FCN model
        dataset: Training dataset
        device: torch device
        params: dict containing 'learning_rate', 'momentum', 'batch_size'
        num_epochs: Number of training epochs
    """
    # Create data loader for training
    train_loader = DataLoader(dataset,
                            batch_size=params['batch_size'],
                            shuffle=True,
                            num_workers=16)

    # Setup optimizer and criterion
    optimizer = optim.SGD(model.parameters(),
                         lr=params['learning_rate'],
                         momentum=params['momentum'])
    criterion = nn.CrossEntropyLoss()

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)['out']
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f}")

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define hyperparameter grid
    param_grid = {
        'learning_rate': [0.1, 0.01, 0.001],
        'momentum': [0.9, 0.95],
        'batch_size': [4, 8]
    }
    param_grid = {
        'learning_rate': [0.01],
        'momentum': [0.9],
        'batch_size': [4]
    }

    # Get datasets
    train_dataset = torchvision.datasets.OxfordIIITPet(
        root="oxford_pet_dataset",
        split="trainval",
        download=True,
        target_types="segmentation",
        transform=img_transform,
        target_transform=target_transform
    )
    test_dataset = get_test_dataset()

    # Store results
    results = []

    # Try all combinations
    param_combinations = product(*param_grid.values())
    for lr, momentum, batch_size in param_combinations:
        print(f"Training with lr={lr}, momentum={momentum}, batch_size={batch_size}")
        params = {
            'learning_rate': lr,
            'momentum': momentum,
            'batch_size': batch_size
        }

        # Initialize model
        model = torchvision.models.segmentation.fcn_resnet50(weights=None, weights_backbone=None)
        model.classifier[4] = nn.Conv2d(512, 2, kernel_size=1)
        model.to(device)

        # Create parameter string for filename
        params_str = f"lr{lr}_m{momentum}_b{batch_size}"

        # Train model
        train_model(model, train_dataset, device, params)

        # Evaluate
        iou_score = evaluate_model(model, test_dataset, device, args.num_samples)

        # Visualize predictions
        visualize_predictions(
            model,
            test_dataset,
            device,
            num_samples=5,
            params_str=params_str
        )

        # Store results
        results.append({
            'learning_rate': lr,
            'momentum': momentum,
            'batch_size': batch_size,
            'iou_score': iou_score
        })

        print(f"\nResults for lr={lr}, momentum={momentum}, batch_size={batch_size}:")
        print(f"IoU Score: {iou_score:.4f}")
        print(f"Visualization saved as: segmentation_predictions_{params_str}.png")
        del model  # Free up GPU memory

    # Find best parameters
    best_result = max(results, key=lambda x: x['iou_score'])
    print("\nBest hyperparameters:")
    print(f"Learning Rate: {best_result['learning_rate']}")
    print(f"Momentum: {best_result['momentum']}")
    print(f"Batch Size: {best_result['batch_size']}")
    print(f"IoU Score: {best_result['iou_score']:.4f}")

    # save as csv
    df = pd.DataFrame(results)
    df.to_csv("./fcn_scratch/hyper_search_results.csv", index=False)

if __name__ == "__main__":
    args = parse_args()
    main(args)
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from functools import lru_cache
import warnings
warnings.filterwarnings("ignore")

# Load the Oxford-IIIT Pet dataset
dataset = torchvision.datasets.OxfordIIITPet(
    root="oxford_pet_dataset",
    download=True,
    transform=transforms.ToTensor(),
    target_types="segmentation"
)
test_dataset = torchvision.datasets.OxfordIIITPet(
    root="oxford_pet_dataset",
    split="test",
    download=True,
    transform=transforms.ToTensor(),
    target_types="segmentation"
)

@lru_cache(maxsize=None)
def segment_image(image, num_clusters=2):
    """
    Segments an image using K-Means clustering.

    Args:
        image (torch.Tensor): The input image tensor (C, H, W).
        num_clusters (int): The number of clusters for K-Means.

    Returns:
        np.ndarray: The segmented image with shape (H, W).
    """
    image_np = image.permute(1, 2, 0).numpy()  # H x W x C
    h, w, c = image_np.shape
    pixels = image_np.reshape(-1, c)

    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    labels = kmeans.fit_predict(pixels)

    return labels.reshape(h, w)

def plot_segmentation(original_image, segmented_image, num_clusters):
    """
    Plots the original and segmented images side by side.

    Args:
        original_image (np.ndarray): The original image.
        segmented_image (np.ndarray): The segmented image.
        num_clusters (int): The number of clusters used in segmentation.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.imshow(original_image)
    ax1.set_title("Original Image")
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2.imshow(segmented_image, cmap='viridis')
    ax2.set_title(f"K-Means Segmentation ({num_clusters} clusters)")
    ax2.set_xticks([])
    ax2.set_yticks([])

    plt.tight_layout()
    plt.savefig("kmeans_segmentation.png")
    plt.close()

def calculate_iou(pred, target):
    """
    Calculate Intersection over Union (IoU) between prediction and target masks.

    Args:
        pred (np.ndarray): Predicted binary mask
        target (np.ndarray): Target binary mask

    Returns:
        float: IoU score
    """
    intersection = np.logical_and(pred, target)
    union = np.logical_or(pred, target)
    iou = np.sum(intersection) / np.sum(union)
    return iou

from tqdm import tqdm
def evaluate_on_test_split(dataset, num_clusters=2, num_test_images=100):
    """
    Evaluates the K-Means segmentation on a subset of test split using IoU metric.

    Args:
        dataset (torchvision.datasets): The dataset to evaluate
        num_clusters (int): Number of clusters for K-means
        num_test_images (int): Number of test images to evaluate (default: 100)

    Returns:
        float: Average IoU score across the test subset
    """
    total_iou = 0
    num_images = min(num_test_images, len(dataset))  # Use smaller of requested or available images

    for idx in tqdm(range(num_images)):
        img, gt = dataset[idx]

        # Get segmentation prediction
        segmented_image = segment_image(img, num_clusters)

        # Convert PIL Image to tensor, then to numpy and ensure binary
        gt_tensor = transforms.ToTensor()(gt)
        gt_np = gt_tensor.numpy().squeeze().astype(bool)

        # Try both possible label assignments (0->0, 1->1) and (0->1, 1->0)
        pred_original = segmented_image.astype(bool)
        pred_flipped = ~pred_original

        # Calculate IoU for both assignments
        iou_original = calculate_iou(pred_original, gt_np)
        iou_flipped = calculate_iou(pred_flipped, gt_np)

        # Use the better IoU score
        iou = max(iou_original, iou_flipped)
        total_iou += iou

        if idx % 10 == 0:  # Progress update every 10 images (changed from 100)
            print(f"Processed {idx+1}/{num_images} images. Current avg IoU: {total_iou/(idx+1):.3f}")

    average_iou = total_iou / num_images
    print(f"Average IoU Score: {average_iou:.3f}")
    return average_iou

# Update the function call
# average_iou = evaluate_on_test_split(test_dataset, num_clusters=2, num_test_images=100)

def visualize_kmeans_segmentation(dataset, num_samples=3):
    """
    Visualize K-means segmentation predictions.
    Shows only the segmentation masks with their IoU scores.

    Args:
        dataset: Dataset containing images and ground truth masks
        num_samples (int): Number of samples to visualize
    """
    plt.figure(figsize=(5, num_samples * 5))
    for i in range(num_samples):
        # Load image and ground truth
        image, gt = dataset[i]

        # Process ground truth - class 1 is the pet
        gt_np = (np.array(gt) == 1).astype(bool)

        # Apply K-means Segmentation
        segmented_image = segment_image(image, num_clusters=2)

        # Compare both possible label assignments
        pred_original = segmented_image.astype(bool)
        pred_flipped = ~pred_original

        iou_original = calculate_iou(pred_original, gt_np)
        iou_flipped = calculate_iou(pred_flipped, gt_np)

        # Select the prediction with better IoU
        best_pred = pred_original if iou_original > iou_flipped else pred_flipped
        best_iou = max(iou_original, iou_flipped)

        # Plot only the prediction
        plt.subplot(num_samples, 1, i + 1)
        plt.imshow(best_pred, cmap='gray')
        plt.axis('off')
        plt.title(f"K-means Segmentation\nIoU: {best_iou:.3f}")

    plt.tight_layout()
    plt.savefig("kmeans_segmentation_predictions.png")
    plt.close()

# Add at the bottom of your file:
visualize_kmeans_segmentation(test_dataset, num_samples=5)

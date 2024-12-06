import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import ToTensor
from skimage.color import rgb2gray
from skimage.filters import gaussian
from functools import lru_cache
from tqdm import tqdm
import torch
from torchvision import transforms

# Preprocess the image
def preprocess_image(image):
    """
    Converts the image to grayscale and smooths it to reduce noise.
    """
    gray_image = rgb2gray(image)
    smoothed_image = gaussian(gray_image, sigma=1.0)  # Smooth the image
    return smoothed_image

# Apply GMM Clustering for Segmentation
def gmm_segmentation(image, n_clusters=2):
    """
    Performs segmentation using GMM clustering.

    Args:
        image (torch.Tensor or np.ndarray): Input image (C, H, W) if tensor or (H, W, C) if numpy
        n_clusters (int): Number of Gaussian components

    Returns:
        np.ndarray: Segmented image with shape (H, W)
    """
    # Convert tensor to numpy if needed
    if torch.is_tensor(image):
        image = image.permute(1, 2, 0).numpy()

    h, w, c = image.shape
    features = image.reshape(-1, c)

    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    labels = gmm.fit_predict(features)

    return labels.reshape(h, w)

def calculate_iou(pred, target):
    """Reusing the IoU calculation function from kmeans.py"""
    intersection = np.logical_and(pred, target)
    union = np.logical_or(pred, target)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def evaluate_gmm_segmentation(dataset, n_clusters=2, num_test_images=100):
    """
    Evaluates GMM segmentation performance on test dataset using IoU metric.

    Args:
        dataset: Test dataset containing images and ground truth masks
        n_clusters (int): Number of Gaussian components
        num_test_images (int): Number of images to evaluate

    Returns:
        float: Average IoU score across test images
    """
    total_iou = 0
    num_images = min(num_test_images, len(dataset))

    for idx in tqdm(range(num_images)):
        img, gt = dataset[idx]

        # Get GMM segmentation
        segmented_image = gmm_segmentation(img, n_clusters)

        # Process ground truth - convert PIL Image to numpy array directly
        gt_np = np.array(gt).astype(bool)

        # Try both label assignments
        pred_original = segmented_image.astype(bool)
        pred_flipped = ~pred_original

        iou_original = calculate_iou(pred_original, gt_np)
        iou_flipped = calculate_iou(pred_flipped, gt_np)

        iou = max(iou_original, iou_flipped)
        total_iou += iou

        if idx % 10 == 0:
            print(f"Processed {idx+1}/{num_images} images. Current avg IoU: {total_iou/(idx+1):.3f}")

    average_iou = total_iou / num_images
    print(f"GMM Average IoU Score: {average_iou:.3f}")
    return average_iou

# Load the Oxford-IIIT Pet Dataset
dataset = OxfordIIITPet(
    root="oxford_pet_dataset",
    download=True,
    target_types="segmentation",
    transforms=lambda img, mask: (ToTensor()(img), mask)
)

def visualize_gmm_segmentation(dataset, num_samples=3):
    """
    Visualize GMM segmentation results alongside original images and ground truth.
    Shows original image, ground truth mask, and the segmentation mask with better IoU.

    Args:
        dataset: Dataset containing images and ground truth masks
        num_samples (int): Number of samples to visualize
    """
    plt.figure(figsize=(15, num_samples * 5))
    for i in range(num_samples):
        # Load image and ground truth
        image, gt = dataset[i]
        image_np = image.permute(1, 2, 0).numpy()  # Convert CHW to HWC

        # Process ground truth - class 1 is the pet
        gt_np = (np.array(gt) == 1).astype(bool)

        # Apply GMM Segmentation
        segmented_image = gmm_segmentation(image_np, n_clusters=2)

        # Compare both possible label assignments
        pred_original = segmented_image.astype(bool)
        pred_flipped = ~pred_original

        iou_original = calculate_iou(pred_original, gt_np)
        iou_flipped = calculate_iou(pred_flipped, gt_np)

        # Select the prediction with better IoU
        best_pred = pred_original if iou_original > iou_flipped else pred_flipped
        best_iou = max(iou_original, iou_flipped)

        # Plot in three columns
        plt.subplot(num_samples, 3, i * 3 + 1)
        plt.imshow(image_np)
        plt.axis('off')
        plt.title("Original Image")

        plt.subplot(num_samples, 3, i * 3 + 2)
        plt.imshow(gt_np, cmap='gray')
        plt.axis('off')
        plt.title("Ground Truth")

        plt.subplot(num_samples, 3, i * 3 + 3)
        plt.imshow(best_pred, cmap='gray')
        plt.axis('off')
        plt.title(f"GMM Segmentation\nIoU: {best_iou:.3f}")

    plt.tight_layout()
    plt.savefig("gmm_segmentation_samples.png")
    plt.close()

# Load test dataset
test_dataset = OxfordIIITPet(
    root="oxford_pet_dataset",
    split="test",
    download=True,
    transforms=lambda img, mask: (ToTensor()(img), mask),
    target_types="segmentation"
)

import os
os.environ['OPENBLAS_NUM_THREADS'] = '4'  # Limit OpenBLAS threads
os.environ['OMP_NUM_THREADS'] = '4'       # Limit OpenMP threads
# # Run visualization for 3 samples
visualize_gmm_segmentation(test_dataset, num_samples=5)


# Evaluate GMM performance
# gmm_iou = evaluate_gmm_segmentation(test_dataset, n_clusters=2, num_test_images=100)

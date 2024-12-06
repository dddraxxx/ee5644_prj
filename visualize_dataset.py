import torch
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np

# Define a function to decode the segmentation mask
def decode_segmentation_mask(mask):
    """
    Decodes the segmentation mask to a color-coded visualization.
    """
    label_colors = {
        0: (0, 0, 0),       # Background
        1: (128, 0, 0),     # Foreground (Pet)
        # 2: (255, 255, 255), # Border
    }
    decoded_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for label, color in label_colors.items():
        decoded_mask[mask == label] = color
    return decoded_mask

# Load the dataset
dataset = OxfordIIITPet(
    root="oxford_pet_dataset",
    download=True,
    target_types="segmentation",
    transforms=lambda img, mask: (ToTensor()(img), torch.tensor(np.array(mask, dtype=np.int64)))
)
test_dataset = OxfordIIITPet(
    root="oxford_pet_dataset",
    download=True,
    target_types="segmentation",
    split="test",
)

# Display a few samples
def visualize_samples(dataset, num_samples=5):
    plt.figure(figsize=(15, num_samples * 5))
    sample_indices = np.random.choice(len(dataset), num_samples, replace=False)
    for i in range(num_samples):
        image, mask = dataset[sample_indices[i]]
        mask = mask.squeeze().numpy()

        # Decode the mask for visualization
        decoded_mask = decode_segmentation_mask(mask)

        # Plot the image and mask side by side
        plt.subplot(num_samples, 2, i * 2 + 1)
        plt.imshow(image.permute(1, 2, 0))  # Convert CHW to HWC for visualization
        plt.axis('off')
        plt.title("Image")

        plt.subplot(num_samples, 2, i * 2 + 2)
        plt.imshow(decoded_mask)
        plt.axis('off')
        plt.title("Segmentation Mask")
    plt.tight_layout()
    plt.savefig("dataset_samples.png")
    plt.close()

# Call the visualization function
visualize_samples(dataset, num_samples=3)

def print_dataset_info(dataset, split):
    """
    Prints dataset information in CSV format.

    Args:
        dataset: The dataset object to extract information from.

    Output:
        Prints the number of images and masks in trainval and test splits.
    """
    # Assuming the dataset is split into trainval and test
    # This is a placeholder logic; adjust based on actual dataset structure
    num_images = len(dataset)
    num_masks = len(dataset)  # Assuming each image has a corresponding mask

    # Print in CSV format
    print("Split,Num_Images,Num_Masks")
    print(f"{split},{num_images},{num_masks}")

# Call the function to print dataset info
print_dataset_info(dataset, "TrainVal")
print_dataset_info(test_dataset, "Test" )

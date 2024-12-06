import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import felzenszwalb
from skimage.util import img_as_float

# Load one sample from the Oxford-IIIT Pet dataset
dataset = torchvision.datasets.OxfordIIITPet(
    root="oxford_pet_dataset",
    download=True,
    transform=transforms.ToTensor(),
    target_types="segmentation"
)

# Use the first image as a test
img, trimap = dataset[0]
image_np = img.permute(1, 2, 0).numpy()

# Convert the image to float (Felzenszwalb expects float images)
image_float = img_as_float(image_np)

# Parameters for Felzenszwalb segmentation
scale = 200           # Larger values mean coarser segmentation
sigma = 0.8           # Gaussian smoothing parameter
min_size = 50         # Merge segments smaller than this

# Perform Felzenszwalb segmentation
segments = felzenszwalb(
    image_float,
    scale=scale,
    sigma=sigma,
    min_size=min_size
)

# 'segments' is a 2D array labeling each pixel with a segment ID.
# The number of segments can be inspected:
num_segments = len(np.unique(segments))
print(f"Number of segments: {num_segments}")

# Plot and save results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.imshow(image_np)
ax1.set_title("Original Image")
ax1.set_xticks([])
ax1.set_yticks([])

ax2.imshow(segments, cmap='tab20')
ax2.set_title(f"Felzenszwalb Segmentation\nSegments: {num_segments}")
ax2.set_xticks([])
ax2.set_yticks([])

plt.tight_layout()
plt.savefig("felzenszwalb_segmentation.png")
plt.close()

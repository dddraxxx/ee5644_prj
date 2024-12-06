import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

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
# Training Loop (Example)
##############################

num_epochs = 2  # Increase this for real training
from tqdm import tqdm
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

##############################
# Inference Demo on One Sample
##############################

model.eval()
with torch.no_grad():
    # Take one image from the dataset
    img, gt = train_dataset[0]
    # Move to GPU and add batch dimension
    img_gpu = img.unsqueeze(0).to('cuda')
    pred = model(img_gpu)['out'].argmax(dim=1).cpu().squeeze().numpy()

# Show the results
# pred is our predicted mask for the single image
fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(15,5))

# Original image
ax1.imshow(img.permute(1,2,0).numpy())
ax1.set_title("Original Image")
ax1.axis('off')

# Ground truth
ax2.imshow(gt, cmap='gray')
ax2.set_title("Ground Truth (Binary)")
ax2.axis('off')

# Prediction
ax3.imshow(pred, cmap='gray')
ax3.set_title("Model Prediction")
ax3.axis('off')

plt.savefig("finetuned_segmentation_result.png")
plt.close()

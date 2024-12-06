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

train_dataset = torchvision.datasets.OxfordIIITPet(
    root="oxford_pet_dataset",
    split="trainval",
    download=True,
    target_types="segmentation",
    transform=img_transform,
    target_transform=target_transform
)
test_dataset = torchvision.datasets.OxfordIIITPet(
    root="oxford_pet_dataset",
    split="test",
    download=True,
    target_types="segmentation",
    transform=img_transform,
    target_transform=target_transform
)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)

##############################
# Model Setup (From Scratch, Simpler Architecture)
##############################

# Use FCN with a ResNet50 backbone, no pretrained weights
model = torchvision.models.segmentation.fcn_resnet50(pretrained=False, weights=None)

# The FCN model typically ends with:
# model.classifier[4] = Conv2d(out_channels=21 for COCO)
# We have 2 classes now: background (0) and pet (1)
model.classifier[4] = nn.Conv2d(512, 2, kernel_size=1)

model.to('cuda')
model.train()

##############################
# Loss and Optimizer
##############################

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

##############################
# Training Loop
##############################

num_epochs = 2  # Increase this for real training
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, targets in tqdm(train_loader):
        images = images.to('cuda')
        targets = targets.to('cuda')

        optimizer.zero_grad()
        outputs = model(images)['out']
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
    img, gt = train_dataset[0]
    img_gpu = img.unsqueeze(0).to('cuda')
    pred = model(img_gpu)['out'].argmax(dim=1).cpu().squeeze().numpy()

# Show results
fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(15,5))

# Original image
ax1.imshow(img.permute(1,2,0).numpy())
ax1.set_title("Original Image")
ax1.axis('off')

# Ground Truth
ax2.imshow(gt, cmap='gray')
ax2.set_title("Ground Truth (Binary)")
ax2.axis('off')

# Prediction
ax3.imshow(pred, cmap='gray')
ax3.set_title("Model Prediction")
ax3.axis('off')

plt.savefig("from_scratch_FCN_segmentation_result.png")
plt.close()

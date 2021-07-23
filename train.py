import os
import glob
import timm
import numpy as np
import albumentations as A
from natsort import natsorted
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import AppleDataset

# set seed 
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

EPOCHS = 1
BATCH_SIZE = 1
LR = 0.001
NUM_WORKERS = 0
mean = (162.8122, 115.4318, 106.5746)
std = (99.4059, 101.3887, 102.0320)

root = os.getcwd()
train = os.path.join(root, 'data', 'train')
val = os.path.join(root, 'data', 'val')
weightspath = os.path.join(root, 'weights', 'best_weights.pth')

transform = A.Compose([
    A.geometric.resize.Resize(height=512, width=512, p=1),
    # A.RandomCrop(width=256, height=256, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.ElasticTransform(p=0.2),
    A.HueSaturationValue(p=0.2),
    # A.transforms.Normalize(
    #     mean=mean,
    #     std=std, 
    #     max_pixel_value=255.0, 
    #     p=1.0
    # ),
    ToTensorV2(),
])

# Load the dataset
train_dataset = AppleDataset(
    # file_paths=natsorted(os.listdir(train)),
    # file_paths=[os.path.join(train, file) for file in natsorted(os.listdir(train))],
    file_paths = natsorted(glob.glob(os.path.join(train, '*'))), 
    labels=np.array(
        [
            0 if 'apple' in file_path else 1
            for file_path in natsorted(os.listdir(train))
        ]
    ),
    transform=transform, #None
)

# Create a dataloader
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    # train=True,
    num_workers=NUM_WORKERS,
)

model = timm.create_model('resnet50', pretrained=True)
model = nn.Sequential(model, nn.Linear(1000, 2))

# Optimizer 
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training
for epoch in range(EPOCHS):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.data

        # save the best weight files 
        print(f"Training loss: {running_loss/100}")
        running_loss = 0.0
        torch.save(model.state_dict(), f'./weights/weights_{EPOCHS}.pth')
        print("Epoch: {}, Batch: {}, Loss: {}".format(epoch+1, i+1, loss.data))
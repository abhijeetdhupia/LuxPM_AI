import os 
import timm 
import glob
import numpy as np
import torch
import torch.nn as nn
import albumentations as A
import matplotlib.pyplot as plt
from natsort import natsorted 
from albumentations.pytorch import ToTensorV2

from dataset import AppleDataset

BATCH_SIZE = 1
NUM_WORKERS = 0
# set seed 
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

root = os.getcwd()
test = os.path.join(root, 'data', 'test')
weightspath = os.path.join(root, 'weights', 'best_weights.pth')

model = timm.create_model('resnet50', pretrained=True)
model = nn.Sequential(model, nn.Linear(1000, 2))
model.load_state_dict(torch.load(weightspath))

test_dataset = AppleDataset(
    # file_paths=natsorted(os.listdir(test)),
    # file_paths=[os.path.join(test, file) for file in natsorted(os.listdir(test))],
    file_paths=natsorted(glob.glob(os.path.join(test, '*'))), 
    labels=np.array(
        [
            0 if 'apple' in file_path else 1
            for file_path in natsorted(os.listdir(test))
        ]
    ),
    transform=A.Compose([
        A.Resize(512,512),
        ToTensorV2(),
    ]),
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)

# Whole dataset prediction
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = model(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        for i in range(len(predicted)):
            if predicted[i] != labels[i]:
                print(f"Predicted: {predicted[i]}, Actual: {labels[i]}")
                img = images[i].permute(1, 2, 0).numpy()
                img = img.astype('uint8')
                plt.imshow(img)
                plt.show()

print(f"Accuracy of the network on the {total} test images: {100 * correct / total} %")
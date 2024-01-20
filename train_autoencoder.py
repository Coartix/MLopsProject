import torch
import torch.nn as nn
import torch.nn.functional as F
from Autoencoder import Autoencoder, transform_train
from PIL import Image
import glob
import numpy as np
from torch.utils.data import DataLoader

from segtools.data import Dataset
from segtools.classes import classes

# define dataset
paths = []

for gt_path in glob.glob("VOCdevkit/VOC2012/SegmentationClass/*png"):
  img = np.array(Image.open(gt_path)).reshape(-1)
  present_classes = np.unique(img)

  overlap = any([c in classes for c in present_classes])
  if overlap:
    paths.append(gt_path.split("/")[-1].split('.')[0])

with open("VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt", "r") as f:
  train_ids = set(map(lambda x: x.strip(), f.readlines()))

with open("VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt", "r") as f:
  val_ids = set(map(lambda x: x.strip(), f.readlines()))


ids = train_ids.union(val_ids)
paths = list(filter(lambda x: x in ids, paths))

# Load the dataset
dataset = Dataset(paths, transform=transform_train)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define the model
autoencoder = Autoencoder()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
autoencoder.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.0005)

num_epochs = 10
for epoch in range(num_epochs):
    for data in loader:
        img, _ = data
        img = img.to(device)

        # Forward pass
        output = autoencoder(img)
        loss = criterion(output, img)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the model
torch.save(autoencoder.state_dict(), 'models/autoencoder.pth')

import glob
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torchvision.transforms.functional as Fv
import torchvision
from PIL import Image
import joblib

import os
import subprocess

# our imports
from segtools.data import Dataset, SelectClasses, Compose, Resize, RandomHorizontalFlip, ToTensor, Normalize, RandomResizedCrop
from segtools.model import DeepLab
from segtools.iou import confusion_matrix, get_miou, test_iou

# download data
def download_and_extract(url, tar_file):
    # Check if the file already exists
    if not os.path.exists(tar_file):
        # Download the file using wget
        subprocess.run(["wget", url])
        
        # Extract the file
        subprocess.run(["tar", "-xf", tar_file])
    else:
        print(f"{tar_file} already exists.")

# URLs and tar file names
downloads = [
    ("http://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar", "VOCtrainval_06-Nov-2007.tar"),
    ("http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar", "VOCtrainval_11-May-2012.tar")
]

for url, tar_file in downloads:
    download_and_extract(url, tar_file)


# define dataset
paths = []
classes = {
    #1: 'aeroplane',
    #2: 'bicycle',
    #3: 'bird',
    #4: 'boat',
    #5: 'bottle',
    #6: 'bus',
    #7: 'car',
    #8: 'cat',
    #9: 'chair',
    #10: 'cow',
    #11: 'diningtable',
    12: 'dog',
    #13: 'horse',
    #14: 'motorbike',
    15: 'person',
    #16: 'pottedplant',
    #17: 'sheep',
    #18: 'sofa',
    #19: 'train',
    #20: 'tvmonitor'
}

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

  imagenet_mean = torch.tensor([0.485, 0.456, 0.406])
imagenet_std = torch.tensor([0.229, 0.224, 0.225])

val_transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(imagenet_mean, imagenet_std),
    SelectClasses(list(classes.keys()))
])

train_transform = Compose([
    RandomResizedCrop((224, 224), (0.5, 2.0)),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize(imagenet_mean, imagenet_std),
    SelectClasses(list(classes.keys()))
])

train_paths = list(filter(lambda x: x in train_ids, paths))
val_paths = list(filter(lambda x: x in val_ids, paths))

train_dataset = Dataset(train_paths, train_transform)
val_dataset = Dataset(val_paths, val_transform)

test_dataset = val_dataset

#print(f"{len(train_dataset)} train images and {len(test_dataset)} test images")

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# define evaluation function
def eval_model(net, loader):
  net.eval()
  all_miou, classes_miou, loss = 0., [0. for _ in range(len(classes) + 1)], 0.
  c = 0
  for x, y in loader:
    with torch.no_grad():
      # No need to compute gradient here thus we avoid storing intermediary activations
      logits = net(x.cuda()).cpu()

    loss += F.cross_entropy(logits, y.long(), ignore_index=255).item()
    preds = logits.argmax(dim=1)
    for i, pred in enumerate(preds):
      _all_miou, _classes_miou = get_miou(confusion_matrix(y[i], pred, len(classes) + 1))
      all_miou += _all_miou.item()
      for class_index in range(len(classes_miou)):
        classes_miou[class_index] += _classes_miou[class_index].item()
    c += len(x)

  all_miou = round(100 * all_miou / c, 2)
  classes_miou = [round(100 * iou / c, 2) for iou in classes_miou]
  loss /= len(loader)
  net.train()
  return round(loss, 5), all_miou, classes_miou

# define model
model = DeepLab(len(classes) + 1).cuda()

# train model
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
epochs = 1

for epoch in range(epochs):
  _loss, c = 0., 0

  for x, y in train_loader:
    x, y = x.cuda(), y.cuda()

    optimizer.zero_grad()
    logits = model(x)
    loss = F.cross_entropy(logits, y.long(), ignore_index=255)

    loss.backward()
    optimizer.step()

    _loss += loss.item()
    c += len(x)

  print(f"Epoch {epoch}: train loss: {round(_loss / c, 5)}")

_, test_miou, test_mious = eval_model(model, test_loader)
print(f"Final mIoU: {test_miou}, and per class:")
print(f"\tBackground mIoU: {test_mious[0]}")
for i, class_name in enumerate(classes.values(), start=1):
  print(f"\t{class_name} mIoU: {test_mious[i]}")

# save model
joblib.dump(model, "models/model.joblib")
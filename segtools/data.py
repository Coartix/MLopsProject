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
import random
import math

class Dataset(torch.utils.data.Dataset):
  def __init__(self, ids, transform=None):
    self.ids = ids

    self.transform = transform

  def __len__(self):
    return len(self.ids)

  def __getitem__(self, index):
    img = Image.open(f"VOCdevkit/VOC2012/JPEGImages/{self.ids[index]}.jpg").convert("RGB")
    target = Image.open(f"VOCdevkit/VOC2012/SegmentationClass/{self.ids[index]}.png")

    img, target = self.transform(img, target)

    return img, target

class SelectClasses:
  def __init__(self, classes):
    self.classes = {0: 0, 255: 255}
    for i, c in enumerate(classes, start=1):
      self.classes[c] = i

  def __call__(self, x, y):
    return x, y.apply_(
        lambda c: self.classes.get(c, 0)
    )

class Compose:
  def __init__(self, transforms):
    self.transforms = transforms

  def __call__(self, x, y):
    for t in self.transforms:
      x, y = t(x, y)
    return x, y


class Resize:
  def __init__(self, size):
    self.size = size

  def __call__(self, x, y):
    x = Fv.resize(x, self.size, Image.BILINEAR)
    y = Fv.resize(y, self.size, Image.NEAREST)
    return x, y


class RandomHorizontalFlip:
  def __call__(self, x, y):
    if random.random() < 0.5:
      return Fv.hflip(x), Fv.hflip(y)
    return x, y 


class ToTensor:
  def __call__(self, x, y):
    return Fv.to_tensor(x), torch.from_numpy(np.array(y, dtype=np.uint8))


class Normalize:
  def __init__(self, mu, sigma):
    self.mu, self.sigma = mu, sigma

  def __call__(self, x, y):
    return Fv.normalize(x, self.mu, self.sigma), y


class RandomResizedCrop:
    """From https://github.com/fcdl94/MiB"""
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        area = img.size[0] * img.size[1]

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = img.size[0] / img.size[1]
        if (in_ratio < min(ratio)):
            w = img.size[0]
            h = int(round(w / min(ratio)))
        elif (in_ratio > max(ratio)):
            h = img.size[1]
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = img.size[0]
            h = img.size[1]
        i = (img.size[1] - h) // 2
        j = (img.size[0] - w) // 2
        return i, j, h, w

    def __call__(self, img, lbl=None):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        if lbl is not None:
            return Fv.resized_crop(img, i, j, h, w, self.size, self.interpolation), \
                   Fv.resized_crop(lbl, i, j, h, w, self.size, Image.NEAREST)
        else:
            return Fv.resized_crop(img, i, j, h, w, self.size, self.interpolation)

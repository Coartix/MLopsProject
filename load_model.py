import glob
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torch
from PIL import Image
import joblib
import numpy as np
import matplotlib.pyplot as plt

from segtools.data import Compose, Resize, ToTensor, Normalize, SelectClasses
from segtools.model import DeepLab, forward_extract
from segtools.classes import classes

def preprocess_image(image_path, val_transform):
    # Load and transform the image
    image = Image.open(image_path).convert("RGB")
    image = val_transform(image)
    return image

def predict(model, image):
    # Predict the segmentation map
    with torch.no_grad():
        pred = model(image.cuda()[None]).cpu().argmax(dim=1).numpy().astype(np.uint8)
    return pred

def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

voc_colormap = color_map()[:, None, :]


def colorize(image, array):
  new_im = np.dot(array == 0, voc_colormap[0])
  for i in range(1, voc_colormap.shape[0]):
    new_im += np.dot(array == i, voc_colormap[i])
  new_im = Image.fromarray(new_im.astype(np.uint8))
  return Image.blend(image, new_im, alpha=0.8) 

def display_results(image_path, target, pred):
    # Display the original image, ground truth, and prediction
    image_jpg = Image.open(image_path).resize((224, 224))
    plt.figure(figsize=(12, 8))
    plt.subplot(1, 3, 1)
    plt.imshow(image_jpg)
    plt.subplot(1, 3, 2)
    plt.imshow(colorize(image_jpg, target[..., None]))
    plt.subplot(1, 3, 3)
    plt.imshow(colorize(image_jpg, pred[0][..., None]))
    plt.show()

if __name__ == "__main__":
    path = "2007_000346"
    model_path = "models/model.pth"
    image_path = f"/home/pili/scia/MLopsProject/VOCdevkit/VOC2012/JPEGImages/{path}.jpg"  # Replace with your image path
    target_path = f"VOCdevkit/VOC2012/SegmentationClass/{path}.png"  # Replace with your target path

    # Define your val_transform here (same as used during training)
    val_transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])),
        SelectClasses(list(classes.keys()))
    ])

    model = DeepLab(len(classes) + 1).cuda()  # Initialize model architecture
    model.load_state_dict(torch.load(model_path))
    model.eval()
    image = Image.open(image_path).convert("RGB")
    image, target = val_transform(image, Image.open(target_path))

    with torch.no_grad():
        pred = model(image.cuda()[None]).cpu().argmax(dim=1).numpy().astype(np.uint8)

    display_results(image_path, target, pred)

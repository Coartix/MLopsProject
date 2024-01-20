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
import torch
from torch import nn
from torch.nn import functional as F
import torchvision.transforms.functional as Fv
from Autoencoder import Autoencoder, transform


from segtools.data import Compose, Resize_single, ToTensor_single, Normalize_single, SelectClasses
from segtools.model import DeepLab, forward_extract
from segtools.classes import classes

imagenet_mean = torch.tensor([0.485, 0.456, 0.406])
imagenet_std = torch.tensor([0.229, 0.224, 0.225])

class Resize:
  def __init__(self, size):
    self.size = size

  def __call__(self, x, y):
    x = Fv.resize(x, self.size, Image.BILINEAR)
    y = Fv.resize(y, self.size, Image.NEAREST)
    return x, y
  
class ToTensor:
  def __call__(self, x, y):
    return Fv.to_tensor(x), torch.from_numpy(np.array(y, dtype=np.uint8))

class Normalize:
  def __init__(self, mu, sigma):
    self.mu, self.sigma = mu, sigma

  def __call__(self, x, y):
    return Fv.normalize(x, self.mu, self.sigma), y

val_transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(imagenet_mean, imagenet_std),
    SelectClasses(list(classes.keys()))
])

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


def colorize(image, array) -> Image:
  new_im = np.dot(array == 0, voc_colormap[0])
  for i in range(1, voc_colormap.shape[0]):
    new_im += np.dot(array == i, voc_colormap[i])
  new_im = Image.fromarray(new_im.astype(np.uint8))
  return Image.blend(image, new_im, alpha=0.8) 

def display_results(image_path, pred):
    # Display the original image and prediction
    image_jpg = Image.open(image_path).resize((224, 224))
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(image_jpg)
    plt.title("Original Image")
    plt.subplot(1, 2, 2)
    plt.imshow(colorize(image_jpg, pred[0][..., None]))
    plt.title("Predicted Segmentation")
    plt.show()

def pred_using_model(image_data, model_path):
    # Define your val_transform here (same as used during training)
    val_transform = Compose([
        Resize_single((224, 224)),
        ToTensor_single(),
        Normalize_single(torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])),
        #SelectClasses(list(classes.keys()))
    ])

    model = DeepLab(len(classes) + 1).cuda()  # Initialize model architecture
    model.load_state_dict(torch.load(model_path))

    real_image = Image.fromarray(np.array(image_data, dtype=np.uint8)).convert("RGB")
    image = val_transform(real_image)  # Only transform the image, no target is needed

    model.eval()
    with torch.no_grad():
        pred = model(image.cuda()[None]).cpu().argmax(dim=1).numpy().astype(np.uint8)

    return colorize(real_image.resize((224,224)), pred[0][..., None]).convert("RGB")

def similarity_using_ae(image_data, model_path):
    # Load the autoencoder model
    autoencoder = Autoencoder()
    autoencoder.load_state_dict(torch.load(model_path))
    autoencoder.eval()

    # Convert the image data to PIL image and apply transform
    image = Image.fromarray(np.array(image_data, dtype=np.uint8)).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Move image to the same device as the model
    device = next(autoencoder.parameters()).device
    image = image.to(device)

    # Process the image using the autoencoder
    with torch.no_grad():
        reconstructed_image = autoencoder(image)

    # Compute similarity (reconstruction error)
    error = torch.mean((image - reconstructed_image) ** 2)
    return error.item()
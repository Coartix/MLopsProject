import torch
import torch.nn as nn
import torch.nn.functional as F

from segtools.data import Dataset, SelectClasses, Compose, Resize, RandomHorizontalFlip, ToTensor, Normalize, RandomResizedCrop, Resize_single, ToTensor_single, Normalize_single
from segtools.classes import classes

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.enc1 = nn.Conv2d(3, 16, 3, stride=2, padding=1) # input channels, output channels, kernel size
        self.enc2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.enc3 = nn.Conv2d(32, 64, 7)

        # Decoder
        self.dec1 = nn.ConvTranspose2d(64, 32, 7)
        self.dec2 = nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1)
        self.dec3 = nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        # Encoder
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))

        # Decoder
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = torch.sigmoid(self.dec3(x))
        return x
    
imagenet_mean = torch.tensor([0.485, 0.456, 0.406])
imagenet_std = torch.tensor([0.229, 0.224, 0.225])

transform_train = Compose([
    RandomResizedCrop((224, 224), (0.5, 2.0)),
    ToTensor(),
    Normalize(imagenet_mean, imagenet_std),
    SelectClasses(list(classes.keys()))
])

transform = Compose([
        Resize_single((224, 224)),
        ToTensor_single(),
        Normalize_single(torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225]))
    ])
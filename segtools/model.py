import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torchvision

def forward_extract(self, x):
  x = self.conv1(x)
  x = self.bn1(x)
  x = self.relu(x)
  x = self.maxpool(x)
  x = self.layer1(x)
  x = self.layer2(x)
  x = self.layer3(x)
  x = self.layer4(x)
  return x

class DeepLabHead(nn.Sequential):
  def __init__(self, in_channels, num_classes):
      super(DeepLabHead, self).__init__(
          ASPP(in_channels, [6, 12, 18]),
          nn.Conv2d(256, 256, 3, padding=1, bias=False),
          nn.BatchNorm2d(256),
          nn.ReLU(),
          nn.Conv2d(256, num_classes, 1)
      )

class ASPPPooling(nn.Sequential):
  def __init__(self, in_channels, out_channels):
      super(ASPPPooling, self).__init__(
          nn.AdaptiveAvgPool2d(1),
          nn.Conv2d(in_channels, out_channels, 1, bias=False),
          nn.BatchNorm2d(out_channels),
          nn.ReLU())

  def forward(self, x):
      size = x.shape[-2:]
      for mod in self:
          x = mod(x)
      return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

class DeepLab(nn.Module):
  def __init__(self, nb_classes):
    super().__init__()

    self.resnet = torchvision.models.resnet50(pretrained=True, replace_stride_with_dilation=[False, True, True])
    self.resnet.forward = forward_extract.__get__(
        self.resnet,
        torchvision.models.ResNet
    )  # monkey-patching

    self.deeplab = DeepLabHead(2048, nb_classes)

  def forward(self, x):
    x = self.resnet(x)
    x = self.deeplab(x)

    return F.interpolate(
        x, size=(224, 224), mode="bilinear", align_corners=False
    )
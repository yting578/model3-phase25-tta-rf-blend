"""Official DeepYeast CNN — PyTorch (tanelp/deepyeast)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
  """Conv2D → BatchNorm → ReLU (Keras layer names preserved for weight import)."""

  def __init__(self, in_channels: int, out_channels: int) -> None:
    super().__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=True)
    self.bn = nn.BatchNorm2d(out_channels)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return F.relu(self.bn(self.conv(x)))


class DeepYeastNet(nn.Module):
  """DeepYeast architecture from Pärnamaa & Parts (2017). Input: NCHW float32."""

  def __init__(self, num_classes: int = 12, dropout_rate: float = 0.5) -> None:
    super().__init__()
    self.conv1_1 = ConvBNReLU(2, 64)
    self.conv1_2 = ConvBNReLU(64, 64)
    self.pool1 = nn.MaxPool2d(2, 2)

    self.conv2_1 = ConvBNReLU(64, 128)
    self.conv2_2 = ConvBNReLU(128, 128)
    self.pool2 = nn.MaxPool2d(2, 2)

    self.conv3_1 = ConvBNReLU(128, 256)
    self.conv3_2 = ConvBNReLU(256, 256)
    self.conv3_3 = ConvBNReLU(256, 256)
    self.conv3_4 = ConvBNReLU(256, 256)
    self.pool3 = nn.MaxPool2d(2, 2)

    flat_dim = 256 * 8 * 8
    self.ip1 = nn.Linear(flat_dim, 512)
    self.bn4 = nn.BatchNorm1d(512)
    self.ip2 = nn.Linear(512, 512)
    self.bn5 = nn.BatchNorm1d(512)
    self.ip3 = nn.Linear(512, num_classes)
    self.dropout_rate = dropout_rate

  def forward(self, x: torch.Tensor, train: bool = True) -> torch.Tensor:
    x = self.conv1_1(x)
    x = self.conv1_2(x)
    x = self.pool1(x)
    x = self.conv2_1(x)
    x = self.conv2_2(x)
    x = self.pool2(x)
    x = self.conv3_1(x)
    x = self.conv3_2(x)
    x = self.conv3_3(x)
    x = self.conv3_4(x)
    x = self.pool3(x)

    x = x.flatten(1)
    x = F.relu(self.bn4(self.ip1(x)))
    x = F.dropout(x, p=self.dropout_rate, training=train)
    x = F.relu(self.bn5(self.ip2(x)))
    x = F.dropout(x, p=self.dropout_rate, training=train)
    return self.ip3(x)

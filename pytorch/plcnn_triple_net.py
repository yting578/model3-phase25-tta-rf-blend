"""PLCNN 三分支：保留原文档容量，加正则（5% 与全量通用）。

结构（与朋友方案一致）：
  VGG / ResNet / DenseNet 三分支，通道 32→64→128→256
  每分支 GAP1~4 拼接 → SE → 512 维 → 1536 → 512 → 12

相对初版仅加：
  - Dropout（分支头 + 分类头）
  - 训练侧：flip 增强、weight decay、ReduceLR、early stop
这些对 5% 防过拟合、全量稳定训练都适用，不牺牲全量容量。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

STAGE_CHANNELS = (32, 64, 128, 256)
BRANCH_DIM = 512


class SEVector(nn.Module):
  def __init__(self, dim: int, reduction: int = 16) -> None:
    super().__init__()
    hidden = max(dim // reduction, 4)
    self.fc1 = nn.Linear(dim, hidden)
    self.fc2 = nn.Linear(hidden, dim)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    w = torch.sigmoid(self.fc2(F.relu(self.fc1(x))))
    return x * w


class BranchHead(nn.Module):
  def __init__(
    self,
    gap_dim: int = sum(STAGE_CHANNELS),
    out_dim: int = BRANCH_DIM,
    dropout: float = 0.3,
  ) -> None:
    super().__init__()
    self.se = SEVector(gap_dim, reduction=16)
    self.fc = nn.Linear(gap_dim, out_dim)
    self.drop = nn.Dropout(dropout)

  def forward(self, gap_list: list[torch.Tensor]) -> torch.Tensor:
    x = torch.cat(gap_list, dim=1)
    x = self.se(x)
    return self.drop(F.relu(self.fc(x)))


def _gap_vec(feat: torch.Tensor) -> torch.Tensor:
  return feat.mean(dim=(2, 3))


class VGGBranch(nn.Module):
  def __init__(self, dropout: float = 0.3) -> None:
    super().__init__()
    self.stages = nn.ModuleList()
    in_ch = 2
    for out_ch in STAGE_CHANNELS:
      self.stages.append(
        nn.Sequential(
          nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
          nn.BatchNorm2d(out_ch),
          nn.ReLU(inplace=True),
          nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
          nn.BatchNorm2d(out_ch),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(2),
        )
      )
      in_ch = out_ch
    self.head = BranchHead(dropout=dropout)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    gaps: list[torch.Tensor] = []
    for stage in self.stages:
      x = stage(x)
      gaps.append(_gap_vec(x))
    return self.head(gaps)


class ResidualBlock(nn.Module):
  def __init__(self, channels: int) -> None:
    super().__init__()
    self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(channels)
    self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(channels)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    residual = x
    y = F.relu(self.bn1(self.conv1(x)))
    y = self.bn2(self.conv2(y))
    return F.relu(residual + y)


class ResNetBranch(nn.Module):
  def __init__(self, dropout: float = 0.3) -> None:
    super().__init__()
    self.stages = nn.ModuleList()
    in_ch = 2
    for out_ch in STAGE_CHANNELS:
      self.stages.append(
        nn.Sequential(
          nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
          nn.BatchNorm2d(out_ch),
          nn.ReLU(inplace=True),
          ResidualBlock(out_ch),
          ResidualBlock(out_ch),
          nn.MaxPool2d(2),
        )
      )
      in_ch = out_ch
    self.head = BranchHead(dropout=dropout)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    gaps: list[torch.Tensor] = []
    for stage in self.stages:
      x = stage(x)
      gaps.append(_gap_vec(x))
    return self.head(gaps)


class _DenseLayer(nn.Module):
  def __init__(self, in_ch: int, growth_rate: int) -> None:
    super().__init__()
    self.net = nn.Sequential(
      nn.BatchNorm2d(in_ch),
      nn.ReLU(inplace=True),
      nn.Conv2d(in_ch, growth_rate, 3, padding=1, bias=False),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return torch.cat([x, self.net(x)], dim=1)


class DenseBlock(nn.Module):
  def __init__(self, in_ch: int, growth_rate: int = 32, num_layers: int = 4) -> None:
    super().__init__()
    layers = []
    ch = in_ch
    for _ in range(num_layers):
      layers.append(_DenseLayer(ch, growth_rate))
      ch += growth_rate
    self.layers = nn.ModuleList(layers)
    self.out_channels = ch

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    for layer in self.layers:
      x = layer(x)
    return x


class DenseNetBranch(nn.Module):
  def __init__(
    self,
    growth_rate: int = 32,
    num_layers: int = 4,
    dropout: float = 0.3,
  ) -> None:
    super().__init__()
    self.stages = nn.ModuleList()
    in_ch = 2
    for out_ch in STAGE_CHANNELS:
      dense_out = in_ch + growth_rate * num_layers
      self.stages.append(
        nn.ModuleDict(
          {
            "dense": DenseBlock(in_ch, growth_rate=growth_rate, num_layers=num_layers),
            "trans": nn.Sequential(
              nn.BatchNorm2d(dense_out),
              nn.ReLU(inplace=True),
              nn.Conv2d(dense_out, out_ch, 1, bias=False),
              nn.MaxPool2d(2),
            ),
          }
        )
      )
      in_ch = out_ch
    self.head = BranchHead(dropout=dropout)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    gaps: list[torch.Tensor] = []
    for stage in self.stages:
      x = stage["dense"](x)
      x = stage["trans"](x)
      gaps.append(_gap_vec(x))
    return self.head(gaps)


class PLCNNTripleNet(nn.Module):
  """三分支 PLCNN（原容量 + Dropout，5%/全量通用）。"""

  def __init__(self, num_classes: int = 12, dropout: float = 0.5) -> None:
    super().__init__()
    branch_drop = min(dropout, 0.3)
    self.branch_vgg = VGGBranch(dropout=branch_drop)
    self.branch_resnet = ResNetBranch(dropout=branch_drop)
    self.branch_densenet = DenseNetBranch(dropout=branch_drop)
    self.classifier = nn.Sequential(
      nn.Linear(BRANCH_DIM * 3, 512),
      nn.ReLU(inplace=True),
      nn.Dropout(dropout),
      nn.Linear(512, num_classes),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    v = self.branch_vgg(x)
    r = self.branch_resnet(x)
    d = self.branch_densenet(x)
    return self.classifier(torch.cat([v, r, d], dim=1))

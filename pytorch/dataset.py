"""DeepYeast data loading for PyTorch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def preprocess_input(x: np.ndarray) -> np.ndarray:
  """Official DeepYeast scaling: uint8 or [0,1] → [-1, 1]."""
  x = x.astype(np.float32)
  if x.max() > 1.0:
    x = x / 255.0
  return (x - 0.5) * 2.0


def ensure_metadata(data_dir: Path) -> Path:
  """Build metadata.csv from labels.csv if missing (prepare script output)."""
  meta_path = data_dir / "metadata.csv"
  if meta_path.exists():
    return meta_path
  labels_path = data_dir / "labels.csv"
  if not labels_path.exists():
    raise FileNotFoundError(
      f"Neither metadata.csv nor labels.csv found under {data_dir}"
    )
  labels = pd.read_csv(labels_path)
  records = []
  for _, row in labels.iterrows():
    records.append(
      {
        "split_orig": row["split"],
        "class_orig": row["class_folder"],
        "full_path": str(data_dir / row["relative_path"]),
        "label": int(row["label_idx"]),
        "class": row["label_name"],
      }
    )
  meta = pd.DataFrame(records).rename_axis("frame_id").reset_index()
  meta.to_csv(meta_path, index=False)
  return meta_path


def load_metadata(data_dir: Path) -> pd.DataFrame:
  ensure_metadata(data_dir)
  return pd.read_csv(data_dir / "metadata.csv")


class DeepYeastTorchDataset(Dataset):
  """Load 64×64×2 PNGs; optional flip / 90° rotation augment."""

  def __init__(
    self,
    metadata: pd.DataFrame,
    split: str,
    augment: bool = False,
    strong_augment: bool = False,
  ) -> None:
    self.rows = metadata[metadata["split_orig"] == split].reset_index(drop=True)
    self.augment = augment
    self.strong_augment = strong_augment

  def __len__(self) -> int:
    return len(self.rows)

  def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    row = self.rows.iloc[idx]
    img = np.asarray(Image.open(row["full_path"]), dtype=np.uint8)
    x = preprocess_input(img[:, :, :2])
    if self.augment:
      if np.random.rand() < 0.5:
        x = np.flip(x, axis=1).copy()
      if np.random.rand() < 0.5:
        x = np.flip(x, axis=0).copy()
      if self.strong_augment:
        # Random k*90° rotation (k=0..3); fluorescence is rotation-equivariant enough
        k = int(np.random.randint(0, 4))
        if k:
          x = np.rot90(x, k=k, axes=(0, 1)).copy()
    x_t = torch.from_numpy(x).permute(2, 0, 1)
    y_t = torch.tensor(int(row["label"]), dtype=torch.long)
    return x_t, y_t


def make_loaders(
  data_dir: Path,
  batch_size: int,
  augment: bool = False,
  strong_augment: bool = False,
  num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
  meta = load_metadata(data_dir)
  train_ds = DeepYeastTorchDataset(
    meta, "train", augment=augment, strong_augment=strong_augment
  )
  val_ds = DeepYeastTorchDataset(meta, "val", augment=False)
  test_ds = DeepYeastTorchDataset(meta, "test", augment=False)

  loader_kwargs = {
    "num_workers": num_workers,
    "pin_memory": torch.cuda.is_available(),
  }
  train_loader = DataLoader(
    train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs
  )
  val_loader = DataLoader(
    val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs
  )
  test_loader = DataLoader(
    test_ds, batch_size=batch_size, shuffle=False, **loader_kwargs
  )
  return train_loader, val_loader, test_loader

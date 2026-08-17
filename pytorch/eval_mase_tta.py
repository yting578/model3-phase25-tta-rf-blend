#!/usr/bin/env python3
"""Test-time augmentation (TTA) eval for MaSE-Net Lite — no retraining.

Averages softmax probabilities over deterministic flip / rotation views.

Example:
  python pytorch/eval_mase_tta.py \\
    --data-dir /path/to/deepyeast_full \\
    --checkpoint /path/to/mase_lite_full_v7/best.pt \\
    --split test \\
    --tta-mode flip_rot \\
    --out-dir /path/to/mase_lite_full_v7/tta_eval
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
  sys.path.insert(0, str(_HERE))

from dataset import make_loaders
from eval_mase_uq import config_from_results_json, load_checkpoint, resolve_device, set_seed
from mase_lite_net import MASE_LITE_DEFAULT, MaSELiteConfig, MaSELiteNet, config_to_dict


def tta_views(x: torch.Tensor, mode: str) -> list[torch.Tensor]:
  """Deterministic TTA views for batch x (B, C, H, W)."""
  views = [x]
  views.append(torch.flip(x, dims=[-1]))
  views.append(torch.flip(x, dims=[-2]))
  views.append(torch.flip(x, dims=[-1, -2]))
  if mode == "flip_rot":
    for k in (1, 2, 3):
      views.append(torch.rot90(x, k, dims=(2, 3)))
  elif mode != "flip":
    raise ValueError(f"Unknown tta-mode: {mode!r} (use flip|flip_rot)")
  return views


@torch.inference_mode()
def forward_tta(
  model: MaSELiteNet,
  images: torch.Tensor,
  mode: str,
) -> torch.Tensor:
  views = tta_views(images, mode)
  probs_sum: torch.Tensor | None = None
  for view in views:
    log_probs = model(view, train=False, return_details=False)
    assert isinstance(log_probs, torch.Tensor)
    probs = log_probs.exp()
    probs_sum = probs if probs_sum is None else probs_sum + probs
  assert probs_sum is not None
  return probs_sum / float(len(views))


@torch.inference_mode()
def eval_split(
  model: MaSELiteNet,
  loader: DataLoader,
  device: torch.device,
  *,
  use_tta: bool,
  tta_mode: str,
) -> dict[str, Any]:
  model.eval()
  correct_base = correct_tta = total = 0
  coverages: list[float] = []

  for images, labels in loader:
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)

    log_probs, details = model(images, train=False, return_details=True)
    pred_base = log_probs.argmax(dim=-1)
    correct_base += int((pred_base == labels).sum().item())

    if use_tta:
      probs_tta = forward_tta(model, images, tta_mode)
      pred_tta = probs_tta.argmax(dim=-1)
      correct_tta += int((pred_tta == labels).sum().item())

    total += int(labels.numel())
    coverages.append(float(details["mask"].mean().item()))

  out: dict[str, Any] = {
    "n": total,
    "baseline_accuracy": correct_base / max(total, 1),
    "mean_mask_coverage": float(np.mean(coverages)) if coverages else float("nan"),
  }
  if use_tta:
    out["tta_accuracy"] = correct_tta / max(total, 1)
    out["tta_gain_pp"] = 100.0 * (out["tta_accuracy"] - out["baseline_accuracy"])
  return out


def run_tta_eval(
  *,
  data_dir: Path,
  checkpoint: Path,
  results_json: Path | None = None,
  split: str = "test",
  tta_mode: str = "flip_rot",
  batch_size: int = 64,
  device_pref: str = "auto",
  seed: int = 42,
  out_dir: Path | None = None,
) -> dict[str, Any]:
  set_seed(seed)
  device = resolve_device(device_pref)
  data_dir = data_dir.resolve()
  checkpoint = checkpoint.resolve()

  if results_json is None:
    candidate = checkpoint.parent / "results.json"
    results_json = candidate if candidate.exists() else None

  config = config_from_results_json(results_json) or MaSELiteConfig(
    **config_to_dict(MASE_LITE_DEFAULT)
  )
  model = MaSELiteNet(config).to(device)
  load_checkpoint(model, checkpoint)

  _, val_loader, test_loader = make_loaders(
    data_dir,
    batch_size=batch_size,
    augment=False,
    strong_augment=False,
    num_workers=0,
  )
  loaders = {"val": val_loader, "test": test_loader}

  splits: dict[str, Any] = {}
  for name in ("val", "test"):
    if split not in (name, "both"):
      continue
    base = eval_split(model, loaders[name], device, use_tta=False, tta_mode=tta_mode)
    tta = eval_split(model, loaders[name], device, use_tta=True, tta_mode=tta_mode)
    splits[name] = {
      **base,
      "tta_accuracy": tta["tta_accuracy"],
      "tta_gain_pp": tta["tta_gain_pp"],
    }

  payload: dict[str, Any] = {
    "method": "mase_tta_eval",
    "checkpoint": str(checkpoint),
    "data_dir": str(data_dir),
    "tta_mode": tta_mode,
    "n_views": len(tta_views(torch.zeros(1, 2, 64, 64), tta_mode)),
    "seed": seed,
    "splits": splits,
  }
  if splits.get("test"):
    payload["test_baseline_accuracy"] = splits["test"]["baseline_accuracy"]
    payload["test_tta_accuracy"] = splits["test"]["tta_accuracy"]
    payload["test_tta_gain_pp"] = splits["test"]["tta_gain_pp"]

  if out_dir is not None:
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "summary.json").open("w") as f:
      json.dump(payload, f, indent=2)

  return payload


def main() -> None:
  parser = argparse.ArgumentParser(description="MaSE TTA eval (no retrain)")
  parser.add_argument("--data-dir", type=Path, required=True)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--results-json", type=Path, default=None)
  parser.add_argument(
    "--out-dir",
    type=Path,
    default=None,
    help="Default: <checkpoint_dir>/tta_eval",
  )
  parser.add_argument("--split", choices=["test", "val", "both"], default="test")
  parser.add_argument(
    "--tta-mode",
    choices=["flip", "flip_rot"],
    default="flip_rot",
    help="flip=4 views; flip_rot=8 views (default)",
  )
  parser.add_argument("--batch-size", type=int, default=64)
  parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
  parser.add_argument("--seed", type=int, default=42)
  args = parser.parse_args()

  out_dir = args.out_dir or args.checkpoint.resolve().parent / "tta_eval"
  payload = run_tta_eval(
    data_dir=args.data_dir,
    checkpoint=args.checkpoint,
    results_json=args.results_json,
    split=args.split,
    tta_mode=args.tta_mode,
    batch_size=args.batch_size,
    device_pref=args.device,
    seed=args.seed,
    out_dir=out_dir,
  )

  print(f"TTA eval saved: {out_dir / 'summary.json'}", flush=True)
  for name, split in payload["splits"].items():
    print(
      f"  {name}: baseline={split['baseline_accuracy']:.4f}  "
      f"tta={split['tta_accuracy']:.4f}  "
      f"gain={split['tta_gain_pp']:+.2f}pp",
      flush=True,
    )


if __name__ == "__main__":
  main()

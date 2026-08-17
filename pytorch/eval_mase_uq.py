"""MaSE-Net Lite: MC Dropout UQ evaluation (PE, UAUC, UQ confusion matrix).

Does not change training or architecture. Loads an existing best.pt and runs
stochastic forward passes with Dropout enabled at test time.

Example:
  python pytorch/eval_mase_uq.py \\
    --data-dir /path/to/deepyeast_5pct \\
    --checkpoint /path/to/checkpoints/abl_min05/best.pt \\
    --mc-samples 30 \\
    --out-dir /path/to/checkpoints/abl_min05/uq_mc_dropout
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
  sys.path.insert(0, str(_HERE))

from dataset import make_loaders
from mase_lite_net import (
  MASE_LITE_DEFAULT,
  MaSELiteConfig,
  MaSELiteNet,
  config_to_dict,
  enable_mc_dropout,
  predictive_entropy,
)


def set_seed(seed: int) -> None:
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def resolve_device(pref: str = "auto") -> torch.device:
  pref = (pref or "auto").lower()
  if pref == "cuda":
    if not torch.cuda.is_available():
      raise RuntimeError("--device cuda requested but CUDA is unavailable")
    return torch.device("cuda")
  if pref == "mps":
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
      raise RuntimeError("--device mps requested but MPS is unavailable")
    return torch.device("mps")
  if pref == "cpu":
    return torch.device("cpu")
  if pref != "auto":
    raise ValueError(f"Unknown --device {pref!r} (use auto|cuda|mps|cpu)")
  if torch.cuda.is_available():
    return torch.device("cuda")
  if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    return torch.device("mps")
  return torch.device("cpu")


def load_checkpoint(model: MaSELiteNet, path: Path) -> None:
  path = Path(path)
  if not path.exists():
    raise FileNotFoundError(path)
  size = path.stat().st_size
  if size < 100_000:
    raise RuntimeError(
      f"Checkpoint too small ({size} bytes): {path}\n"
      "Likely a Git LFS pointer. Run: git lfs install && git lfs pull"
    )
  try:
    state = torch.load(path, map_location="cpu", weights_only=True)
  except TypeError:
    state = torch.load(path, map_location="cpu")
  if isinstance(state, dict) and "state_dict" in state:
    state = state["state_dict"]
  missing, unexpected = model.load_state_dict(state, strict=False)
  if missing:
    print(f"Warning: missing keys ({len(missing)}): {missing[:5]}...", flush=True)
  if unexpected:
    print(
      f"Warning: unexpected keys ({len(unexpected)}): {unexpected[:5]}...",
      flush=True,
    )


def config_from_results_json(results_path: Path | None) -> MaSELiteConfig | None:
  if results_path is None or not results_path.exists():
    return None
  with results_path.open() as f:
    data = json.load(f)
  cfg = data.get("config") or {}
  base = config_to_dict(MASE_LITE_DEFAULT)
  # Only keep keys that MaSELiteConfig knows
  known = {k: cfg[k] for k in base if k in cfg}
  return MaSELiteConfig(**{**base, **known})


def roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
  """Binary ROC AUC without sklearn. Returns nan if only one class present."""
  y_true = np.asarray(y_true).astype(np.int32)
  y_score = np.asarray(y_score).astype(np.float64)
  n_pos = int((y_true == 1).sum())
  n_neg = int((y_true == 0).sum())
  if n_pos == 0 or n_neg == 0:
    return float("nan")
  order = np.argsort(-y_score, kind="mergesort")
  y_true = y_true[order]
  tps = np.cumsum(y_true == 1)
  fps = np.cumsum(y_true == 0)
  tpr = tps / n_pos
  fpr = fps / n_neg
  # prepend (0,0)
  tpr = np.concatenate([[0.0], tpr])
  fpr = np.concatenate([[0.0], fpr])
  try:
    area = float(np.trapezoid(tpr, fpr))
  except AttributeError:
    area = float(np.trapz(tpr, fpr))
  return area


def uq_confusion_at_threshold(
  pe: np.ndarray,
  correct: np.ndarray,
  tau: float,
) -> dict[str, float]:
  """PE < tau → certain; PE >= tau → uncertain."""
  certain = pe < tau
  uncertain = ~certain
  corr = correct.astype(bool)
  wrong = ~corr

  tc = int((corr & certain).sum())
  fu = int((corr & uncertain).sum())
  fc = int((wrong & certain).sum())
  tu = int((wrong & uncertain).sum())
  total = max(tc + fu + fc + tu, 1)

  def _safe(num: int, den: int) -> float:
    return float(num / den) if den > 0 else float("nan")

  return {
    "tau": float(tau),
    "TC": tc,
    "FU": fu,
    "FC": fc,
    "TU": tu,
    "UAcc": _safe(tc + tu, total),
    "USen": _safe(tu, tu + fc),
    "USpe": _safe(tc, tc + fu),
    "UPre": _safe(tu, tu + fu),
  }


def mc_predict_loader(
  model: MaSELiteNet,
  loader: DataLoader,
  device: torch.device,
  mc_samples: int,
) -> dict[str, np.ndarray]:
  """Return per-image y_true, y_pred, pe, max_prob, correct."""
  enable_mc_dropout(model)
  y_true_all: list[np.ndarray] = []
  y_pred_all: list[np.ndarray] = []
  pe_all: list[np.ndarray] = []
  max_prob_all: list[np.ndarray] = []

  with torch.inference_mode():
    for images, labels in loader:
      images = images.to(device, non_blocking=True)
      labels = labels.to(device, non_blocking=True)
      # Accumulate mean probability across MC samples
      probs_sum: torch.Tensor | None = None
      for _ in range(mc_samples):
        # train=False → deterministic mask; Dropout still on via enable_mc_dropout
        log_probs = model(images, train=False, return_details=False)
        assert isinstance(log_probs, torch.Tensor)
        probs = log_probs.exp()
        probs_sum = probs if probs_sum is None else probs_sum + probs
      assert probs_sum is not None
      mu = probs_sum / float(mc_samples)
      pe = predictive_entropy(mu)
      y_pred = mu.argmax(dim=-1)
      max_p = mu.max(dim=-1).values

      y_true_all.append(labels.cpu().numpy())
      y_pred_all.append(y_pred.cpu().numpy())
      pe_all.append(pe.cpu().numpy())
      max_prob_all.append(max_p.cpu().numpy())

  y_true = np.concatenate(y_true_all)
  y_pred = np.concatenate(y_pred_all)
  pe = np.concatenate(pe_all)
  max_prob = np.concatenate(max_prob_all)
  correct = (y_true == y_pred).astype(np.int32)
  return {
    "y_true": y_true,
    "y_pred": y_pred,
    "pe": pe,
    "max_prob": max_prob,
    "correct": correct,
  }


def summarize_split(
  arrays: dict[str, np.ndarray],
  thresholds: list[float],
  num_classes: int,
) -> dict[str, Any]:
  pe = arrays["pe"]
  correct = arrays["correct"]
  incorrect = 1 - correct
  acc = float(correct.mean())
  uauc = roc_auc_score(incorrect, pe)
  # Also: how well max_prob ranks correctness (higher max_prob → more correct)
  # Invert for "uncertainty" view: 1 - max_prob
  uauc_maxprob = roc_auc_score(incorrect, 1.0 - arrays["max_prob"])

  sweep = [uq_confusion_at_threshold(pe, correct, tau) for tau in thresholds]

  def _finite(rows: list[dict], key: str) -> list[dict]:
    return [r for r in rows if r[key] == r[key]]

  # Practical picks (max UPre alone pushes tau→extreme, USen→0)
  best_uacc = max(_finite(sweep, "UAcc") or sweep, key=lambda r: r["UAcc"])
  # Balance UPre & USen (F1-like); require both finite
  balanced = []
  for r in sweep:
    if r["UPre"] == r["UPre"] and r["USen"] == r["USen"] and (r["UPre"] + r["USen"]) > 0:
      f1 = 2 * r["UPre"] * r["USen"] / (r["UPre"] + r["USen"])
      balanced.append({**r, "U_F1": f1})
  best_balance = max(balanced, key=lambda r: r["U_F1"]) if balanced else None
  # Among rows with USen >= 0.5, max UPre (deferral-friendly)
  high_sens = [r for r in _finite(sweep, "UPre") if r["USen"] == r["USen"] and r["USen"] >= 0.5]
  best_upre_at_usen50 = max(high_sens, key=lambda r: r["UPre"]) if high_sens else None

  return {
    "n": int(len(pe)),
    "accuracy": acc,
    "pe_mean": float(pe.mean()),
    "pe_std": float(pe.std()),
    "pe_min": float(pe.min()),
    "pe_max": float(pe.max()),
    "pe_max_theoretical": float(math.log(num_classes)),
    "UAUC": uauc,
    "UAUC_via_1_minus_maxprob": uauc_maxprob,
    "threshold_sweep": sweep,
    "suggested_tau_max_UAcc": best_uacc,
    "suggested_tau_max_U_F1": best_balance,
    "suggested_tau_max_UPre_given_USen_ge_0.5": best_upre_at_usen50,
  }


def write_per_image_csv(path: Path, arrays: dict[str, np.ndarray]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["index", "y_true", "y_pred", "correct", "pe", "max_prob"])
    for i in range(len(arrays["y_true"])):
      writer.writerow(
        [
          i,
          int(arrays["y_true"][i]),
          int(arrays["y_pred"][i]),
          int(arrays["correct"][i]),
          float(arrays["pe"][i]),
          float(arrays["max_prob"][i]),
        ]
      )


def main() -> None:
  parser = argparse.ArgumentParser(
    description="MC Dropout UQ evaluation for MaSE-Net Lite"
  )
  parser.add_argument("--data-dir", type=Path, required=True)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument(
    "--device",
    type=str,
    default="auto",
    choices=["auto", "cuda", "mps", "cpu"],
    help="auto = CUDA > MPS > CPU",
  )
  parser.add_argument(
    "--results-json",
    type=Path,
    default=None,
    help="Optional results.json next to checkpoint (to restore config)",
  )
  parser.add_argument("--out-dir", type=Path, default=None)
  parser.add_argument("--mc-samples", type=int, default=30)
  parser.add_argument("--batch-size", type=int, default=64)
  parser.add_argument("--num-workers", type=int, default=0)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--split", choices=["test", "val", "both"], default="both")
  parser.add_argument("--top-k", type=int, default=None)
  parser.add_argument("--soft-alpha", type=float, default=None)
  parser.add_argument("--branch-dropout", type=float, default=None)
  parser.add_argument("--head-dropout", type=float, default=None)
  parser.add_argument(
    "--min-ensemble-weight",
    type=float,
    default=None,
    help="Override config floor (use 0.05 for v3, 0 for v2)",
  )
  parser.add_argument(
    "--n-thresholds",
    type=int,
    default=21,
    help="Number of PE thresholds from 0 to ln(C)",
  )
  args = parser.parse_args()

  set_seed(args.seed)
  device = resolve_device(args.device)
  data_dir = args.data_dir.resolve()
  ckpt = args.checkpoint.resolve()
  results_json = args.results_json
  if results_json is None:
    cand = ckpt.parent / "results.json"
    results_json = cand if cand.exists() else None

  out_dir = (
    args.out_dir.resolve()
    if args.out_dir is not None
    else (ckpt.parent / "uq_mc_dropout")
  )
  out_dir.mkdir(parents=True, exist_ok=True)

  config = config_from_results_json(results_json) or MaSELiteConfig(
    **config_to_dict(MASE_LITE_DEFAULT)
  )
  # CLI overrides
  if args.top_k is not None:
    config.top_k_patches = args.top_k
  if args.soft_alpha is not None:
    config.soft_mask_alpha = args.soft_alpha
  if args.branch_dropout is not None:
    config.branch_dropout = args.branch_dropout
  if args.head_dropout is not None:
    config.head_dropout = args.head_dropout
  if args.min_ensemble_weight is not None:
    config.min_ensemble_weight = args.min_ensemble_weight

  model = MaSELiteNet(config).to(device)
  load_checkpoint(model, ckpt)
  print(f"Loaded {ckpt}", flush=True)
  print(f"Device={device}  MC samples={args.mc_samples}", flush=True)
  print(f"Config: {config_to_dict(config)}", flush=True)

  _, val_loader, test_loader = make_loaders(
    data_dir,
    batch_size=args.batch_size,
    augment=False,
    strong_augment=False,
    num_workers=args.num_workers,
  )

  pe_max = math.log(config.num_classes)
  thresholds = [
    float(x) for x in np.linspace(0.0, pe_max, args.n_thresholds)
  ]

  splits: dict[str, DataLoader] = {}
  if args.split in ("val", "both"):
    splits["val"] = val_loader
  if args.split in ("test", "both"):
    splits["test"] = test_loader

  summary: dict[str, Any] = {
    "method": "mase_lite_mc_dropout",
    "checkpoint": str(ckpt),
    "data_dir": str(data_dir),
    "mc_samples": args.mc_samples,
    "seed": args.seed,
    "config": config_to_dict(config),
    "device": str(device),
    "note": (
      "PE = predictive entropy of mean MC softmax. "
      "UAUC = ROC AUC of PE ranking incorrect predictions. "
      "UQ matrix: PE < tau → certain; PE >= tau → uncertain. "
      "Architecture unchanged; Dropout re-enabled at test time only."
    ),
    "splits": {},
  }

  for split_name, loader in splits.items():
    print(f"\n=== MC Dropout on {split_name} (T={args.mc_samples}) ===", flush=True)
    arrays = mc_predict_loader(model, loader, device, args.mc_samples)
    write_per_image_csv(out_dir / f"per_image_{split_name}.csv", arrays)
    split_summary = summarize_split(arrays, thresholds, config.num_classes)
    summary["splits"][split_name] = split_summary
    print(
      f"  n={split_summary['n']}  acc={split_summary['accuracy']:.4f}  "
      f"UAUC={split_summary['UAUC']:.4f}  "
      f"PE mean={split_summary['pe_mean']:.4f} "
      f"(max theoretical ln({config.num_classes})={pe_max:.4f})",
      flush=True,
    )
    sug = split_summary.get("suggested_tau_max_U_F1")
    if sug is not None:
      print(
        f"  suggested tau (max U_F1): {sug['tau']:.4f}  "
        f"U_F1={sug['U_F1']:.4f} UPre={sug['UPre']:.4f} "
        f"USen={sug['USen']:.4f} USpe={sug['USpe']:.4f} UAcc={sug['UAcc']:.4f}",
        flush=True,
      )

  with (out_dir / "summary.json").open("w") as f:
    json.dump(summary, f, indent=2)
  print(f"\nWrote {out_dir / 'summary.json'}", flush=True)
  print(f"Wrote per-image CSVs under {out_dir}", flush=True)


if __name__ == "__main__":
  main()

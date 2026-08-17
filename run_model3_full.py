#!/usr/bin/env python3
"""Full-data Model 3: Phase2.5 logits→RF (Model 2) then blend with MaSE TTA.

  p = alpha * p_MaSE_TTA + (1 - alpha) * p_RF
  alpha selected on val only (ties → median of val plateau).

Run from repo root after you have deepyeast_full + Phase 2.5 best.pt:

  python run_model3_full.py \\
    --data-dir /path/to/deepyeast_full \\
    --phase25-dir /path/to/deepyeast_full/checkpoints/mase_lite_full_v6_phase25
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "pytorch"))

from dataset import make_loaders  # noqa: E402
from eval_mase_tta import forward_tta, tta_views  # noqa: E402
from eval_mase_uq import (  # noqa: E402
  config_from_results_json,
  load_checkpoint,
  resolve_device,
  set_seed,
)
from mase_lite_net import MASE_LITE_DEFAULT, MaSELiteNet  # noqa: E402
from mase_trad_net import build_stack_features  # noqa: E402


def acc(p: np.ndarray, y: np.ndarray) -> float:
  return float((p.argmax(axis=1) == y).mean())


def align_rf_proba(rf, p_rf: np.ndarray) -> np.ndarray:
  if hasattr(rf, "classes_") and not np.array_equal(
    rf.classes_, np.arange(len(rf.classes_))
  ):
    mapped = np.zeros_like(p_rf)
    for j, c in enumerate(rf.classes_):
      mapped[:, int(c)] = p_rf[:, j]
    return mapped
  return p_rf


@torch.inference_mode()
def extract_split(model, loader, device, *, feature_mode: str, use_tta: bool):
  xs, ys = [], []
  n_batches = 0
  t0 = time.time()
  for images, labels in loader:
    images = images.to(device)
    if use_tta:
      view_feats = []
      for view in tta_views(images, "flip_rot"):
        _, details = model(view, train=False, return_details=True)
        f, _ = build_stack_features(details, mode=feature_mode, use_disagreement=True)
        view_feats.append(f)
      feats = torch.stack(view_feats, dim=0).mean(dim=0)
    else:
      _, details = model(images, train=False, return_details=True)
      feats, _ = build_stack_features(details, mode=feature_mode, use_disagreement=True)
    xs.append(feats.cpu().numpy())
    ys.append(labels.numpy())
    n_batches += 1
    if n_batches % 20 == 0:
      print(f"    batches={n_batches}  elapsed={time.time() - t0:.0f}s", flush=True)
  return (
    np.concatenate(xs, axis=0).astype(np.float64),
    np.concatenate(ys, axis=0).astype(np.int64),
  )


@torch.inference_mode()
def collect_mase_tta_probs(model, loader, device, tta_mode: str):
  ps, ys = [], []
  n_batches = 0
  t0 = time.time()
  for images, labels in loader:
    images = images.to(device)
    p = forward_tta(model, images, tta_mode).cpu().numpy()
    ps.append(p)
    ys.append(labels.numpy())
    n_batches += 1
    if n_batches % 20 == 0:
      print(f"    batches={n_batches}  elapsed={time.time() - t0:.0f}s", flush=True)
  return (
    np.concatenate(ps, axis=0).astype(np.float64),
    np.concatenate(ys, axis=0).astype(np.int64),
  )


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--data-dir", type=Path, required=True)
  p.add_argument(
    "--phase25-dir",
    type=Path,
    default=None,
    help="Folder with best.pt (default: DATA/checkpoints/mase_lite_full_v6_phase25)",
  )
  p.add_argument("--out-dir", type=Path, default=None)
  p.add_argument("--device", default="auto")
  p.add_argument("--batch-size", type=int, default=64)
  p.add_argument("--num-workers", type=int, default=0)
  p.add_argument("--seed", type=int, default=42)
  p.add_argument("--feature-mode", default="logits")
  p.add_argument("--no-tta-features", action="store_true")
  p.add_argument("--force", action="store_true", help="Overwrite existing results.json")
  return p.parse_args()


def main() -> None:
  args = parse_args()
  data_dir = args.data_dir.resolve()
  phase25_dir = (
    args.phase25_dir.resolve()
    if args.phase25_dir is not None
    else data_dir / "checkpoints" / "mase_lite_full_v6_phase25"
  )
  phase25_ckpt = phase25_dir / "best.pt"
  phase25_results = phase25_dir / "results.json"
  tta_summary = phase25_dir / "tta_eval" / "summary.json"
  out_dir = (
    args.out_dir.resolve()
    if args.out_dir is not None
    else data_dir / "checkpoints" / "mase_model3_phase25_tta_rf_blend_full"
  )
  model2_dir = out_dir / "model2_rf"
  seed = args.seed
  feature_mode = args.feature_mode
  use_tta_features = not args.no_tta_features
  alphas = [i / 20 for i in range(21)]

  if not phase25_ckpt.exists():
    raise SystemExit(f"missing Phase2.5 checkpoint: {phase25_ckpt}")
  if out_dir.exists() and (out_dir / "results.json").exists() and not args.force:
    raise SystemExit(f"refusing overwrite: {out_dir}/results.json (pass --force)")

  out_dir.mkdir(parents=True, exist_ok=True)
  model2_dir.mkdir(parents=True, exist_ok=True)
  set_seed(seed)
  device = resolve_device(args.device)
  print(f"device={device}  data={data_dir}", flush=True)
  print(f"phase25={phase25_ckpt}", flush=True)
  print(f"out={out_dir}", flush=True)

  cfg = config_from_results_json(phase25_results if phase25_results.exists() else None)
  cfg = cfg or MASE_LITE_DEFAULT
  model = MaSELiteNet(cfg).to(device).eval()
  load_checkpoint(model, phase25_ckpt)
  print(f"loaded {phase25_ckpt}", flush=True)

  train_loader, val_loader, test_loader = make_loaders(
    data_dir,
    batch_size=args.batch_size,
    augment=False,
    strong_augment=False,
    num_workers=args.num_workers,
  )

  feat_paths = {s: model2_dir / f"features_{s}.npz" for s in ("train", "val", "test")}
  if all(p.exists() for p in feat_paths.values()):
    print("=== Stage 1: reuse existing Model 2 features ===", flush=True)
  else:
    print("=== Stage 1: extract TTA stack features (Model 2) ===", flush=True)
    t_feat = time.time()
    for name, loader in [
      ("train", train_loader),
      ("val", val_loader),
      ("test", test_loader),
    ]:
      print(f"  extracting {name} ...", flush=True)
      X, y = extract_split(
        model,
        loader,
        device,
        feature_mode=feature_mode,
        use_tta=use_tta_features,
      )
      np.savez_compressed(feat_paths[name], X=X, y=y)
      print(f"  {name}: X={X.shape} y={y.shape}", flush=True)
    meta = {
      "feature_mode": feature_mode,
      "use_tta_features": use_tta_features,
      "tta_mode": "flip_rot" if use_tta_features else None,
      "checkpoint": str(phase25_ckpt),
      "seed": seed,
    }
    with (model2_dir / "feature_meta.json").open("w") as f:
      json.dump(meta, f, indent=2)
    print(f"feature extraction done in {(time.time() - t_feat) / 60:.1f} min", flush=True)

  rf_path = model2_dir / "rf_best.joblib"
  model2_results_path = model2_dir / "results.json"
  if rf_path.exists() and model2_results_path.exists():
    print("=== Stage 2: reuse existing RF ===", flush=True)
    rf = joblib.load(rf_path)
    model2_summary = json.load(open(model2_results_path))
  else:
    print("=== Stage 2: Random Forest grid ===", flush=True)
    tr = np.load(feat_paths["train"])
    va = np.load(feat_paths["val"])
    te = np.load(feat_paths["test"])
    Xtr, ytr = tr["X"], tr["y"]
    Xva, yva = va["X"], va["y"]
    Xte, yte = te["X"], te["y"]
    rows = []
    best = None
    for n in [100, 200, 300, 1000]:
      clf = RandomForestClassifier(n_estimators=n, n_jobs=-1, random_state=seed)
      t0 = time.time()
      clf.fit(Xtr, ytr)
      val_acc = float((clf.predict(Xva) == yva).mean())
      test_acc = float((clf.predict(Xte) == yte).mean())
      row = {"n_estimators": n, "val_accuracy": val_acc, "test_accuracy": test_acc}
      rows.append(row)
      print(
        f"  RF n={n:4d}  val={val_acc:.4f}  test={test_acc:.4f}  "
        f"({time.time() - t0:.0f}s)",
        flush=True,
      )
      if best is None or val_acc > best["val_accuracy"]:
        best = {**row, "clf": clf}
    assert best is not None
    joblib.dump(best["clf"], rf_path)
    model2_summary = {
      "method": "mase_model2_phase25_tta_rf_full",
      "backbone": str(phase25_ckpt),
      "feature_mode": feature_mode,
      "use_tta_features": use_tta_features,
      "seed": seed,
      "grid": rows,
      "best_n_estimators": best["n_estimators"],
      "best_val_accuracy": best["val_accuracy"],
      "test_accuracy": best["test_accuracy"],
    }
    with model2_results_path.open("w") as f:
      json.dump(model2_summary, f, indent=2)
    rf = best["clf"]
    print(
      f"  Model 2 best n={best['n_estimators']}  "
      f"val={best['val_accuracy']:.4f}  test={best['test_accuracy']:.4f}",
      flush=True,
    )

  print("=== Stage 3: MaSE TTA probs (val/test) ===", flush=True)
  packs: dict[str, dict[str, np.ndarray]] = {}
  for name, loader in [("val", val_loader), ("test", test_loader)]:
    prob_path = out_dir / f"probs_{name}.npz"
    if prob_path.exists():
      z = np.load(prob_path)
      packs[name] = {"p_mase": z["p_mase"], "p_rf": z["p_rf"], "y": z["y"]}
      print(f"  reused {prob_path.name}", flush=True)
      continue
    print(f"  MaSE TTA → {name} ...", flush=True)
    p_mase, y = collect_mase_tta_probs(model, loader, device, "flip_rot")
    feat = np.load(feat_paths[name])
    if not np.array_equal(feat["y"], y):
      raise SystemExit(f"{name}: feature labels != TTA labels (order mismatch)")
    p_rf = align_rf_proba(rf, rf.predict_proba(feat["X"]).astype(np.float64))
    packs[name] = {"p_mase": p_mase, "p_rf": p_rf, "y": y}
    np.savez_compressed(prob_path, p_mase=p_mase, p_rf=p_rf, y=y)
    print(
      f"  {name}: n={len(y)}  MaSE-TTA={acc(p_mase, y):.4f}  RF={acc(p_rf, y):.4f}",
      flush=True,
    )

  print("=== Stage 4: sweep alpha on val ===", flush=True)
  rows = []
  for a in alphas:
    p_va = a * packs["val"]["p_mase"] + (1.0 - a) * packs["val"]["p_rf"]
    p_te = a * packs["test"]["p_mase"] + (1.0 - a) * packs["test"]["p_rf"]
    row = {
      "alpha": a,
      "val_accuracy": acc(p_va, packs["val"]["y"]),
      "test_accuracy": acc(p_te, packs["test"]["y"]),
    }
    rows.append(row)
    print(
      f"  α={a:.2f}  val={row['val_accuracy']:.4f}  test={row['test_accuracy']:.4f}",
      flush=True,
    )

  best_val = max(r["val_accuracy"] for r in rows)
  plateau = [r for r in rows if r["val_accuracy"] == best_val]
  best = plateau[len(plateau) // 2]
  mase_tta_test = acc(packs["test"]["p_mase"], packs["test"]["y"])
  rf_test = acc(packs["test"]["p_rf"], packs["test"]["y"])
  summary = {
    "method": "mase_model3_phase25_tta_rf_blend_full",
    "formula": "p = alpha * p_MaSE_TTA + (1-alpha) * p_RF",
    "selection_rule": "max val_accuracy; ties → median alpha in plateau",
    "data_dir": str(data_dir),
    "backbone": str(phase25_ckpt),
    "rf_path": str(rf_path),
    "tta_mode": "flip_rot",
    "feature_mode": feature_mode,
    "use_tta_features": use_tta_features,
    "seed": seed,
    "alphas": alphas,
    "grid": rows,
    "val_plateau_alphas": [r["alpha"] for r in plateau],
    "best_alpha": best["alpha"],
    "best_val_accuracy": best["val_accuracy"],
    "test_accuracy": best["test_accuracy"],
    "mase_tta_test": mase_tta_test,
    "rf_test": rf_test,
    "delta_vs_mase_tta_pp": 100.0 * (best["test_accuracy"] - mase_tta_test),
    "delta_vs_rf_pp": 100.0 * (best["test_accuracy"] - rf_test),
    "model2": {
      "best_n_estimators": model2_summary.get("best_n_estimators"),
      "test_accuracy": model2_summary.get("test_accuracy"),
    },
  }
  if tta_summary.exists():
    summary["mase_phase25_tta_test_reported"] = json.load(open(tta_summary)).get(
      "test_tta_accuracy"
    )

  with (out_dir / "results.json").open("w") as f:
    json.dump(summary, f, indent=2)

  print("=== DONE Model 3 full ===", flush=True)
  print(
    f"best α={best['alpha']:.2f}  val={best['val_accuracy']:.4f}  "
    f"test={best['test_accuracy']:.4f}",
    flush=True,
  )
  print(
    f"vs MaSE+TTA {mase_tta_test:.4f}  Δ={summary['delta_vs_mase_tta_pp']:+.2f} pp",
    flush=True,
  )
  print(f"vs RF       {rf_test:.4f}  Δ={summary['delta_vs_rf_pp']:+.2f} pp", flush=True)
  print(f"saved {out_dir}", flush=True)


if __name__ == "__main__":
  main()

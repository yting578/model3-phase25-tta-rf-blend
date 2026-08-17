# Model 3 full-data (Phase2.5 TTA ∥ RF blend)

Self-contained **full DeepYeast** counterpart of the 5% `Model3_Phase25_TTA_RF_Blend.ipynb`.

On a second machine you typically **do not** have the 5% RF, so this run **trains Model 2 first** (frozen Phase 2.5 logits + TTA feature mean → Random Forest), then blends:

```text
p = α · p_MaSE_TTA + (1-α) · p_RF
```

α is chosen on **val only** (ties → median of the val plateau).

## Files

| File | Role |
|------|------|
| [`Model3_Phase25_TTA_RF_Blend_Full.ipynb`](Model3_Phase25_TTA_RF_Blend_Full.ipynb) | Interactive full-data notebook |
| [`run_model3_full.py`](run_model3_full.py) | Unattended GPU/CLI runner (recommended) |

## Prerequisites

1. Clone this repo, `git lfs pull`, `pip install -r requirements.txt`.
2. Full data (`train 65k / val 12.5k / test 12.5k`):

```bash
python prepare_deepyeast_subset.py --fraction 1.0 --out-dir deepyeast_full --seed 42
```

3. A trained **v6 Phase 2.5** checkpoint at:

```text
deepyeast_full/checkpoints/mase_lite_full_v6_phase25/best.pt
```

See [`V6_PHASE25_RUN.md`](V6_PHASE25_RUN.md). Full Phase2.5+TTA reference is **~89.82%**.

**GPU strongly recommended.** Stage 1 does TTA×7 forwards on 90k images.

## Run

From repo root (edit `--data-dir` if needed):

```bash
python run_model3_full.py \
  --data-dir /path/to/deepyeast_full \
  --phase25-dir /path/to/deepyeast_full/checkpoints/mase_lite_full_v6_phase25 \
  --device cuda
```

Notebook: open `Model3_Phase25_TTA_RF_Blend_Full.ipynb` with cwd = repo root, or set `MASE_REPO` / `MASE_DATA`.

Interrupted runs resume if `features_*.npz`, `rf_best.joblib`, or `probs_*.npz` already exist.

## Outputs

```text
deepyeast_full/checkpoints/mase_model3_phase25_tta_rf_blend_full/
  model2_rf/results.json     # RF test acc, best n_estimators
  results.json               # blend test_accuracy, best_alpha, deltas
```

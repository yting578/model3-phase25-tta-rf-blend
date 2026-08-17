# Model 3 full-data (Phase2.5 TTA ∥ RF blend)

冻住 v6 Phase 2.5 → 先训 RF（Model 2）→ 再与 MaSE TTA 概率融合：

```text
p = α · p_MaSE_TTA + (1-α) · p_RF
```

α 只在 **val** 上选（并列取 plateau 中位 α）。全量 Phase2.5+TTA 对照约 **89.82%**。

## 仓库里有什么 / 没有什么

**有（clone 即可）：** 跑 Model 3 的代码。

| 文件 | 作用 |
|------|------|
| `run_model3_full.py` | 命令行（推荐） |
| `Model3_Phase25_TTA_RF_Blend_Full.ipynb` | notebook |
| `pytorch/` | MaSE 网络 + 数据加载 + TTA |
| `prepare_deepyeast_subset.py` | 下载全量 DeepYeast |
| `requirements.txt` | Python 依赖 |

**没有（另一台电脑自己准备，不要指望 GitHub）：**

- 全量图像 `deepyeast_full/`（约 9 万张）
- Phase 2.5 权重 `mase_lite_full_v6_phase25/best.pt`（需事先训好或从 GPU 机器拷过来）

## 另一台电脑

```bash
git clone https://github.com/yting578/model3-phase25-tta-rf-blend.git
cd model3-phase25-tta-rf-blend
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# GPU: pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### 1) 全量数据（若还没有）

```bash
python prepare_deepyeast_subset.py --fraction 1.0 --out-dir deepyeast_full --seed 42
```

应看到 train 65,000 / val 12,500 / test 12,500。

### 2) Phase 2.5 checkpoint

把已有的 `best.pt`（以及旁边的 `results.json` 更好）放到：

```text
deepyeast_full/checkpoints/mase_lite_full_v6_phase25/best.pt
```

本仓库**不训练** Phase 2.5。若另一台电脑还没有这份权重，需要先按 MaSE 的 `V6_PHASE25_RUN.md` 训完再跑本脚本。

### 3) 跑 Model 3

```bash
python run_model3_full.py \
  --data-dir ./deepyeast_full \
  --phase25-dir ./deepyeast_full/checkpoints/mase_lite_full_v6_phase25 \
  --device cuda
```

**请用 GPU。** Stage 1 要对约 9 万张图做 TTA×7 前向。中断后若已有 `features_*.npz` / `rf_best.joblib` / `probs_*.npz` 会续跑。

## 输出

```text
deepyeast_full/checkpoints/mase_model3_phase25_tta_rf_blend_full/
  model2_rf/results.json
  results.json
```

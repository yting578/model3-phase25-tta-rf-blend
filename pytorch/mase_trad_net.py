"""MaSE-TradStack: MaSE Lite experts + traditional stacking meta-classifier.

Motivation
----------
MaSE Lite's four progressive heads are complementary (oracle ≫ fused on 10%
pilots). Global mixture weights (4 scalars) under-use that diversity.
MaSE-TradStack keeps the MaSE backbone and replaces / refines the final
decision with a stacking meta-learner — the classic ML pattern of combining
expert probabilities with a linear / tree classifier.

Design (accuracy-oriented)
--------------------------
1. Residual neural stacker (zero-init):
     log p_final = log p_mase + Δ(concat head logits / probs)
   At init Δ≡0 ⇒ identical to MaSE; CE fine-tune can only add corrections.
2. Optional traditional sklearn head (logistic / hist-GBDT) on the same
   stack features, trained on the train split.
3. Val-tuned convex blend:
     p = (1-α) p_mase + α p_trad
   and optional disagreement gate (use trad more when heads disagree).

This module is the model definition + feature builders. Training lives in
``train_mase_trad.py``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from mase_lite_net import MaSELiteConfig, MaSELiteNet


@dataclass
class MaSETradConfig:
  num_classes: int = 12
  feature_mode: str = "logits"  # logits | probs | logits_stats
  hidden: int = 0  # 0 → linear residual; >0 → 1-hidden MLP residual
  dropout: float = 0.1
  use_disagreement: bool = True
  freeze_backbone: bool = True


def config_to_dict(config: MaSETradConfig) -> dict[str, Any]:
  return asdict(config)


def head_disagreement(head_probs: torch.Tensor) -> torch.Tensor:
  """Mean pairwise L1 distance among 4 head softmaxes. Shape (B, 4, C) → (B, 1)."""
  # (B, 4, C)
  b, k, c = head_probs.shape
  assert k == 4
  # average |p_i - p_j| over pairs
  diffs = []
  for i in range(k):
    for j in range(i + 1, k):
      diffs.append((head_probs[:, i] - head_probs[:, j]).abs().mean(dim=-1))
  return torch.stack(diffs, dim=-1).mean(dim=-1, keepdim=True)


def stack_feature_dim(num_classes: int, mode: str, use_disagreement: bool) -> int:
  c = num_classes
  if mode == "logits":
    d = 4 * c + c  # head logits + mase log-probs
  elif mode == "probs":
    d = 4 * c + c
  elif mode == "logits_stats":
    # head logits + mase log-probs + maxprob + entropy + margin
    d = 4 * c + c + 3
  else:
    raise ValueError(f"Unknown feature_mode: {mode!r}")
  if use_disagreement:
    d += 1
  return d


def build_stack_features(
  details: dict[str, Any],
  *,
  mode: str = "logits",
  use_disagreement: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Build meta-classifier features from MaSE ``forward(..., return_details=True)``.

  Returns
  -------
  feats : (B, D)
  mase_log_probs : (B, C)
  """
  branch_logits: list[torch.Tensor] = details["branch_logits"]
  mase_log_probs: torch.Tensor = details["ensemble_log_probs"]
  head_probs = torch.stack(
    [F.softmax(logits, dim=-1) for logits in branch_logits], dim=1
  )
  mase_probs = mase_log_probs.exp()

  parts: list[torch.Tensor] = []
  if mode == "logits":
    parts.append(torch.cat(branch_logits, dim=-1))
    parts.append(mase_log_probs)
  elif mode == "probs":
    parts.append(head_probs.reshape(head_probs.size(0), -1))
    parts.append(mase_probs)
  elif mode == "logits_stats":
    parts.append(torch.cat(branch_logits, dim=-1))
    parts.append(mase_log_probs)
    top2 = torch.topk(mase_probs, k=2, dim=-1).values
    maxp = top2[:, :1]
    margin = (top2[:, 0] - top2[:, 1]).unsqueeze(-1)
    ent = -(mase_probs.clamp_min(1e-8) * mase_probs.clamp_min(1e-8).log()).sum(
      dim=-1, keepdim=True
    )
    parts.extend([maxp, ent, margin])
  else:
    raise ValueError(f"Unknown feature_mode: {mode!r}")

  if use_disagreement:
    parts.append(head_disagreement(head_probs))

  return torch.cat(parts, dim=-1), mase_log_probs


class ResidualStackHead(nn.Module):
  """Zero-init residual correction on MaSE log-probabilities.

  log p' = log p_mase + f(stack_features)
  At initialization f≡0, so predictions match MaSE exactly.
  """

  def __init__(
    self,
    in_dim: int,
    num_classes: int,
    hidden: int = 0,
    dropout: float = 0.1,
  ) -> None:
    super().__init__()
    self.num_classes = num_classes
    if hidden and hidden > 0:
      self.net = nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(hidden, num_classes),
      )
      nn.init.zeros_(self.net[-1].weight)
      nn.init.zeros_(self.net[-1].bias)
    else:
      self.net = nn.Linear(in_dim, num_classes)
      nn.init.zeros_(self.net.weight)
      nn.init.zeros_(self.net.bias)

  def forward(self, feats: torch.Tensor, mase_log_probs: torch.Tensor) -> torch.Tensor:
    delta = self.net(feats)
    # Identity at Δ=0 for already-normalized MaSE log-probs; renorm otherwise.
    scores = mase_log_probs + delta
    return scores - torch.logsumexp(scores, dim=-1, keepdim=True)


class MaSETradNet(nn.Module):
  """MaSE Lite backbone + residual traditional-style stacker."""

  def __init__(
    self,
    backbone: MaSELiteNet,
    trad_config: MaSETradConfig | None = None,
  ) -> None:
    super().__init__()
    self.backbone = backbone
    self.trad_config = trad_config or MaSETradConfig(
      num_classes=backbone.config.num_classes
    )
    cfg = self.trad_config
    in_dim = stack_feature_dim(cfg.num_classes, cfg.feature_mode, cfg.use_disagreement)
    self.stack_head = ResidualStackHead(
      in_dim=in_dim,
      num_classes=cfg.num_classes,
      hidden=cfg.hidden,
      dropout=cfg.dropout,
    )
    if cfg.freeze_backbone:
      for p in self.backbone.parameters():
        p.requires_grad = False
      self.backbone.eval()

  @property
  def config(self) -> MaSELiteConfig:
    return self.backbone.config

  def train(self, mode: bool = True) -> MaSETradNet:
    super().train(mode)
    # Keep backbone in eval when frozen so BN / dropout stay deterministic.
    if self.trad_config.freeze_backbone:
      self.backbone.eval()
    return self

  def forward(
    self,
    x: torch.Tensor,
    train: bool = True,
    return_details: bool = False,
  ) -> torch.Tensor | tuple[torch.Tensor, dict]:
    bb_train = train and not self.trad_config.freeze_backbone
    if self.trad_config.freeze_backbone:
      self.backbone.eval()
    out = self.backbone(x, train=bb_train, return_details=True)
    assert isinstance(out, tuple)
    _, details = out
    feats, mase_log_probs = build_stack_features(
      details,
      mode=self.trad_config.feature_mode,
      use_disagreement=self.trad_config.use_disagreement,
    )
    log_probs = self.stack_head(feats, mase_log_probs)
    if return_details:
      details = dict(details)
      details["stack_feats"] = feats
      details["mase_log_probs"] = mase_log_probs
      details["trad_log_probs"] = log_probs
      return log_probs, details
    return log_probs


def blend_probs(
  p_mase: torch.Tensor,
  p_trad: torch.Tensor,
  alpha: float,
  disagree: torch.Tensor | None = None,
  disagree_gate: float = 0.0,
) -> torch.Tensor:
  """Convex blend; optional extra α when head disagreement is high.

  disagree_gate in [0, 1]: fraction of extra weight given to trad when
  disagreement is at its batch-wise max (uses per-sample disagree in [0,1]
  after sigmoid scaling if raw).
  """
  alpha = float(alpha)
  if disagree is not None and disagree_gate > 0.0:
    # Map disagree → [0, 1] softly; typical L1 disagree is small (~0.05–0.2)
    d = torch.sigmoid(10.0 * (disagree.view(-1) - 0.05))
    a = alpha + disagree_gate * d * (1.0 - alpha)
    a = a.view(-1, 1)
  else:
    a = alpha
  mixed = (1.0 - a) * p_mase + a * p_trad
  return mixed / mixed.sum(dim=-1, keepdim=True).clamp_min(1e-8)

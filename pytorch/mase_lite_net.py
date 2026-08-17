"""MaSE-Net Lite for CPU / 10% pilots (v2: learnable ensemble + stronger reg).

  Masked  — ViT patch selector, hard top-k + soft α + sparsity regularizer
  PLCNN   — VGG / ResNet / DenseNet diversity + SE + higher dropout
  MSMM    — progressive multi-head ensemble with learnable weights / temperature
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from masked_net import (
  MaskedModelConfig,
  PatchRegionSelector,
  apply_spatial_mask,
  mcherry_cell_mask,
  mask_sparsity_loss,
)
from plcnn_triple_net import (
  BRANCH_DIM,
  DenseNetBranch,
  ResNetBranch,
  VGGBranch,
)


@dataclass
class MaSELiteConfig(MaskedModelConfig):
  branch_dropout: float = 0.4
  head_dropout: float = 0.5
  learnable_ensemble: bool = True
  ensemble_temperature: float = 1.0
  min_ensemble_weight: float = 0.0
  id_gate: bool = False
  id_gate_hidden: int = 128


MASE_LITE_DEFAULT = MaSELiteConfig(
  top_k_patches=40,
  soft_mask_alpha=0.5,
  selector_layers=2,
  branch_dropout=0.4,
  head_dropout=0.5,
  learnable_ensemble=True,
  ensemble_temperature=1.0,
  min_ensemble_weight=0.0,
  id_gate=False,
  id_gate_hidden=128,
)


class MaSELiteNet(nn.Module):
  """Masked PLCNN branches + MSMM-style multi-head ensemble."""

  def __init__(self, config: MaSELiteConfig | None = None) -> None:
    super().__init__()
    self.config = config or MASE_LITE_DEFAULT
    cfg = self.config
    self.selector = PatchRegionSelector(cfg)
    self.branch_vgg = VGGBranch(dropout=cfg.branch_dropout)
    self.branch_resnet = ResNetBranch(dropout=cfg.branch_dropout)
    self.branch_densenet = DenseNetBranch(dropout=cfg.branch_dropout)

    self.heads = nn.ModuleList(
      [
        nn.Linear(BRANCH_DIM, cfg.num_classes),
        nn.Linear(BRANCH_DIM * 2, cfg.num_classes),
        nn.Linear(BRANCH_DIM * 3, cfg.num_classes),
        nn.Sequential(
          nn.Linear(BRANCH_DIM * 3, 512),
          nn.ReLU(inplace=True),
          nn.Dropout(cfg.head_dropout),
          nn.Linear(512, cfg.num_classes),
        ),
      ]
    )
    # Learnable log-weights over the 4 heads (softmax → mixture)
    if cfg.learnable_ensemble:
      self.ensemble_logits = nn.Parameter(torch.zeros(4))
    else:
      self.register_buffer("ensemble_logits", torch.zeros(4), persistent=False)
    self.ensemble_temperature = max(float(cfg.ensemble_temperature), 1e-3)

    # Input-dependent gate (soft MoE-style; Jacobs et al. 1991):
    # concat(v,r,d) → 4 head logits (zero-init → uniform). See V8_ID_GATE_RUN.md.
    self.id_gate: nn.Sequential | None = None
    if cfg.id_gate:
      hidden = max(int(cfg.id_gate_hidden), 16)
      self.id_gate = nn.Sequential(
        nn.Linear(BRANCH_DIM * 3, hidden),
        nn.ReLU(inplace=True),
        nn.Dropout(cfg.head_dropout),
        nn.Linear(hidden, 4),
      )
      nn.init.zeros_(self.id_gate[-1].weight)
      nn.init.zeros_(self.id_gate[-1].bias)

  def _mask_input(
    self, x: torch.Tensor, train: bool
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cfg = self.config
    mask, patch_probs = self.selector(x, train=train)
    mcherry_prior = None
    if cfg.mcherry_guided:
      mcherry_prior = mcherry_cell_mask(x, cfg.mcherry_threshold)
    masked_x = apply_spatial_mask(
      x,
      mask,
      mask_gfp_only=cfg.mask_gfp_only,
      soft_alpha=cfg.soft_mask_alpha,
      mcherry_prior=mcherry_prior,
    )
    return masked_x, mask, patch_probs

  def _apply_weight_floor(self, weights: torch.Tensor) -> torch.Tensor:
    """Floor each mixture weight then renormalize. Supports (4,) or (B, 4)."""
    eps = float(self.config.min_ensemble_weight)
    if eps <= 0.0:
      return weights
    n = weights.shape[-1]
    eps = min(eps, (1.0 / n) - 1e-4)
    return weights * (1.0 - n * eps) + eps

  def mixture_weights(
    self, gate_feats: torch.Tensor | None = None
  ) -> torch.Tensor:
    """Softmax ensemble weights with optional floor (anti-collapse).

    Global mode: returns (4,).
    ID-Gate mode (gate_feats = concat(v,r,d)): returns (B, 4).
    """
    if self.id_gate is not None and gate_feats is not None:
      logits = self.id_gate(gate_feats) / self.ensemble_temperature
      weights = F.softmax(logits, dim=-1)
    else:
      weights = F.softmax(self.ensemble_logits / self.ensemble_temperature, dim=0)
    return self._apply_weight_floor(weights)

  def _ensemble_probs(self, branch_logits: list[torch.Tensor]) -> torch.Tensor:
    stacked = torch.stack(
      [F.softmax(logits, dim=-1) for logits in branch_logits], dim=1
    )  # (B, 4, C)
    weights = self.mixture_weights()
    return (stacked * weights.view(1, 4, 1)).sum(dim=1)

  def forward(
    self,
    x: torch.Tensor,
    train: bool = True,
    return_details: bool = False,
  ) -> torch.Tensor | tuple[torch.Tensor, dict]:
    masked_x, mask, patch_probs = self._mask_input(x, train=train)
    v = self.branch_vgg(masked_x)
    r = self.branch_resnet(masked_x)
    d = self.branch_densenet(masked_x)

    feats = [
      v,
      torch.cat([v, r], dim=1),
      torch.cat([v, r, d], dim=1),
      torch.cat([v, r, d], dim=1),
    ]
    branch_logits = [head(feat) for head, feat in zip(self.heads, feats)]
    gate_in = torch.cat([v, r, d], dim=1)
    weights = self.mixture_weights(
      gate_in if self.id_gate is not None else None
    )
    stacked = torch.stack(
      [F.softmax(logits, dim=-1) for logits in branch_logits], dim=1
    )
    if weights.dim() == 1:
      probs = (stacked * weights.view(1, 4, 1)).sum(dim=1)
      weights_log = weights.detach()
    else:
      probs = (stacked * weights.unsqueeze(-1)).sum(dim=1)
      weights_log = weights.detach().mean(dim=0)
    ensemble = torch.log(probs.clamp_min(1e-8))

    if return_details:
      return ensemble, {
        "mask": mask,
        "patch_probs": patch_probs,
        "branch_logits": branch_logits,
        "branch_feats": (v, r, d),
        "ensemble_log_probs": ensemble,
        "ensemble_weights_live": weights,
        "ensemble_weights": weights_log,
      }
    return ensemble


def enable_mc_dropout(model: nn.Module) -> None:
  """Keep BatchNorm in eval; turn Dropout on for MC sampling at test time."""
  model.eval()
  for module in model.modules():
    if isinstance(module, nn.Dropout):
      module.train()


def predictive_entropy(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
  """PE = -Σ μ_c log μ_c over class dim. Accepts (C,) or (B, C)."""
  probs = probs.clamp_min(eps)
  return -(probs * probs.log()).sum(dim=-1)


def _nll_with_label_smoothing(
  log_probs: torch.Tensor,
  labels: torch.Tensor,
  label_smoothing: float = 0.0,
) -> torch.Tensor:
  """NLL on log-probabilities (already log-normalized), with optional LS."""
  if label_smoothing <= 0.0:
    return F.nll_loss(log_probs, labels)
  n_class = log_probs.size(-1)
  with torch.no_grad():
    soft = torch.full_like(log_probs, label_smoothing / (n_class - 1))
    soft.scatter_(1, labels.unsqueeze(1), 1.0 - label_smoothing)
  return -(soft * log_probs).sum(dim=-1).mean()


def mase_lite_legacy_loss(
  details: dict,
  labels: torch.Tensor,
  mask_sparsity_weight: float = 0.0,
  target_mask_fraction: float = 0.6,
  label_smoothing: float = 0.0,
) -> torch.Tensor:
  """v2 loss: mean per-head CE only (ensemble weights stay at 0.25)."""
  losses = [
    F.cross_entropy(logits, labels, label_smoothing=label_smoothing)
    for logits in details["branch_logits"]
  ]
  loss = torch.stack(losses).mean()
  if mask_sparsity_weight > 0.0:
    loss = loss + mask_sparsity_weight * mask_sparsity_loss(
      details["patch_probs"], target_mask_fraction
    )
  return loss


def mase_lite_loss(
  details: dict,
  labels: torch.Tensor,
  mask_sparsity_weight: float = 0.0,
  target_mask_fraction: float = 0.6,
  label_smoothing: float = 0.0,
  aux_head_weight: float = 0.5,
  distill_weight: float = 0.1,
  ensemble_entropy_weight: float = 0.0,
) -> torch.Tensor:
  """Fused-prediction CE (+ aux / KL / entropy anti-collapse).

  Primary term uses the mixture log-probs so ``ensemble_logits`` get gradients.
  Aux head CE keeps individual heads competent; KL pulls heads toward the fused
  teacher (detached). Entropy term pushes mixture weights away from one-hot.
  """
  ensemble_log_probs = details.get("ensemble_log_probs")
  if ensemble_log_probs is None:
    raise KeyError("details must include ensemble_log_probs from forward()")

  loss = _nll_with_label_smoothing(
    ensemble_log_probs, labels, label_smoothing=label_smoothing
  )

  branch_logits: list[torch.Tensor] = details["branch_logits"]
  if aux_head_weight > 0.0:
    head_losses = [
      F.cross_entropy(logits, labels, label_smoothing=label_smoothing)
      for logits in branch_logits
    ]
    loss = loss + aux_head_weight * torch.stack(head_losses).mean()

  if distill_weight > 0.0:
    teacher = ensemble_log_probs.detach().exp().clamp_min(1e-8)
    kl = torch.stack(
      [
        F.kl_div(
          F.log_softmax(logits, dim=-1),
          teacher,
          reduction="batchmean",
        )
        for logits in branch_logits
      ]
    ).mean()
    loss = loss + distill_weight * kl

  if ensemble_entropy_weight > 0.0:
    weights = details.get("ensemble_weights_live")
    if weights is None:
      raise KeyError("details must include ensemble_weights_live for entropy reg")
    # maximize H(w) ≡ minimize -H(w); supports global (4,) or ID-Gate (B, 4)
    log_w = weights.clamp_min(1e-8).log()
    if weights.dim() == 1:
      entropy = -(weights * log_w).sum()
    else:
      entropy = -(weights * log_w).sum(dim=-1).mean()
    loss = loss - ensemble_entropy_weight * entropy

  if mask_sparsity_weight > 0.0:
    loss = loss + mask_sparsity_weight * mask_sparsity_loss(
      details["patch_probs"], target_mask_fraction
    )
  return loss


def load_partial_state(
  model: MaSELiteNet,
  checkpoint_path: Path,
  prefixes: tuple[str, ...] | None = None,
) -> list[str]:
  """Load matching keys from a checkpoint; optional key-prefix filter."""
  checkpoint_path = Path(checkpoint_path)
  if not checkpoint_path.exists():
    return []
  size = checkpoint_path.stat().st_size
  if size < 100_000:
    raise RuntimeError(
      f"Checkpoint too small ({size} bytes): {checkpoint_path}\n"
      "Likely a Git LFS pointer. Run:\n"
      "  git lfs install\n"
      "  git lfs pull\n"
      "See COLLABORATOR_V2_RUN.md"
    )
  try:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
  except TypeError:
    state = torch.load(checkpoint_path, map_location="cpu")
  if isinstance(state, dict) and "state_dict" in state:
    state = state["state_dict"]
  model_state = model.state_dict()
  loaded = []
  for key, value in state.items():
    if prefixes is not None and not any(key.startswith(p) for p in prefixes):
      continue
    if key in model_state and model_state[key].shape == value.shape:
      model_state[key] = value
      loaded.append(key)
  model.load_state_dict(model_state)
  return loaded


def config_to_dict(config: MaSELiteConfig) -> dict:
  return asdict(config)

"""iFAM-inspired masked DeepYeast — PyTorch."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepyeast_net import DeepYeastNet


@dataclass
class MaskedModelConfig:
  num_classes: int = 12
  dropout_rate: float = 0.5
  patch_size: int = 8
  selector_layers: int = 2
  embed_dim: int = 64
  num_heads: int = 4
  mlp_dim: int = 128
  top_k_patches: int | None = None
  mask_gfp_only: bool = False
  image_size: int = 64
  soft_mask_alpha: float = 0.0
  use_soft_probs: bool = False
  mcherry_guided: bool = False
  mcherry_threshold: float = -0.5


def straight_through_mask(probs: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
  hard = (probs > threshold).to(probs.dtype)
  return hard - probs.detach() + probs


def top_k_patch_mask(scores: torch.Tensor, k: int) -> torch.Tensor:
  """Hard top-k patch mask with straight-through gradients. (B, num_patches)."""
  soft = torch.sigmoid(scores)
  _, top_idx = torch.topk(scores, k, dim=-1)
  hard = torch.zeros_like(soft)
  hard.scatter_(1, top_idx, 1.0)
  return hard - soft.detach() + soft


def patch_mask_to_pixels(
  patch_mask: torch.Tensor, patch_size: int
) -> torch.Tensor:
  """Upsample (B, grid, grid) → (B, H, W)."""
  return patch_mask.repeat_interleave(patch_size, dim=1).repeat_interleave(
    patch_size, dim=2
  )


def mcherry_cell_mask(images: torch.Tensor, threshold: float = -0.5) -> torch.Tensor:
  """Binary cell region from mCherry channel; images NCHW in [-1, 1]."""
  return (images[:, 0] > threshold).to(images.dtype)


def apply_spatial_mask(
  images: torch.Tensor,
  mask: torch.Tensor,
  mask_gfp_only: bool = False,
  soft_alpha: float = 0.0,
  mcherry_prior: torch.Tensor | None = None,
) -> torch.Tensor:
  effective = mask
  if mcherry_prior is not None:
    effective = effective * mcherry_prior
  if soft_alpha > 0.0:
    effective = soft_alpha + (1.0 - soft_alpha) * effective

  if mask_gfp_only:
    channel_mask = torch.stack(
      [torch.ones_like(effective), effective],
      dim=1,
    )
    return images * channel_mask
  return images * effective.unsqueeze(1)


class PatchRegionSelector(nn.Module):
  """ViT-style selector: 64×64 NCHW → pixel importance mask."""

  def __init__(self, config: MaskedModelConfig) -> None:
    super().__init__()
    self.config = config
    grid = config.image_size // config.patch_size
    num_patches = grid * grid
    patch_dim = config.patch_size * config.patch_size * 2

    self.patch_proj = nn.Linear(patch_dim, config.embed_dim)
    self.pos_embed = nn.Parameter(torch.randn(1, num_patches, config.embed_dim) * 0.02)
    self.blocks = nn.ModuleList(
      [_SelectorBlock(config) for _ in range(config.selector_layers)]
    )
    self.score_head = nn.Linear(config.embed_dim, 1)

  def forward(
    self, x: torch.Tensor, train: bool = True
  ) -> tuple[torch.Tensor, torch.Tensor]:
    cfg = self.config
    grid = cfg.image_size // cfg.patch_size
    b, _, _, _ = x.shape

    patches = x.view(
      b, 2, grid, cfg.patch_size, grid, cfg.patch_size
    ).permute(0, 2, 4, 3, 5, 1)
    tokens = patches.reshape(b, grid * grid, -1)
    tokens = self.patch_proj(tokens) + self.pos_embed

    for block in self.blocks:
      tokens = block(tokens, train=train)

    patch_scores = self.score_head(tokens).squeeze(-1)
    patch_probs = torch.sigmoid(patch_scores)

    if cfg.top_k_patches is not None:
      patch_flat = top_k_patch_mask(patch_scores, cfg.top_k_patches)
    elif cfg.use_soft_probs:
      patch_flat = patch_probs
    else:
      patch_flat = straight_through_mask(patch_probs)

    patch_mask = patch_flat.view(b, grid, grid)
    pixel_mask = patch_mask_to_pixels(patch_mask, cfg.patch_size)
    return pixel_mask, patch_probs


class _SelectorBlock(nn.Module):
  def __init__(self, config: MaskedModelConfig) -> None:
    super().__init__()
    self.norm1 = nn.LayerNorm(config.embed_dim)
    self.attn = nn.MultiheadAttention(
      config.embed_dim,
      config.num_heads,
      batch_first=True,
    )
    self.norm2 = nn.LayerNorm(config.embed_dim)
    self.mlp = nn.Sequential(
      nn.Linear(config.embed_dim, config.mlp_dim),
      nn.GELU(),
      nn.Linear(config.mlp_dim, config.embed_dim),
    )

  def forward(self, tokens: torch.Tensor, train: bool = True) -> torch.Tensor:
    normed = self.norm1(tokens)
    attn_out, _ = self.attn(
      normed, normed, normed, need_weights=False
    )
    tokens = tokens + attn_out
    tokens = tokens + self.mlp(self.norm2(tokens))
    return tokens


class MaskedDeepYeastNet(nn.Module):
  """Stage 1: region selector. Stage 2: DeepYeastNet on masked input."""

  def __init__(self, config: MaskedModelConfig | None = None) -> None:
    super().__init__()
    self.config = config or MaskedModelConfig()
    self.selector = PatchRegionSelector(self.config)
    self.classifier = DeepYeastNet(
      num_classes=self.config.num_classes,
      dropout_rate=self.config.dropout_rate,
    )

  def forward(
    self,
    x: torch.Tensor,
    train: bool = True,
    return_mask: bool = False,
    mask_mode: int = 0,
  ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cfg = self.config
    mask, patch_probs = self.selector(x, train=train)

    mcherry_prior = None
    if cfg.mcherry_guided:
      mcherry_prior = mcherry_cell_mask(x, cfg.mcherry_threshold)

    def masked(normal_mask: torch.Tensor) -> torch.Tensor:
      return apply_spatial_mask(
        x,
        normal_mask,
        mask_gfp_only=cfg.mask_gfp_only,
        soft_alpha=cfg.soft_mask_alpha,
        mcherry_prior=mcherry_prior,
      )

    if mask_mode == 1:
      masked_x = x
    elif mask_mode == 2:
      masked_x = masked(1.0 - mask)
    else:
      masked_x = masked(mask)

    logits = self.classifier(masked_x, train=train)
    if return_mask:
      return logits, mask, patch_probs
    return logits


def mask_sparsity_loss(
  patch_probs: torch.Tensor, target_fraction: float = 0.6
) -> torch.Tensor:
  mean_cov = patch_probs.mean()
  return (mean_cov - target_fraction) ** 2


# Keras init + hard top-38 (~60% patch coverage) + soft blend α=0.5
V3K60_CONFIG = MaskedModelConfig(
  top_k_patches=38,
  soft_mask_alpha=0.5,
  selector_layers=2,
)


def config_to_dict(config: MaskedModelConfig) -> dict:
  return asdict(config)

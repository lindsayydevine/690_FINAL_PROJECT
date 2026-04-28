"""
Stage-1 decoder: BioPM token sequence -> movement-element representation.

This module reconstructs the aligned x_acc_filt target:
  [patch(32) | pos(1) | axis(1) | len(1) | min(1) | max(1) | dirct(1)]

The model uses separate heads because the target mixes regression and
classification-style fields.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


PATCH_DIM = 32
TARGET_DIM = 38
POS_INDEX = 32
AXIS_INDEX = 33
LEN_INDEX = 34
MIN_INDEX = 35
MAX_INDEX = 36
DIR_INDEX = 37
LENGTH_SCALE = 300.0


class ResidualMLPBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.ff(self.norm(x))


class BioPMStage1Decoder(nn.Module):
    """Decode 64-d contextual BioPM tokens back into x_acc_filt fields."""

    def __init__(self, token_dim: int = 64, hidden_dim: int = 256,
                 depth: int = 3, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(token_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            ResidualMLPBlock(hidden_dim, dropout=dropout) for _ in range(depth)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim)

        self.patch_head = nn.Linear(hidden_dim, PATCH_DIM)
        self.pos_head = nn.Linear(hidden_dim, 1)
        self.axis_head = nn.Linear(hidden_dim, 3)
        self.length_head = nn.Linear(hidden_dim, 1)
        self.min_head = nn.Linear(hidden_dim, 1)
        self.max_head = nn.Linear(hidden_dim, 1)
        self.dir_head = nn.Linear(hidden_dim, 2)

    def forward(self, tokens):
        x = self.input_proj(tokens)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return {
            "patch": self.patch_head(x),
            "pos": self.pos_head(x).squeeze(-1),
            "axis_logits": self.axis_head(x),
            "length": self.length_head(x).squeeze(-1),
            "min_val": self.min_head(x).squeeze(-1),
            "max_val": self.max_head(x).squeeze(-1),
            "dir_logits": self.dir_head(x),
        }


def masked_regression_loss(pred, target, valid_mask, loss_type: str = "smooth_l1"):
    """Apply a regression loss only on non-padded movement-element tokens."""
    if pred.ndim == valid_mask.ndim + 1:
        valid = valid_mask.unsqueeze(-1)
    else:
        valid = valid_mask

    target = torch.nan_to_num(target, nan=0.0)
    if loss_type == "mse":
        loss = F.mse_loss(pred, target, reduction="none")
    else:
        loss = F.smooth_l1_loss(pred, target, reduction="none")

    valid = valid.to(dtype=loss.dtype)
    denom = valid.sum().clamp(min=1.0)
    return (loss * valid).sum() / denom


def masked_classification_loss(logits, target, valid_mask):
    """Cross-entropy over valid tokens only."""
    if valid_mask.sum() == 0:
        return logits.sum() * 0.0
    return F.cross_entropy(logits[valid_mask], target[valid_mask])


def compute_stage1_losses(preds, target_x_acc_filt, valid_mask):
    """Compute the multi-head loss for stage-1 decoder training."""
    patch_target = target_x_acc_filt[:, :, :PATCH_DIM]
    pos_target = target_x_acc_filt[:, :, POS_INDEX]
    axis_target = torch.nan_to_num(target_x_acc_filt[:, :, AXIS_INDEX], nan=0.0).long()
    len_target = torch.nan_to_num(target_x_acc_filt[:, :, LEN_INDEX], nan=0.0) / LENGTH_SCALE
    min_target = target_x_acc_filt[:, :, MIN_INDEX]
    max_target = target_x_acc_filt[:, :, MAX_INDEX]
    dir_target = (torch.nan_to_num(target_x_acc_filt[:, :, DIR_INDEX], nan=-1.0) > 0).long()

    patch_loss = masked_regression_loss(preds["patch"], patch_target, valid_mask)
    pos_loss = masked_regression_loss(preds["pos"], pos_target, valid_mask)
    axis_loss = masked_classification_loss(preds["axis_logits"], axis_target, valid_mask)
    length_loss = masked_regression_loss(preds["length"], len_target, valid_mask)
    min_loss = masked_regression_loss(preds["min_val"], min_target, valid_mask)
    max_loss = masked_regression_loss(preds["max_val"], max_target, valid_mask)
    dir_loss = masked_classification_loss(preds["dir_logits"], dir_target, valid_mask)

    total = (
        1.00 * patch_loss +
        0.25 * pos_loss +
        0.50 * axis_loss +
        0.25 * length_loss +
        0.25 * min_loss +
        0.25 * max_loss +
        0.25 * dir_loss
    )

    with torch.no_grad():
        axis_acc = (
            (preds["axis_logits"].argmax(dim=-1)[valid_mask] == axis_target[valid_mask])
            .float().mean() if valid_mask.any() else torch.tensor(0.0, device=valid_mask.device)
        )
        dir_acc = (
            (preds["dir_logits"].argmax(dim=-1)[valid_mask] == dir_target[valid_mask])
            .float().mean() if valid_mask.any() else torch.tensor(0.0, device=valid_mask.device)
        )

    return {
        "loss": total,
        "patch_loss": patch_loss,
        "pos_loss": pos_loss,
        "axis_loss": axis_loss,
        "length_loss": length_loss,
        "min_loss": min_loss,
        "max_loss": max_loss,
        "dir_loss": dir_loss,
        "axis_acc": axis_acc,
        "dir_acc": dir_acc,
    }


def render_body_prior_from_stage1_preds(preds, valid_mask=None, steps: int = 300,
                                        support_sharpness: float = 10.0):
    """
    Soft differentiable renderer from Stage-1 token predictions to a 300x3 body window.

    This avoids hard rounding of predicted metadata and provides a more faithful
    body-motion prior for downstream Stage-2 reconstruction.
    """
    patch = preds["patch"]
    pos = preds["pos"].clamp(0.0, 1.0)
    axis_prob = torch.softmax(preds["axis_logits"], dim=-1)
    length_samples = F.softplus(preds["length"] * LENGTH_SCALE) + 1.0
    length_samples = length_samples.clamp(min=2.0, max=float(steps))

    if valid_mask is None:
        valid_mask = torch.ones(
            patch.shape[:2], device=patch.device, dtype=torch.bool)

    valid = valid_mask.to(dtype=patch.dtype)
    axis_prob = axis_prob * valid.unsqueeze(-1)

    batch_size, num_tokens, _ = patch.shape
    patch_in = patch.reshape(batch_size * num_tokens, 1, 1, PATCH_DIM)

    time_axis = torch.linspace(
        0.0, float(steps - 1), steps, device=patch.device, dtype=patch.dtype)
    centers = pos * float(steps - 1)
    half_lengths = (length_samples / 2.0).clamp(min=1.0)
    local_coord = (
        time_axis.view(1, 1, steps) - centers.unsqueeze(-1)
    ) / half_lengths.unsqueeze(-1)

    grid = torch.zeros(
        batch_size * num_tokens, 1, steps, 2,
        device=patch.device, dtype=patch.dtype)
    grid[..., 0] = local_coord.reshape(batch_size * num_tokens, 1, steps)

    rendered = F.grid_sample(
        patch_in,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).view(batch_size, num_tokens, steps)

    soft_support = torch.sigmoid(
        support_sharpness * (1.0 - local_coord.abs())
    ) * valid.unsqueeze(-1)

    contrib = (
        rendered.unsqueeze(-1) *
        soft_support.unsqueeze(-1) *
        axis_prob.unsqueeze(2)
    )
    denom = (
        soft_support.unsqueeze(-1) *
        axis_prob.unsqueeze(2)
    ).sum(dim=1).clamp(min=1e-4)

    return contrib.sum(dim=1) / denom


@torch.no_grad()
def assemble_x_acc_filt(preds, valid_mask=None):
    """Convert decoder heads back into an x_acc_filt-style tensor."""
    patch = preds["patch"]
    pos = preds["pos"].unsqueeze(-1)
    axis = preds["axis_logits"].argmax(dim=-1, keepdim=True).float()
    length = (preds["length"] * LENGTH_SCALE).unsqueeze(-1)
    min_val = preds["min_val"].unsqueeze(-1)
    max_val = preds["max_val"].unsqueeze(-1)
    dirct = torch.where(
        preds["dir_logits"].argmax(dim=-1, keepdim=True) > 0,
        torch.ones_like(length),
        -torch.ones_like(length),
    )

    out = torch.cat([patch, pos, axis, length, min_val, max_val, dirct], dim=-1)
    if valid_mask is not None:
        out = out.clone()
        out[~valid_mask] = float("nan")
    return out

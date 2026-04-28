"""
Stage-2 decoder: BioPM token sequence -> window_acc_filt_gravity.

This decoder maps contextual BioPM token embeddings back to a full 10-second
window with 300 time steps and 6 channels:
  [filtered_acc_x, filtered_acc_y, filtered_acc_z,
   gravity_x,      gravity_y,      gravity_z]

Compared with Stage 1, this is a stronger and more ambiguous decoding target,
so the model uses learned per-timestep queries plus cross-attention over the
token sequence instead of a token-aligned prediction head.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


WINDOW_STEPS = 300
WINDOW_CHANNELS = 6
BODY_CHANNELS = slice(0, 3)
GRAVITY_CHANNELS = slice(3, 6)


class FeedForwardBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.ff(self.norm(x))


class CrossAttentionDecoderLayer(nn.Module):
    """Query self-attention + token cross-attention + feed-forward block."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.self_norm = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm_query = nn.LayerNorm(hidden_dim)
        self.cross_norm_memory = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.ff = FeedForwardBlock(hidden_dim, dropout=dropout)

    def forward(self, query, memory, memory_padding_mask=None):
        q_norm = self.self_norm(query)
        self_out, _ = self.self_attn(q_norm, q_norm, q_norm, need_weights=False)
        query = query + self.dropout(self_out)

        q_norm = self.cross_norm_query(query)
        m_norm = self.cross_norm_memory(memory)
        cross_out, _ = self.cross_attn(
            q_norm,
            m_norm,
            m_norm,
            key_padding_mask=memory_padding_mask,
            need_weights=False,
        )
        query = query + self.dropout(cross_out)
        query = self.ff(query)
        return query


class TemporalRefinementBlock(nn.Module):
    """Small residual Conv1d block to smooth timestep-wise decoder outputs."""

    def __init__(self, hidden_dim: int, kernel_size: int, dropout: float):
        super().__init__()
        padding = kernel_size // 2
        self.norm = nn.LayerNorm(hidden_dim)
        self.conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        y = self.norm(x).transpose(1, 2)
        y = self.conv(y).transpose(1, 2)
        return x + y


class BioPMStage2Decoder(nn.Module):
    """
    Decode contextual BioPM tokens into a full 300x6 filtered+gravity window.

    Inputs:
      tokens:     (B, L, 64)
      token_mask: (B, L) bool mask, True for valid/non-padded tokens
    """

    def __init__(
        self,
        token_dim: int = 64,
        hidden_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        query_steps: int = WINDOW_STEPS,
        refine_depth: int = 2,
        kernel_size: int = 9,
        dropout: float = 0.1,
        separate_gravity_branch: bool = False,
        gravity_query_steps: int | None = None,
    ):
        super().__init__()
        self.query_steps = query_steps
        self.separate_gravity_branch = separate_gravity_branch
        self.gravity_query_steps = gravity_query_steps or query_steps

        self.input_proj = nn.Linear(token_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        if self.separate_gravity_branch:
            self.body_query_tokens = nn.Parameter(torch.empty(query_steps, hidden_dim))
            nn.init.trunc_normal_(self.body_query_tokens, std=0.02)

            self.body_query_pos_emb = nn.Parameter(torch.empty(query_steps, hidden_dim))
            nn.init.trunc_normal_(self.body_query_pos_emb, std=0.02)

            self.body_layers = nn.ModuleList([
                CrossAttentionDecoderLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(depth)
            ])

            self.body_refine_blocks = nn.ModuleList([
                TemporalRefinementBlock(
                    hidden_dim=hidden_dim,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
                for _ in range(refine_depth)
            ])

            self.gravity_query_tokens = nn.Parameter(
                torch.empty(self.gravity_query_steps, hidden_dim))
            nn.init.trunc_normal_(self.gravity_query_tokens, std=0.02)

            self.gravity_query_pos_emb = nn.Parameter(
                torch.empty(self.gravity_query_steps, hidden_dim))
            nn.init.trunc_normal_(self.gravity_query_pos_emb, std=0.02)

            self.gravity_layers = nn.ModuleList([
                CrossAttentionDecoderLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(depth)
            ])

            self.gravity_refine_blocks = nn.ModuleList([
                TemporalRefinementBlock(
                    hidden_dim=hidden_dim,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
                for _ in range(max(1, refine_depth))
            ])

            self.body_final_norm = nn.LayerNorm(hidden_dim)
            self.gravity_final_norm = nn.LayerNorm(hidden_dim)
            self.gravity_post_upsample = TemporalRefinementBlock(
                hidden_dim=hidden_dim,
                kernel_size=kernel_size,
                dropout=dropout,
            )
            self.body_head = nn.Linear(hidden_dim, 3)
            self.gravity_head = nn.Linear(hidden_dim, 3)
        else:
            self.query_tokens = nn.Parameter(torch.empty(query_steps, hidden_dim))
            nn.init.trunc_normal_(self.query_tokens, std=0.02)

            self.query_pos_emb = nn.Parameter(torch.empty(query_steps, hidden_dim))
            nn.init.trunc_normal_(self.query_pos_emb, std=0.02)

            self.layers = nn.ModuleList([
                CrossAttentionDecoderLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(depth)
            ])

            self.refine_blocks = nn.ModuleList([
                TemporalRefinementBlock(
                    hidden_dim=hidden_dim,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
                for _ in range(refine_depth)
            ])

            self.final_norm = nn.LayerNorm(hidden_dim)
            self.body_head = nn.Linear(hidden_dim, 3)
            self.gravity_head = nn.Linear(hidden_dim, 3)

    def forward(self, tokens, token_mask=None):
        memory = self.input_norm(self.input_proj(tokens))

        if token_mask is not None:
            memory = memory * token_mask.unsqueeze(-1).to(dtype=memory.dtype)
            memory_padding_mask = ~token_mask.bool()
        else:
            memory_padding_mask = None

        batch_size = memory.shape[0]

        if self.separate_gravity_branch:
            body_query = self.body_query_tokens.unsqueeze(0).expand(batch_size, -1, -1)
            body_query = body_query + self.body_query_pos_emb.unsqueeze(0)
            for layer in self.body_layers:
                body_query = layer(
                    body_query,
                    memory,
                    memory_padding_mask=memory_padding_mask,
                )
            for block in self.body_refine_blocks:
                body_query = block(body_query)
            body_query = self.body_final_norm(body_query)
            body = self.body_head(body_query)

            gravity_query = self.gravity_query_tokens.unsqueeze(0).expand(batch_size, -1, -1)
            gravity_query = gravity_query + self.gravity_query_pos_emb.unsqueeze(0)
            for layer in self.gravity_layers:
                gravity_query = layer(
                    gravity_query,
                    memory,
                    memory_padding_mask=memory_padding_mask,
                )
            for block in self.gravity_refine_blocks:
                gravity_query = block(gravity_query)
            gravity_query = self.gravity_final_norm(gravity_query)
            gravity_query = F.interpolate(
                gravity_query.transpose(1, 2),
                size=self.query_steps,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
            gravity_query = self.gravity_post_upsample(gravity_query)
            gravity = self.gravity_head(gravity_query)
        else:
            query = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)
            query = query + self.query_pos_emb.unsqueeze(0)

            for layer in self.layers:
                query = layer(
                    query,
                    memory,
                    memory_padding_mask=memory_padding_mask,
                )

            for block in self.refine_blocks:
                query = block(query)

            query = self.final_norm(query)
            body = self.body_head(query)
            gravity = self.gravity_head(query)
        window = torch.cat([body, gravity], dim=-1)
        return {
            "body": body,
            "gravity": gravity,
            "window": window,
        }


def regression_loss(pred, target, loss_type: str = "smooth_l1"):
    """Per-window regression loss over dense 300xC targets."""
    if loss_type == "mse":
        return F.mse_loss(pred, target)
    if loss_type == "l1":
        return F.l1_loss(pred, target)
    return F.smooth_l1_loss(pred, target)


def temporal_difference_loss(pred, target, loss_type: str = "smooth_l1"):
    """Regression loss on first temporal differences to encourage waveform shape."""
    pred_delta = pred[:, 1:, :] - pred[:, :-1, :]
    target_delta = target[:, 1:, :] - target[:, :-1, :]
    return regression_loss(pred_delta, target_delta, loss_type=loss_type)


def pooled_sequence_loss(pred, target, kernel_size: int, loss_type: str = "smooth_l1"):
    """Compare low-frequency sequence structure using average pooling."""
    if kernel_size <= 1:
        return regression_loss(pred, target, loss_type=loss_type)
    pred_pool = F.avg_pool1d(
        pred.transpose(1, 2), kernel_size=kernel_size, stride=kernel_size)
    target_pool = F.avg_pool1d(
        target.transpose(1, 2), kernel_size=kernel_size, stride=kernel_size)
    return regression_loss(
        pred_pool.transpose(1, 2),
        target_pool.transpose(1, 2),
        loss_type=loss_type,
    )


def compute_stage2_losses(preds, target_window, body_weight: float = 1.0,
                          gravity_weight: float = 1.0,
                          body_delta_weight: float = 0.0,
                          gravity_delta_weight: float = 0.0,
                          gravity_coarse_weight: float = 0.0,
                          gravity_coarse_kernel: int = 15):
    """
    Compute dense sequence reconstruction losses for Stage 2.

    target_window: (B, 300, 6) = filtered_acc(3) + gravity(3)
    """
    body_target = target_window[:, :, BODY_CHANNELS]
    gravity_target = target_window[:, :, GRAVITY_CHANNELS]

    body_loss = regression_loss(preds["body"], body_target, loss_type="smooth_l1")
    gravity_loss = regression_loss(preds["gravity"], gravity_target, loss_type="smooth_l1")
    body_delta_loss = temporal_difference_loss(
        preds["body"], body_target, loss_type="smooth_l1")
    gravity_delta_loss = temporal_difference_loss(
        preds["gravity"], gravity_target, loss_type="smooth_l1")
    gravity_coarse_loss = pooled_sequence_loss(
        preds["gravity"],
        gravity_target,
        kernel_size=gravity_coarse_kernel,
        loss_type="smooth_l1",
    )

    total = (
        body_weight * body_loss +
        gravity_weight * gravity_loss +
        body_delta_weight * body_delta_loss +
        gravity_delta_weight * gravity_delta_loss +
        gravity_coarse_weight * gravity_coarse_loss
    )

    with torch.no_grad():
        body_diff = preds["body"] - body_target
        gravity_diff = preds["gravity"] - gravity_target
        window_diff = preds["window"] - target_window

        body_rmse = torch.sqrt(torch.mean(body_diff.pow(2)) + 1e-12)
        gravity_rmse = torch.sqrt(torch.mean(gravity_diff.pow(2)) + 1e-12)
        window_rmse = torch.sqrt(torch.mean(window_diff.pow(2)) + 1e-12)
        body_mae = torch.mean(body_diff.abs())
        gravity_mae = torch.mean(gravity_diff.abs())
        window_mae = torch.mean(window_diff.abs())

    return {
        "loss": total,
        "body_loss": body_loss,
        "gravity_loss": gravity_loss,
        "body_delta_loss": body_delta_loss,
        "gravity_delta_loss": gravity_delta_loss,
        "gravity_coarse_loss": gravity_coarse_loss,
        "window_rmse": window_rmse,
        "body_rmse": body_rmse,
        "gravity_rmse": gravity_rmse,
        "window_mae": window_mae,
        "body_mae": body_mae,
        "gravity_mae": gravity_mae,
    }

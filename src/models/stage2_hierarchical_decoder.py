"""
Hierarchical Stage-2 decoder:
BioPM tokens -> Stage-1 ME reconstruction -> rendered body prior -> body residual + gravity.

This decoder uses the already-strong Stage-1 movement-element reconstruction as
an explicit body-motion scaffold. A lightweight temporal decoder then predicts:
  1. a small residual correction for the rendered body-acceleration window
  2. the gravity channels with a dedicated branch
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .stage2_decoder import CrossAttentionDecoderLayer, TemporalRefinementBlock


WINDOW_STEPS = 300
PATCH_DIM = 32
POS_INDEX = 32
AXIS_INDEX = 33
LEN_INDEX = 34


@torch.no_grad()
def render_x_acc_filt_to_body(x_acc_filt: torch.Tensor,
                              valid_mask: torch.Tensor | None = None,
                              steps: int = WINDOW_STEPS) -> torch.Tensor:
    """
    Render token-level x_acc_filt movement elements back into a 300x3 body window.

    This is a simple deterministic renderer:
      - place each ME on its predicted axis
      - center it according to the predicted fractional position
      - stretch/compress the 32-point patch to the predicted length
      - average overlapping contributions per timestep/axis
    """
    device = x_acc_filt.device
    dtype = x_acc_filt.dtype
    x_acc_filt = torch.nan_to_num(x_acc_filt, nan=0.0)

    if valid_mask is None:
        valid_mask = ~torch.isnan(x_acc_filt[:, :, :PATCH_DIM]).any(dim=-1)

    batch_size, _, _ = x_acc_filt.shape
    body = torch.zeros(batch_size, steps, 3, device=device, dtype=dtype)
    counts = torch.zeros_like(body)

    for b in range(batch_size):
        valid_indices = torch.where(valid_mask[b])[0]
        for token_idx in valid_indices.tolist():
            token = x_acc_filt[b, token_idx]
            patch = token[:PATCH_DIM].view(1, 1, PATCH_DIM)

            pos = float(torch.clamp(token[POS_INDEX], 0.0, 1.0).item())
            axis = int(torch.clamp(torch.round(token[AXIS_INDEX]), 0, 2).item())
            length = int(torch.clamp(torch.round(token[LEN_INDEX]), 1, steps).item())

            midpoint = pos * steps
            start = int(round(midpoint - (length / 2.0)))
            start = max(0, min(start, steps - 1))
            end = min(steps, start + length)
            length = max(1, end - start)

            resized = F.interpolate(
                patch,
                size=length,
                mode="linear",
                align_corners=False,
            ).view(-1)

            body[b, start:end, axis] += resized
            counts[b, start:end, axis] += 1.0

    return torch.where(counts > 0, body / counts.clamp(min=1.0), body)


class BioPMHierarchicalStage2Decoder(nn.Module):
    """
    Decode tokens with a rendered Stage-1 body prior as temporal scaffold.

    Inputs:
      tokens:     (B, L, 64)
      body_prior: (B, 300, 3)
      token_mask: (B, L) bool mask, True for valid/non-padded tokens
    """

    def __init__(
        self,
        token_dim: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        num_heads: int = 4,
        refine_depth: int = 1,
        kernel_size: int = 7,
        dropout: float = 0.1,
        residual_scale: float = 0.25,
    ):
        super().__init__()
        self.residual_scale = residual_scale

        self.input_proj = nn.Linear(token_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        self.body_query_proj = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.query_pos_emb = nn.Parameter(torch.empty(WINDOW_STEPS, hidden_dim))
        nn.init.trunc_normal_(self.query_pos_emb, std=0.02)

        self.shared_layers = nn.ModuleList([
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
        self.body_residual_head = nn.Linear(hidden_dim, 3)
        self.gravity_head = nn.Linear(hidden_dim, 3)

    def forward(self, tokens, body_prior, token_mask=None):
        memory = self.input_norm(self.input_proj(tokens))
        if token_mask is not None:
            memory = memory * token_mask.unsqueeze(-1).to(dtype=memory.dtype)
            memory_padding_mask = ~token_mask.bool()
        else:
            memory_padding_mask = None

        query = self.body_query_proj(body_prior) + self.query_pos_emb.unsqueeze(0)
        for layer in self.shared_layers:
            query = layer(
                query,
                memory,
                memory_padding_mask=memory_padding_mask,
            )

        body_hidden = query
        for block in self.body_refine_blocks:
            body_hidden = block(body_hidden)
        body_hidden = self.body_final_norm(body_hidden)
        body_residual = torch.tanh(self.body_residual_head(body_hidden)) * self.residual_scale
        body = body_prior + body_residual

        gravity_hidden = query
        for block in self.gravity_refine_blocks:
            gravity_hidden = block(gravity_hidden)
        gravity_hidden = self.gravity_final_norm(gravity_hidden)
        gravity = self.gravity_head(gravity_hidden)

        window = torch.cat([body, gravity], dim=-1)
        return {
            "body_prior": body_prior,
            "body_residual": body_residual,
            "body": body,
            "gravity": gravity,
            "window": window,
        }

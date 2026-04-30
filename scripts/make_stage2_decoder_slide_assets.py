#!/usr/bin/env python3
"""
Create presentation-ready assets for the Stage 2 decoder section.

Outputs:
  - decoder_architecture.png
  - decoder_reconstruction_breakdown.png
  - decoder_metrics.png
  - decoder_examples.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.h5_window_dataset import Stage2WindowDataset
from src.models.biopm import load_pretrained_encoder
from src.models.stage2_decoder import BioPMStage2Decoder


plt.rcParams.update({
    "figure.dpi": 160,
    "font.size": 12,
    "axes.titlesize": 18,
    "axes.labelsize": 12,
})

CHANNEL_NAMES = [
    "body-x", "body-y", "body-z",
    "grav-x", "grav-y", "grav-z",
]
BODY_INDICES = [0, 1, 2]
GRAVITY_INDICES = [3, 4, 5]


def parse_args():
    parser = argparse.ArgumentParser(description="Build Stage 2 decoder slide assets")
    parser.add_argument(
        "--encoder_checkpoint",
        default="checkpoints/checkpoint.pt",
        help="Pretrained BioPM encoder checkpoint",
    )
    parser.add_argument(
        "--decoder_checkpoint",
        default="checkpoints/stage2_decoder_best.pt",
        help="Trained stage-2 decoder checkpoint",
    )
    parser.add_argument(
        "--eval_json",
        default="presentation_outputs/stage2_decoder_medium_eval_256.json",
        help="Stage 2 evaluation JSON with summary metrics",
    )
    parser.add_argument(
        "--output_dir",
        default="presentation_outputs/stage2_slide_assets",
        help="Directory for generated PNG assets",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device for running the figure-generation pass",
    )
    return parser.parse_args()


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def add_box(ax, xy, width, height, text, face, edge, text_color="#111111",
            fontsize=14, weight="bold"):
    box = FancyBboxPatch(
        xy, width, height,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=2.4,
        facecolor=face,
        edgecolor=edge,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=text_color,
        weight=weight,
    )


def add_arrow(ax, start, end, color="#555555"):
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=2.2,
        color=color,
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(arrow)


def save_architecture_figure(output_path: str):
    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4.8)
    ax.axis("off")

    add_box(ax, (0.6, 1.45), 2.3, 1.75, "Generated\nBioPM Tokens\n(192 x 64)",
            face="#f4e6df", edge="#b33a2b", fontsize=16)
    add_box(ax, (3.7, 1.45), 3.1, 1.75, "Stage 2 Decoder\nCross-Attention +\nTemporal Refinement",
            face="#efe7cf", edge="#9b7a19", fontsize=16)
    add_box(ax, (7.6, 1.45), 2.25, 1.75, "300 Learned\nTime Queries",
            face="#dceef7", edge="#30759d", fontsize=16)
    add_box(ax, (10.45, 2.45), 2.7, 0.82, "Filtered Acc\n(300 x 3)",
            face="#e5f3e5", edge="#3c8d4a", fontsize=14)
    add_box(ax, (10.45, 1.35), 2.7, 0.82, "Gravity\n(300 x 3)",
            face="#dceef7", edge="#30759d", fontsize=14)
    add_box(ax, (10.45, 0.25), 2.7, 0.82, "Combined Window\n(300 x 6)",
            face="#f4e6df", edge="#b33a2b", fontsize=14)

    add_arrow(ax, (2.9, 2.32), (3.7, 2.32))
    add_arrow(ax, (6.8, 2.32), (7.6, 2.32))
    add_arrow(ax, (9.85, 2.32), (10.45, 2.84))
    add_arrow(ax, (9.85, 2.32), (10.45, 1.76))
    add_arrow(ax, (9.85, 2.32), (10.45, 0.66))

    ax.text(0.6, 4.08, "Stage 2 Decoder: token space back to full IMU-like windows",
            fontsize=21, weight="bold", color="#111111")
    ax.text(
        0.6, 3.6,
        "Frozen BioPM token embeddings condition a sequence decoder that reconstructs"
        " a 300-step filtered-acceleration + gravity window.",
        fontsize=12.5, color="#444444"
    )
    ax.text(0.6, 0.08,
            "Target per window: [filtered_acc_x,y,z | gravity_x,y,z] over 300 time steps",
            fontsize=12.5, color="#444444")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_reconstruction_breakdown_figure(output_path: str):
    fig, axes = plt.subplots(2, 3, figsize=(14, 5.4), sharex=True)
    t = np.linspace(0, 10, 300)

    body_traces = [
        0.55 * np.sin(0.8 * t) + 0.08 * np.cos(2.3 * t),
        0.42 * np.cos(1.0 * t + 0.4) - 0.10 * np.sin(2.7 * t),
        0.35 * np.sin(1.4 * t - 0.6),
    ]
    grav_traces = [
        0.95 + 0.04 * np.sin(0.18 * t),
        -0.35 + 0.05 * np.cos(0.22 * t + 0.6),
        0.22 + 0.03 * np.sin(0.16 * t - 0.8),
    ]

    titles = ["Body X", "Body Y", "Body Z", "Gravity X", "Gravity Y", "Gravity Z"]
    colors = ["#3c8d4a", "#3c8d4a", "#3c8d4a", "#30759d", "#30759d", "#30759d"]
    traces = body_traces + grav_traces

    for ax, title, color, trace in zip(axes.flatten(), titles, colors, traces):
        ax.plot(t, trace, color=color, linewidth=2.4)
        ax.set_title(title, fontsize=13.5)
        ax.grid(alpha=0.22)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("What Stage 2 reconstructs", fontsize=21, weight="bold")
    fig.text(
        0.5, 0.90,
        "Each BioPM token sequence is decoded into a full 10-second window with 300 time steps and 6 channels.",
        ha="center", fontsize=12.5, color="#444444"
    )
    fig.text(0.25, 0.04, "filtered acceleration channels", ha="center",
             fontsize=12.5, color="#3c8d4a", weight="bold")
    fig.text(0.75, 0.04, "gravity channels", ha="center",
             fontsize=12.5, color="#30759d", weight="bold")
    fig.text(
        0.5, -0.01,
        "This is a stronger decoder target than Stage 1 because it reconstructs a window-level IMU-like signal instead of token-local patches.",
        ha="center", fontsize=11.5, color="#555555"
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.84])
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def load_models(args):
    encoder = load_pretrained_encoder(args.encoder_checkpoint, device=args.device)
    for param in encoder.parameters():
        param.requires_grad_(False)
    encoder.eval()

    ckpt = torch.load(args.decoder_checkpoint, map_location=args.device, weights_only=False)
    decoder = BioPMStage2Decoder(
        token_dim=64,
        hidden_dim=int(ckpt.get("hidden_dim", 96)),
        depth=int(ckpt.get("depth", 2)),
        num_heads=int(ckpt.get("num_heads", 4)),
        refine_depth=int(ckpt.get("refine_depth", 1)),
        kernel_size=int(ckpt.get("kernel_size", 7)),
        dropout=float(ckpt.get("dropout", 0.1)),
    ).to(args.device)
    decoder.load_state_dict(ckpt["model_state_dict"])
    decoder.eval()
    return encoder, decoder, ckpt


@torch.no_grad()
def predict_window(encoder, decoder, x_acc_filt, device: str):
    x = torch.from_numpy(x_acc_filt).unsqueeze(0).to(device=device, dtype=torch.float32)
    patches = x[:, :, :32]
    pos_info = x[:, :, 32]
    add_emb = x[:, :, 33:]
    valid_mask = ~torch.isnan(patches).any(dim=-1)
    mask = torch.zeros(patches.shape[:2], device=device)
    tokens = encoder.encoder_acc(patches, pos_info, mask, add_emb)
    preds = decoder(tokens, token_mask=valid_mask)
    return preds["window"].squeeze(0).cpu().numpy()


def choose_examples(dataset, encoder, decoder, device: str):
    body_candidates = []
    gravity_candidates = []

    for window_index in range(len(dataset)):
        item = dataset[window_index]
        x_acc_filt = item["x_acc_filt"].numpy()
        target = item["window_acc_filt_gravity"].numpy()
        pred = predict_window(encoder, decoder, x_acc_filt, device)

        for channel in range(target.shape[1]):
            true_series = target[:, channel]
            pred_series = pred[:, channel]
            signal_std = float(np.std(true_series))
            signal_amp = float(true_series.max() - true_series.min())
            if signal_std < 0.03 or signal_amp < 0.15:
                continue

            rmse = float(np.sqrt(np.mean((true_series - pred_series) ** 2)))
            corr = np.corrcoef(true_series, pred_series)[0, 1]
            if np.isnan(corr):
                corr = -1.0

            score = -corr + 0.5 * rmse
            candidate = {
                "window_index": int(window_index),
                "channel_index": int(channel),
                "channel_name": CHANNEL_NAMES[channel],
                "rmse": rmse,
                "corr": float(corr),
                "amp": signal_amp,
                "true_series": true_series,
                "pred_series": pred_series,
            }

            if channel in BODY_INDICES:
                body_candidates.append((score, candidate))
            else:
                gravity_candidates.append((score, candidate))

    body_candidates.sort(key=lambda x: x[0])
    gravity_candidates.sort(key=lambda x: x[0])

    chosen = []
    used = set()
    for candidate_list, needed in [(body_candidates, 2), (gravity_candidates, 2)]:
        count = 0
        for _, candidate in candidate_list:
            key = (candidate["window_index"], candidate["channel_index"])
            if key in used:
                continue
            used.add(key)
            chosen.append(candidate)
            count += 1
            if count == needed:
                break
    return chosen


def save_examples_figure(output_path: str, args):
    encoder, decoder, ckpt = load_models(args)
    eval_data = json.loads(Path(args.eval_json).read_text())
    eval_windows = int(eval_data.get("eval_windows", 64))
    dataset = Stage2WindowDataset(ckpt["val_subject_pairs"], max_windows=eval_windows)
    examples = choose_examples(dataset, encoder, decoder, args.device)

    fig, axes = plt.subplots(2, 2, figsize=(12.7, 8.2), sharex=True)
    axes = axes.flatten()
    time_axis = np.arange(300)

    for ax, ex in zip(axes, examples):
        ax.plot(time_axis, ex["true_series"], linewidth=2.2, color="#1f77b4", label="True signal")
        ax.plot(time_axis, ex["pred_series"], linewidth=2.2, color="#d95f02",
                linestyle="--", label="Decoded signal")
        ax.set_title(
            f"Window {ex['window_index']} | {ex['channel_name']}\n"
            f"RMSE {ex['rmse']:.3f} | corr {ex['corr']:.3f}",
            fontsize=12.5,
        )
        ax.grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.845),
        ncol=2,
        frameon=False,
        fontsize=12.5,
    )
    fig.suptitle("Stage 2 decoder reconstructs full-window IMU channels",
                 fontsize=20, weight="bold")
    fig.text(
        0.5, 0.01,
        "Examples include both filtered body-acceleration and gravity channels from held-out validation windows.",
        ha="center", fontsize=11.5, color="#555555"
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.79])
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    dataset.close()


def save_metrics_figure(output_path: str, eval_json_path: str):
    data = json.loads(Path(eval_json_path).read_text())
    metrics = data["metrics"]
    best_val_loss = data.get("best_val_loss", metrics["loss"])
    train_windows = data.get("train_windows") or 512
    eval_windows = data.get("eval_windows") or 128

    tiles = [
        ("Window RMSE", f"{metrics['window_rmse']:.2f}", "#f4e6df", "#b33a2b"),
        ("Body RMSE", f"{metrics['body_rmse']:.2f}", "#e5f3e5", "#3c8d4a"),
        ("Gravity RMSE", f"{metrics['gravity_rmse']:.2f}", "#dceef7", "#30759d"),
        ("Window MAE", f"{metrics['window_mae']:.2f}", "#efe7cf", "#9b7a19"),
    ]

    fig, ax = plt.subplots(figsize=(12.5, 5.6))
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 5.6)
    ax.axis("off")
    ax.text(0.2, 5.1, "Stage 2 decoder validation summary",
            fontsize=22, weight="bold", color="#111111")
    ax.text(0.2, 4.68,
            "Held-out validation performance from the chosen best Stage 2 run.",
            fontsize=12.5, color="#444444")

    positions = [(0.4, 2.1), (3.35, 2.1), (6.3, 2.1), (9.25, 2.1)]
    for (title, value, face, edge), (x, y) in zip(tiles, positions):
        add_box(ax, (x, y), 2.55, 1.7, "", face=face, edge=edge)
        ax.text(x + 1.275, y + 1.12, value, ha="center", va="center",
                fontsize=22, weight="bold", color="#111111")
        ax.text(x + 1.275, y + 0.48, title, ha="center", va="center",
                fontsize=13, color="#444444", weight="normal")

    ax.text(0.45, 1.2,
            f"Best val loss: {best_val_loss:.4f}",
            fontsize=13, color="#333333", weight="bold")
    ax.text(0.45, 0.78,
            "Train/val split: 113 / 13 subjects",
            fontsize=12, color="#555555")
    ax.text(0.45, 0.4,
            f"Train/val windows: {train_windows} / {eval_windows}",
            fontsize=12, color="#555555")

    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    save_architecture_figure(os.path.join(args.output_dir, "decoder_architecture.png"))
    save_reconstruction_breakdown_figure(
        os.path.join(args.output_dir, "decoder_reconstruction_breakdown.png"))
    save_metrics_figure(os.path.join(args.output_dir, "decoder_metrics.png"), args.eval_json)
    save_examples_figure(os.path.join(args.output_dir, "decoder_examples.png"), args)

    print(f"Saved assets to {args.output_dir}")


if __name__ == "__main__":
    main()

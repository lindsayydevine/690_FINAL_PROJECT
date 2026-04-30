#!/usr/bin/env python3
"""
Create presentation-ready assets for the Stage 1 decoder section.

Outputs a small set of PNG figures:
  - decoder_architecture.png
  - decoder_target_breakdown.png
  - decoder_examples.png
  - decoder_metrics.png
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
from src.data.h5_window_dataset import Stage1WindowDataset
from src.models.biopm import load_pretrained_encoder
from src.models.stage1_decoder import BioPMStage1Decoder


plt.rcParams.update({
    "figure.dpi": 160,
    "font.size": 12,
    "axes.titlesize": 18,
    "axes.labelsize": 12,
})


def parse_args():
    parser = argparse.ArgumentParser(description="Build decoder slide assets")
    parser.add_argument(
        "--encoder_checkpoint",
        default="checkpoints/checkpoint.pt",
        help="Pretrained BioPM encoder checkpoint",
    )
    parser.add_argument(
        "--decoder_checkpoint",
        default="presentation_outputs/stage1_decoder_presentation.pt",
        help="Trained stage-1 decoder checkpoint",
    )
    parser.add_argument(
        "--eval_json",
        default="presentation_outputs/stage1_decoder_eval.json",
        help="Decoder evaluation JSON with summary metrics",
    )
    parser.add_argument(
        "--output_dir",
        default="presentation_outputs/slide_assets",
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
    fig, ax = plt.subplots(figsize=(14, 4.6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4.6)
    ax.axis("off")

    add_box(ax, (0.6, 1.35), 2.3, 1.7, "Generated\nBioPM Tokens\n(192 x 64)",
            face="#f4e6df", edge="#b33a2b", fontsize=16)
    add_box(ax, (3.7, 1.35), 2.9, 1.7, "Stage 1 Decoder\nResidual MLP\n3 blocks, 256-d",
            face="#efe7cf", edge="#9b7a19", fontsize=16)
    add_box(ax, (7.5, 1.35), 2.2, 1.7, "Decoder\nOutputs",
            face="#dceef7", edge="#30759d", fontsize=16)
    add_box(ax, (10.3, 2.65), 2.8, 0.72, "Patch head: 32 values",
            face="#dceef7", edge="#30759d", fontsize=13)
    add_box(ax, (10.3, 1.78), 2.8, 0.72, "Axis + Direction",
            face="#e5f3e5", edge="#3c8d4a", fontsize=13)
    add_box(ax, (10.3, 0.91), 2.8, 0.72, "Position + Length",
            face="#e5f3e5", edge="#3c8d4a", fontsize=13)
    add_box(ax, (10.3, 0.04), 2.8, 0.72, "Min + Max",
            face="#e5f3e5", edge="#3c8d4a", fontsize=13)

    add_arrow(ax, (2.9, 2.2), (3.7, 2.2))
    add_arrow(ax, (6.6, 2.2), (7.5, 2.2))
    add_arrow(ax, (9.7, 2.2), (10.3, 3.0))
    add_arrow(ax, (9.7, 2.2), (10.3, 2.14))
    add_arrow(ax, (9.7, 2.2), (10.3, 1.27))
    add_arrow(ax, (9.7, 2.2), (10.3, 0.4))

    ax.text(0.6, 3.95, "Stage 1 Decoder: token space back to interpretable movement elements",
            fontsize=21, weight="bold", color="#111111")
    ax.text(0.6, 3.5,
            "Frozen BioPM encoder embeddings feed a lightweight decoder that reconstructs each token's"
            " waveform patch and metadata.",
            fontsize=12.5, color="#444444")
    ax.text(0.6, 0.1,
            "Target per token: [patch(32) | pos | axis | len | min | max | dir]",
            fontsize=12.5, color="#444444")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_target_breakdown_figure(output_path: str):
    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.set_xlim(0, 38)
    ax.set_ylim(0, 1)
    ax.axis("off")

    colors = {
        "patch": "#dceef7",
        "meta": "#e5f3e5",
        "special": "#f4e6df",
    }

    for i in range(32):
        rect = FancyBboxPatch(
            (i + 0.08, 0.28), 0.84, 0.32,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            linewidth=1.2, facecolor=colors["patch"], edgecolor="#30759d"
        )
        ax.add_patch(rect)

    meta_labels = [
        ("pos", 32, colors["meta"]),
        ("axis", 33, colors["special"]),
        ("len", 34, colors["meta"]),
        ("min", 35, colors["meta"]),
        ("max", 36, colors["meta"]),
        ("dir", 37, colors["special"]),
    ]
    for label, x, face in meta_labels:
        rect = FancyBboxPatch(
            (x + 0.08, 0.28), 0.84, 0.32,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            linewidth=1.2, facecolor=face, edgecolor="#666666"
        )
        ax.add_patch(rect)
        ax.text(x + 0.5, 0.44, label, ha="center", va="center",
                fontsize=11, weight="bold")

    xs = np.linspace(0.35, 31.65, 32)
    ys = 0.44 + 0.10 * np.sin(np.linspace(0, 2.2 * np.pi, 32))
    ax.plot(xs, ys, color="#b33a2b", linewidth=2.8)

    ax.text(0.0, 0.9, "What the decoder reconstructs",
            fontsize=21, weight="bold", color="#111111")
    ax.text(0.0, 0.78,
            "Each BioPM token is decoded into the aligned movement-element vector x_acc_filt.",
            fontsize=12.5, color="#444444")
    ax.text(0.2, 0.12, "32-point normalized waveform patch", fontsize=12.5, color="#30759d",
            weight="bold")
    ax.text(24.6, 0.12, "metadata fields", fontsize=12.5, color="#555555", weight="bold")
    ax.text(0.0, -0.02,
            "This is a cleaner first decoding target than raw IMU because it stays aligned with BioPM's tokenization.",
            fontsize=11.5, color="#555555")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def load_models(args):
    encoder = load_pretrained_encoder(args.encoder_checkpoint, device=args.device)
    for param in encoder.parameters():
        param.requires_grad_(False)
    encoder.eval()

    ckpt = torch.load(args.decoder_checkpoint, map_location=args.device, weights_only=False)
    decoder = BioPMStage1Decoder(
        token_dim=64,
        hidden_dim=int(ckpt.get("hidden_dim", 256)),
        depth=int(ckpt.get("depth", 3)),
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
    mask = torch.zeros(patches.shape[:2], device=device)
    tokens = encoder.encoder_acc(patches, pos_info, mask, add_emb)
    preds = decoder(tokens)
    return {k: v.squeeze(0).cpu().numpy() for k, v in preds.items()}


def save_examples_figure(output_path: str, args):
    encoder, decoder, ckpt = load_models(args)
    val_files = ckpt["val_subject_files"]
    eval_data = json.loads(Path(args.eval_json).read_text())
    best_examples = eval_data.get("best_examples", [])[:4]

    dataset = Stage1WindowDataset(val_files, max_windows=128)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, ex in zip(axes, best_examples):
        item = dataset[int(ex["window_index"])]
        x_acc_filt = item["x_acc_filt"].numpy()
        preds = predict_window(encoder, decoder, x_acc_filt, args.device)
        token_idx = int(ex["token_index"])

        true_patch = x_acc_filt[token_idx, :32]
        pred_patch = preds["patch"][token_idx]

        ax.plot(true_patch, linewidth=2.4, color="#1f77b4", label="True patch")
        ax.plot(pred_patch, linewidth=2.4, color="#d95f02", linestyle="--", label="Decoded patch")
        ax.set_title(
            f"Window {ex['window_index']} | Token {token_idx}\n"
            f"axis {ex['true_axis']}->{ex['pred_axis']}, len {ex['true_len']:.0f}->{ex['pred_len']:.1f}",
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
    fig.suptitle("Stage 1 decoder reconstructs token-level movement patches", fontsize=20, weight="bold")
    fig.text(0.5, 0.01,
             "Examples come from held-out validation windows using the trained presentation checkpoint.",
             ha="center", fontsize=11.5, color="#555555")
    fig.tight_layout(rect=[0, 0.04, 1, 0.79])
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    dataset.close()


def save_metrics_figure(output_path: str, eval_json_path: str):
    data = json.loads(Path(eval_json_path).read_text())
    metrics = data["metrics"]

    tiles = [
        ("Axis accuracy", f"{metrics['axis_acc'] * 100.0:.2f}%", "#e5f3e5", "#3c8d4a"),
        ("Direction accuracy", f"{metrics['dir_acc'] * 100.0:.2f}%", "#e5f3e5", "#3c8d4a"),
        ("Patch RMSE", f"{metrics['patch_rmse']:.2f}", "#dceef7", "#30759d"),
        ("Length MAE", f"{metrics['len_mae_samples']:.2f}", "#f4e6df", "#b33a2b"),
    ]

    fig, ax = plt.subplots(figsize=(12.5, 5.6))
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 5.6)
    ax.axis("off")
    ax.text(0.2, 5.1, "Decoder validation summary", fontsize=22, weight="bold", color="#111111")
    ax.text(0.2, 4.68,
            "Held-out validation performance from the Stage 1 presentation run.",
            fontsize=12.5, color="#444444")

    positions = [(0.4, 2.1), (3.35, 2.1), (6.3, 2.1), (9.25, 2.1)]
    for (title, value, face, edge), (x, y) in zip(tiles, positions):
        add_box(ax, (x, y), 2.55, 1.7, "", face=face, edge=edge)
        ax.text(x + 1.275, y + 1.12, value, ha="center", va="center",
                fontsize=22, weight="bold", color="#111111")
        ax.text(x + 1.275, y + 0.48, title, ha="center", va="center",
                fontsize=13, color="#444444", weight="normal")

    ax.text(
        0.45, 1.2,
        f"Best val loss: {data['best_val_loss']:.4f}",
        fontsize=13, color="#333333", weight="bold")
    ax.text(
        0.45, 0.78,
        "Train/val split: 113 / 13 subjects",
        fontsize=12, color="#555555")
    ax.text(
        0.45, 0.4,
        "Train/val windows: 512 / 128",
        fontsize=12, color="#555555")

    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    save_architecture_figure(os.path.join(args.output_dir, "decoder_architecture.png"))
    save_target_breakdown_figure(os.path.join(args.output_dir, "decoder_target_breakdown.png"))
    save_examples_figure(os.path.join(args.output_dir, "decoder_examples.png"), args)
    save_metrics_figure(os.path.join(args.output_dir, "decoder_metrics.png"), args.eval_json)

    print(f"Saved assets to {args.output_dir}")


if __name__ == "__main__":
    main()

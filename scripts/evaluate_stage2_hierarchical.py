#!/usr/bin/env python3
"""
Evaluate a trained hierarchical Stage 2 decoder on its held-out validation split.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.h5_window_dataset import Stage2WindowDataset
from src.models.biopm import load_pretrained_encoder
from src.models.stage1_decoder import BioPMStage1Decoder, render_body_prior_from_stage1_preds
from src.models.stage2_decoder import compute_stage2_losses
from src.models.stage2_hierarchical_decoder import BioPMHierarchicalStage2Decoder


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate hierarchical Stage-2 decoder")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--encoder_checkpoint", type=str, default="checkpoints/checkpoint.pt")
    p.add_argument("--output_json", type=str, default=None)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--max_val_windows", type=int, default=None)
    return p.parse_args()


def summarize_losses(accum, steps):
    return {k: (v / max(steps, 1)) for k, v in accum.items()}


def load_stage1_decoder(stage1_checkpoint: str, device: str):
    ckpt = torch.load(stage1_checkpoint, map_location=device, weights_only=False)
    decoder = BioPMStage1Decoder(
        token_dim=64,
        hidden_dim=int(ckpt["hidden_dim"]),
        depth=int(ckpt["depth"]),
        dropout=float(ckpt["dropout"]),
    ).to(device)
    decoder.load_state_dict(ckpt["model_state_dict"])
    for param in decoder.parameters():
        param.requires_grad_(False)
    decoder.eval()
    return decoder


def main():
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    val_pairs = ckpt["val_subject_pairs"]

    dataset = Stage2WindowDataset(val_pairs, max_windows=args.max_val_windows)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
    )

    encoder = load_pretrained_encoder(args.encoder_checkpoint, device=args.device)
    for param in encoder.parameters():
        param.requires_grad_(False)
    encoder.eval()

    stage1_checkpoint = ckpt["stage1_checkpoint"]
    stage1_decoder = load_stage1_decoder(stage1_checkpoint, args.device)

    model = BioPMHierarchicalStage2Decoder(
        token_dim=64,
        hidden_dim=int(ckpt["hidden_dim"]),
        depth=int(ckpt["depth"]),
        num_heads=int(ckpt["num_heads"]),
        refine_depth=int(ckpt["refine_depth"]),
        kernel_size=int(ckpt["kernel_size"]),
        dropout=float(ckpt["dropout"]),
        residual_scale=float(ckpt.get("residual_scale", 0.25)),
    ).to(args.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    accum = defaultdict(float)
    steps = 0

    with torch.no_grad():
        for batch in loader:
            x_acc_filt = batch["x_acc_filt"].to(args.device, dtype=torch.float32)
            target_window = batch["window_acc_filt_gravity"].to(args.device, dtype=torch.float32)

            patches = x_acc_filt[:, :, :32]
            pos_info = x_acc_filt[:, :, 32]
            add_emb = x_acc_filt[:, :, 33:]
            valid_mask = ~torch.isnan(patches).any(dim=-1)
            mask = torch.zeros(patches.shape[:2], device=args.device)

            tokens = encoder.encoder_acc(patches, pos_info, mask, add_emb)
            stage1_preds = stage1_decoder(tokens)
            body_prior = render_body_prior_from_stage1_preds(
                stage1_preds, valid_mask=valid_mask)
            preds = model(tokens, body_prior, token_mask=valid_mask)

            losses = compute_stage2_losses(
                preds,
                target_window,
                body_weight=float(ckpt.get("body_weight", 0.5)),
                gravity_weight=float(ckpt.get("gravity_weight", 1.0)),
                body_delta_weight=float(ckpt.get("body_delta_weight", 0.0)),
                gravity_delta_weight=float(ckpt.get("gravity_delta_weight", 0.0)),
                gravity_coarse_weight=float(ckpt.get("gravity_coarse_weight", 0.25)),
                gravity_coarse_kernel=int(ckpt.get("gravity_coarse_kernel", 15)),
            )

            for key, value in losses.items():
                accum[key] += float(value.detach().cpu())
            steps += 1

    metrics = summarize_losses(accum, steps)
    payload = {
        "checkpoint": args.checkpoint,
        "encoder_checkpoint": args.encoder_checkpoint,
        "stage1_checkpoint": stage1_checkpoint,
        "best_val_loss": float(ckpt.get("best_val_loss", metrics["loss"])),
        "train_subjects": len(ckpt.get("train_subject_pairs", [])),
        "val_subjects": len(val_pairs),
        "train_windows": int(ckpt.get("train_windows", 0)) if "train_windows" in ckpt else None,
        "eval_windows": len(dataset),
        "eval_batches": steps,
        "metrics": metrics,
    }

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))
    dataset.close()


if __name__ == "__main__":
    main()

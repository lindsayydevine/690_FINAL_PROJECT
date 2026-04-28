#!/usr/bin/env python3
"""
Train Stage 2: BioPM tokens -> window_acc_filt_gravity.

This script keeps the pretrained BioPM accelerometer encoder frozen, extracts
contextual 64-d tokens on the fly from x_acc_filt, and trains a sequence
decoder to reconstruct the full 300x6 filtered+gravity window.
"""

import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.h5_window_dataset import Stage2WindowDataset, list_stage2_file_pairs
from src.models.biopm import load_pretrained_encoder
from src.models.stage2_decoder import BioPMStage2Decoder, compute_stage2_losses


def parse_args():
    p = argparse.ArgumentParser(description="Train Stage-2 BioPM window decoder")
    p.add_argument("--data_dir", type=str, default="preprocessed_data",
                   help="Directory with paired Data_MeLabel_*.h5 / Data_AccLabel_*.h5 files")
    p.add_argument("--encoder_checkpoint", type=str, default="checkpoints/checkpoint.pt",
                   help="Path to pretrained BioPM encoder checkpoint")
    p.add_argument("--output", type=str, default="checkpoints/stage2_decoder.pt",
                   help="Where to save the best decoder checkpoint")
    p.add_argument("--init_checkpoint", type=str, default=None,
                   help="Optional stage2 decoder checkpoint to warm-start from")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"],
                   help="Training device")
    p.add_argument("--batch_size", type=int, default=16,
                   help="Batch size")
    p.add_argument("--epochs", type=int, default=10,
                   help="Number of training epochs")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Learning rate")
    p.add_argument("--weight_decay", type=float, default=1e-4,
                   help="AdamW weight decay")
    p.add_argument("--hidden_dim", type=int, default=128,
                   help="Decoder hidden size")
    p.add_argument("--depth", type=int, default=4,
                   help="Number of query decoder layers")
    p.add_argument("--num_heads", type=int, default=4,
                   help="Attention heads per decoder layer")
    p.add_argument("--refine_depth", type=int, default=2,
                   help="Number of post-attention temporal refinement blocks")
    p.add_argument("--kernel_size", type=int, default=9,
                   help="Kernel size for temporal refinement blocks")
    p.add_argument("--dropout", type=float, default=0.1,
                   help="Decoder dropout")
    p.add_argument("--separate_gravity_branch", action="store_true",
                   help="Use a dedicated coarse-to-fine gravity branch")
    p.add_argument("--gravity_query_steps", type=int, default=None,
                   help="Optional number of coarse gravity query steps")
    p.add_argument("--body_weight", type=float, default=1.0,
                   help="Loss weight for filtered acceleration channels")
    p.add_argument("--gravity_weight", type=float, default=1.0,
                   help="Loss weight for gravity channels")
    p.add_argument("--body_delta_weight", type=float, default=0.0,
                   help="Loss weight for temporal-difference loss on body channels")
    p.add_argument("--gravity_delta_weight", type=float, default=0.0,
                   help="Loss weight for temporal-difference loss on gravity channels")
    p.add_argument("--gravity_coarse_weight", type=float, default=0.0,
                   help="Loss weight for coarse pooled gravity reconstruction")
    p.add_argument("--gravity_coarse_kernel", type=int, default=15,
                   help="Average-pooling kernel for coarse gravity loss")
    p.add_argument("--grad_clip", type=float, default=1.0,
                   help="Gradient clipping norm; use <=0 to disable")
    p.add_argument("--val_subject_frac", type=float, default=0.1,
                   help="Fraction of subjects to reserve for validation")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed")
    p.add_argument("--num_workers", type=int, default=0,
                   help="DataLoader workers; keep at 0 unless HDF5 access is stable")
    p.add_argument("--max_train_windows", type=int, default=None,
                   help="Optional cap on train windows for quick experiments")
    p.add_argument("--max_val_windows", type=int, default=None,
                   help="Optional cap on validation windows for quick experiments")
    return p.parse_args()


def split_subject_items(items, val_subject_frac, seed):
    if not 0.0 < val_subject_frac < 1.0:
        raise ValueError("--val_subject_frac must be between 0 and 1")

    rng = np.random.default_rng(seed)
    indices = np.arange(len(items))
    rng.shuffle(indices)
    n_val = max(1, int(round(len(items) * val_subject_frac)))
    val_idx = set(indices[:n_val].tolist())

    train_items = [item for i, item in enumerate(items) if i not in val_idx]
    val_items = [item for i, item in enumerate(items) if i in val_idx]
    return train_items, val_items


def summarize_losses(accum, steps):
    return {k: (v / max(steps, 1)) for k, v in accum.items()}


def run_epoch(decoder, encoder, loader, optimizer, device, body_weight,
              gravity_weight, body_delta_weight, gravity_delta_weight,
              gravity_coarse_weight, gravity_coarse_kernel,
              grad_clip, train=True):
    decoder.train(mode=train)
    encoder.eval()
    accum = defaultdict(float)
    steps = 0

    for batch in loader:
        x_acc_filt = batch["x_acc_filt"].to(device, dtype=torch.float32)
        target_window = batch["window_acc_filt_gravity"].to(device, dtype=torch.float32)

        patches = x_acc_filt[:, :, :32]
        pos_info = x_acc_filt[:, :, 32]
        add_emb = x_acc_filt[:, :, 33:]
        valid_mask = ~torch.isnan(patches).any(dim=-1)
        mask = torch.zeros(patches.shape[:2], device=device)

        with torch.no_grad():
            tokens = encoder.encoder_acc(patches, pos_info, mask, add_emb)

        preds = decoder(tokens, token_mask=valid_mask)
        losses = compute_stage2_losses(
            preds,
            target_window,
            body_weight=body_weight,
            gravity_weight=gravity_weight,
            body_delta_weight=body_delta_weight,
            gravity_delta_weight=gravity_delta_weight,
            gravity_coarse_weight=gravity_coarse_weight,
            gravity_coarse_kernel=gravity_coarse_kernel,
        )

        if train:
            optimizer.zero_grad()
            losses["loss"].backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)
            optimizer.step()

        for key, value in losses.items():
            accum[key] += float(value.detach().cpu())
        steps += 1

    return summarize_losses(accum, steps)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("Stage 2 Decoder Training")
    print("=" * 60)
    print(f"  Data dir:           {args.data_dir}")
    print(f"  Encoder checkpoint: {args.encoder_checkpoint}")
    print(f"  Output:             {args.output}")
    print(f"  Init checkpoint:    {args.init_checkpoint}")
    print(f"  Device:             {args.device}")
    print(f"  Batch size:         {args.batch_size}")
    print(f"  Epochs:             {args.epochs}")

    file_pairs = list_stage2_file_pairs(args.data_dir)
    train_pairs, val_pairs = split_subject_items(
        file_pairs, val_subject_frac=args.val_subject_frac, seed=args.seed)

    train_ds = Stage2WindowDataset(train_pairs, max_windows=args.max_train_windows)
    val_ds = Stage2WindowDataset(val_pairs, max_windows=args.max_val_windows)

    print(f"  Train subjects:     {len(train_pairs)}")
    print(f"  Val subjects:       {len(val_pairs)}")
    print(f"  Train windows:      {len(train_ds)}")
    print(f"  Val windows:        {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
    )

    encoder = load_pretrained_encoder(args.encoder_checkpoint, device=args.device)
    for param in encoder.parameters():
        param.requires_grad_(False)
    encoder.eval()

    decoder = BioPMStage2Decoder(
        token_dim=64,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        refine_depth=args.refine_depth,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        separate_gravity_branch=args.separate_gravity_branch,
        gravity_query_steps=args.gravity_query_steps,
    ).to(args.device)

    if args.init_checkpoint:
        init_ckpt = torch.load(args.init_checkpoint, map_location=args.device, weights_only=False)
        decoder.load_state_dict(init_ckpt["model_state_dict"])
        print(f"  Warm-started decoder from: {args.init_checkpoint}")

    optimizer = torch.optim.AdamW(
        decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(
            decoder, encoder, train_loader, optimizer, args.device,
            body_weight=args.body_weight, gravity_weight=args.gravity_weight,
            body_delta_weight=args.body_delta_weight,
            gravity_delta_weight=args.gravity_delta_weight,
            gravity_coarse_weight=args.gravity_coarse_weight,
            gravity_coarse_kernel=args.gravity_coarse_kernel,
            grad_clip=args.grad_clip, train=True)
        with torch.no_grad():
            val_stats = run_epoch(
                decoder, encoder, val_loader, optimizer, args.device,
                body_weight=args.body_weight, gravity_weight=args.gravity_weight,
                body_delta_weight=args.body_delta_weight,
                gravity_delta_weight=args.gravity_delta_weight,
                gravity_coarse_weight=args.gravity_coarse_weight,
                gravity_coarse_kernel=args.gravity_coarse_kernel,
                grad_clip=args.grad_clip, train=False)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_stats['loss']:.5f} "
            f"val_loss={val_stats['loss']:.5f} | "
            f"body_rmse={val_stats['body_rmse']:.5f} "
            f"gravity_rmse={val_stats['gravity_rmse']:.5f} "
            f"window_rmse={val_stats['window_rmse']:.5f}"
        )

        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            torch.save(
                {
                    "model_state_dict": decoder.state_dict(),
                    "hidden_dim": args.hidden_dim,
                    "depth": args.depth,
                    "num_heads": args.num_heads,
                    "refine_depth": args.refine_depth,
                    "kernel_size": args.kernel_size,
                    "dropout": args.dropout,
                    "separate_gravity_branch": args.separate_gravity_branch,
                    "gravity_query_steps": args.gravity_query_steps,
                    "body_weight": args.body_weight,
                    "gravity_weight": args.gravity_weight,
                    "body_delta_weight": args.body_delta_weight,
                    "gravity_delta_weight": args.gravity_delta_weight,
                    "gravity_coarse_weight": args.gravity_coarse_weight,
                    "gravity_coarse_kernel": args.gravity_coarse_kernel,
                    "best_val_loss": best_val,
                    "train_windows": len(train_ds),
                    "val_windows": len(val_ds),
                    "train_subject_pairs": train_pairs,
                    "val_subject_pairs": val_pairs,
                },
                args.output,
            )
            print(f"  Saved best decoder to: {args.output}")

    train_ds.close()
    val_ds.close()


if __name__ == "__main__":
    main()

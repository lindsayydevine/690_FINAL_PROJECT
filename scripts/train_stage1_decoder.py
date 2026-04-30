#!/usr/bin/env python3
"""
Train Stage 1 only: BioPM tokens -> x_acc_filt.

This script keeps the pretrained BioPM encoder frozen, extracts contextual
64-d tokens on the fly, and trains a lightweight decoder to reconstruct the
aligned movement-element representation from those tokens.
"""

import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.h5_window_dataset import Stage1WindowDataset, list_me_h5_files
from src.models.biopm import load_pretrained_encoder
from src.models.stage1_decoder import BioPMStage1Decoder, compute_stage1_losses


def parse_args():
    p = argparse.ArgumentParser(description="Train Stage-1 BioPM token decoder")
    p.add_argument("--data_dir", type=str, default="preprocessed_data",
                   help="Directory with Data_MeLabel_*.h5 files")
    p.add_argument("--encoder_checkpoint", type=str, default="checkpoints/checkpoint.pt",
                   help="Path to pretrained BioPM encoder checkpoint")
    p.add_argument("--output", type=str, default="checkpoints/stage1_decoder.pt",
                   help="Where to save the best decoder checkpoint")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"],
                   help="Training device")
    p.add_argument("--batch_size", type=int, default=32,
                   help="Batch size")
    p.add_argument("--epochs", type=int, default=10,
                   help="Number of training epochs")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Learning rate")
    p.add_argument("--weight_decay", type=float, default=1e-4,
                   help="AdamW weight decay")
    p.add_argument("--hidden_dim", type=int, default=256,
                   help="Decoder hidden size")
    p.add_argument("--depth", type=int, default=3,
                   help="Number of residual MLP blocks")
    p.add_argument("--dropout", type=float, default=0.1,
                   help="Decoder dropout")
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


def split_subject_files(file_paths, val_subject_frac, seed):
    if not 0.0 < val_subject_frac < 1.0:
        raise ValueError("--val_subject_frac must be between 0 and 1")

    rng = np.random.default_rng(seed)
    indices = np.arange(len(file_paths))
    rng.shuffle(indices)
    n_val = max(1, int(round(len(file_paths) * val_subject_frac)))
    val_idx = set(indices[:n_val].tolist())

    train_files = [path for i, path in enumerate(file_paths) if i not in val_idx]
    val_files = [path for i, path in enumerate(file_paths) if i in val_idx]
    return train_files, val_files


def summarize_losses(accum, steps):
    return {k: (v / max(steps, 1)) for k, v in accum.items()}


def run_epoch(decoder, encoder, loader, optimizer, device, train=True):
    decoder.train(mode=train)
    encoder.eval()
    accum = defaultdict(float)
    steps = 0

    for batch in loader:
        x_acc_filt = batch["x_acc_filt"].to(device, dtype=torch.float32)

        patches = x_acc_filt[:, :, :32]
        pos_info = x_acc_filt[:, :, 32]
        add_emb = x_acc_filt[:, :, 33:]
        valid_mask = ~torch.isnan(patches).any(dim=-1)
        mask = torch.zeros(patches.shape[:2], device=device)

        with torch.no_grad():
            tokens = encoder.encoder_acc(patches, pos_info, mask, add_emb)

        preds = decoder(tokens)
        losses = compute_stage1_losses(preds, x_acc_filt, valid_mask)

        if train:
            optimizer.zero_grad()
            losses["loss"].backward()
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
    print("Stage 1 Decoder Training")
    print("=" * 60)
    print(f"  Data dir:           {args.data_dir}")
    print(f"  Encoder checkpoint: {args.encoder_checkpoint}")
    print(f"  Output:             {args.output}")
    print(f"  Device:             {args.device}")
    print(f"  Batch size:         {args.batch_size}")
    print(f"  Epochs:             {args.epochs}")

    file_paths = list_me_h5_files(args.data_dir)
    train_files, val_files = split_subject_files(
        file_paths, val_subject_frac=args.val_subject_frac, seed=args.seed)

    train_ds = Stage1WindowDataset(train_files, max_windows=args.max_train_windows)
    val_ds = Stage1WindowDataset(val_files, max_windows=args.max_val_windows)

    print(f"  Train subjects:     {len(train_files)}")
    print(f"  Val subjects:       {len(val_files)}")
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

    decoder = BioPMStage1Decoder(
        token_dim=64,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        dropout=args.dropout,
    ).to(args.device)

    optimizer = torch.optim.AdamW(
        decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(decoder, encoder, train_loader, optimizer, args.device, train=True)
        with torch.no_grad():
            val_stats = run_epoch(decoder, encoder, val_loader, optimizer, args.device, train=False)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_stats['loss']:.5f} "
            f"val_loss={val_stats['loss']:.5f} | "
            f"patch={val_stats['patch_loss']:.5f} "
            f"axis_acc={val_stats['axis_acc']:.4f} "
            f"dir_acc={val_stats['dir_acc']:.4f}"
        )

        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            torch.save(
                {
                    "model_state_dict": decoder.state_dict(),
                    "hidden_dim": args.hidden_dim,
                    "depth": args.depth,
                    "dropout": args.dropout,
                    "best_val_loss": best_val,
                    "train_subject_files": train_files,
                    "val_subject_files": val_files,
                },
                args.output,
            )
            print(f"  Saved best decoder to: {args.output}")

    train_ds.close()
    val_ds.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Train Stage 1 with an auxiliary body-prior loss:
BioPM tokens -> x_acc_filt, plus soft rendering back to filtered body acceleration.
"""

import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.h5_window_dataset import Stage2WindowDataset, list_stage2_file_pairs
from src.models.biopm import load_pretrained_encoder
from src.models.stage1_decoder import (
    BioPMStage1Decoder,
    compute_stage1_losses,
    render_body_prior_from_stage1_preds,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train prior-aware Stage-1 decoder")
    p.add_argument("--data_dir", type=str, default="preprocessed_data")
    p.add_argument("--encoder_checkpoint", type=str, default="checkpoints/checkpoint.pt")
    p.add_argument("--init_checkpoint", type=str, default=None)
    p.add_argument("--output", type=str, default="checkpoints/stage1_prior_decoder.pt")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--body_render_weight", type=float, default=1.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--val_subject_frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--max_train_windows", type=int, default=512)
    p.add_argument("--max_val_windows", type=int, default=256)
    return p.parse_args()


def split_subject_items(items, val_subject_frac, seed):
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


def run_epoch(decoder, encoder, loader, optimizer, device, body_render_weight,
              grad_clip, train=True):
    decoder.train(mode=train)
    encoder.eval()
    accum = defaultdict(float)
    steps = 0

    for batch in loader:
        x_acc_filt = batch["x_acc_filt"].to(device, dtype=torch.float32)
        target_window = batch["window_acc_filt_gravity"].to(device, dtype=torch.float32)
        body_target = target_window[:, :, :3]

        patches = x_acc_filt[:, :, :32]
        pos_info = x_acc_filt[:, :, 32]
        add_emb = x_acc_filt[:, :, 33:]
        valid_mask = ~torch.isnan(patches).any(dim=-1)
        mask = torch.zeros(patches.shape[:2], device=device)

        with torch.no_grad():
            tokens = encoder.encoder_acc(patches, pos_info, mask, add_emb)

        preds = decoder(tokens)
        token_losses = compute_stage1_losses(preds, x_acc_filt, valid_mask)
        body_prior = render_body_prior_from_stage1_preds(preds, valid_mask=valid_mask)
        body_render_loss = F.smooth_l1_loss(body_prior, body_target)
        body_prior_rmse = torch.sqrt(torch.mean((body_prior - body_target).pow(2)) + 1e-12)

        total = token_losses["loss"] + body_render_weight * body_render_loss

        if train:
            optimizer.zero_grad()
            total.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)
            optimizer.step()

        accum["loss"] += float(total.detach().cpu())
        accum["token_loss"] += float(token_losses["loss"].detach().cpu())
        accum["body_render_loss"] += float(body_render_loss.detach().cpu())
        accum["body_prior_rmse"] += float(body_prior_rmse.detach().cpu())

        for key, value in token_losses.items():
            if key != "loss":
                accum[key] += float(value.detach().cpu())
        steps += 1

    return summarize_losses(accum, steps)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("Stage 1 Prior-Aware Decoder Training")
    print("=" * 60)
    print(f"  Data dir:           {args.data_dir}")
    print(f"  Encoder checkpoint: {args.encoder_checkpoint}")
    print(f"  Init checkpoint:    {args.init_checkpoint}")
    print(f"  Output:             {args.output}")
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
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(args.device == "cuda"))
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(args.device == "cuda"))

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
            body_render_weight=args.body_render_weight,
            grad_clip=args.grad_clip, train=True)
        with torch.no_grad():
            val_stats = run_epoch(
                decoder, encoder, val_loader, optimizer, args.device,
                body_render_weight=args.body_render_weight,
                grad_clip=args.grad_clip, train=False)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_stats['loss']:.5f} "
            f"val_loss={val_stats['loss']:.5f} | "
            f"body_prior_rmse={val_stats['body_prior_rmse']:.5f} "
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
                    "body_render_weight": args.body_render_weight,
                    "train_subject_pairs": train_pairs,
                    "val_subject_pairs": val_pairs,
                    "train_windows": len(train_ds),
                    "val_windows": len(val_ds),
                },
                args.output,
            )
            print(f"  Saved best decoder to: {args.output}")

    train_ds.close()
    val_ds.close()


if __name__ == "__main__":
    main()

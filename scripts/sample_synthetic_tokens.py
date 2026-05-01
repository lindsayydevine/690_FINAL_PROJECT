#!/usr/bin/env python3
"""
Sample synthetic BioPM token sequences from a trained GRU autoregressor.

This script keeps generation in BioPM representation space:
  1. Load preprocessed Data_MeLabel_*.h5 windows
  2. Use the pretrained BioPM encoder to extract token sequences
  3. Use a trained GRU autoregressor checkpoint to roll out future tokens
  4. Save real and synthetic token sequences for analysis or augmentation
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.preprocessing import load_preprocessed_h5
from src.models.biopm import load_pretrained_encoder
from src.models.BioPMAutoregressor import load_autoregressor_checkpoint
from scripts.generation import extract_tokens, generate_tokens


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate synthetic BioPM token sequences from a GRU checkpoint")
    p.add_argument("--data_dir", type=str, required=True,
                   help="Directory with preprocessed Data_MeLabel_*.h5 files")
    p.add_argument("--encoder_checkpoint", type=str, required=True,
                   help="Path to the pretrained BioPM encoder checkpoint")
    p.add_argument("--autoreg_checkpoint", type=str, required=True,
                   help="Path to a trained GRU autoregressor checkpoint")
    p.add_argument("--output", type=str, required=True,
                   help="Output .npz path for real and synthetic tokens")
    p.add_argument("--device", type=str, default="cpu",
                   choices=["cpu", "cuda"],
                   help="Device for token extraction and generation")
    p.add_argument("--batch_size", type=int, default=64,
                   help="Batch size for BioPM token extraction")
    p.add_argument("--start_idx", type=int, default=0,
                   help="Start window index for sampling")
    p.add_argument("--num_samples", type=int, default=256,
                   help="How many real windows to use as seeds")
    p.add_argument("--seed_len", type=int, default=16,
                   help="Number of real tokens to condition on")
    p.add_argument("--generate_steps", type=int, default=None,
                   help="How many future tokens to generate; default fills the rest of the window")
    p.add_argument("--noise_std", type=float, default=0.01,
                   help="Gaussian noise added to each generated token")
    return p.parse_args()


def main():
    args = parse_args()

    (X, pos_info, add_emb, labels, pids,
     X_grav, raw_acc) = load_preprocessed_h5(args.data_dir)

    end_idx = min(args.start_idx + args.num_samples, X.shape[0])
    if args.start_idx < 0 or args.start_idx >= end_idx:
        raise ValueError("Requested sample range is out of bounds")

    X = X[args.start_idx:end_idx]
    pos_info = pos_info[args.start_idx:end_idx]
    add_emb = add_emb[args.start_idx:end_idx]
    labels = labels[args.start_idx:end_idx]
    pids = pids[args.start_idx:end_idx]

    encoder = load_pretrained_encoder(args.encoder_checkpoint, device=args.device)
    tokens = extract_tokens(
        encoder, X, pos_info, add_emb,
        device=args.device, batch_size=args.batch_size,
    )

    seq_len = tokens.shape[1]
    if not 1 <= args.seed_len < seq_len:
        raise ValueError(f"--seed_len must be in [1, {seq_len - 1}]")

    generate_steps = args.generate_steps
    if generate_steps is None:
        generate_steps = seq_len - args.seed_len
    if generate_steps <= 0:
        raise ValueError("--generate_steps must be positive")

    autoreg_model, ckpt = load_autoregressor_checkpoint(
        args.autoreg_checkpoint, map_location=args.device)

    seed_tokens = tokens[:, :args.seed_len, :]
    synthetic_tokens = generate_tokens(
        autoreg_model,
        seed_tokens=seed_tokens,
        generate_steps=generate_steps,
        device=args.device,
        noise_std=args.noise_std,
    )

    real_tokens = tokens.cpu().numpy()
    synthetic_tokens = synthetic_tokens.cpu().numpy()
    overlap_steps = min(generate_steps, max(0, seq_len - args.seed_len))
    if overlap_steps > 0:
        real_future = real_tokens[:, args.seed_len:args.seed_len + overlap_steps, :]
        synth_future = synthetic_tokens[:, args.seed_len:args.seed_len + overlap_steps, :]
        future_mse = float(np.mean((real_future - synth_future) ** 2))
    else:
        future_mse = float("nan")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.savez(
        args.output,
        real_tokens=real_tokens,
        synthetic_tokens=synthetic_tokens,
        labels=labels,
        pids=pids,
        source_window_indices=np.arange(args.start_idx, end_idx),
        seed_len=np.int32(args.seed_len),
        generate_steps=np.int32(generate_steps),
        noise_std=np.float32(args.noise_std),
        autoreg_best_val_loss=np.float32(ckpt.get("best_val_loss", np.nan)),
    )

    print("=" * 60)
    print("Synthetic Token Sampling Complete")
    print("=" * 60)
    print(f"Windows used:      {len(real_tokens)}")
    print(f"Real token shape:  {real_tokens.shape}")
    print(f"Synth token shape: {synthetic_tokens.shape}")
    print(f"Seed length:       {args.seed_len}")
    print(f"Generate steps:    {generate_steps}")
    print(f"Future-token MSE:  {future_mse:.6f}")
    print(f"Saved to:          {args.output}")


if __name__ == "__main__":
    main()

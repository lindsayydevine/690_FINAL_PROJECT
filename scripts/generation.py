#!/usr/bin/env python3
"""
generation_starter.py — EXPERIMENTAL scaffold for movement generation.

===========================================================================
  ⚠️  THIS IS EXPERIMENTAL AND NOT A VALIDATED GENERATION PIPELINE  ⚠️
===========================================================================

BioPM's encoder_acc is a BIDIRECTIONAL masked transformer (BERT-style).
It was trained to reconstruct masked movement elements from context, NOT
to generate sequences autoregressively.  This means:

  ✗  It is NOT a proper autoregressive generator (like GPT)
  ✗  It cannot natively produce novel movement sequences from scratch
  ✗  Generated outputs have not been validated for scientific correctness

What this script provides instead:

  ✓  A "masked infilling" experiment: mask some movement elements in a
     real window and let the model reconstruct them from context
  ✓  A scaffold for iterative generation (mask → predict → unmask → repeat)
  ✓  Clear documentation of limitations

This may be useful for:
  - Understanding what the model has learned
  - Exploring the model's internal representations
  - Course projects that want to experiment with generation ideas
  - Prototyping augmentation strategies

It is NOT suitable for:
  - Publishing generated movement data as realistic
  - Any claim that BioPM is a generative model

=== USAGE ===

  python scripts/generation_starter.py \
      --data_dir     /path/to/preprocessed \
      --checkpoint   checkpoints/checkpoint.pt \
      --mask_ratio   0.5 \
      --device       cpu
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.models.biopm import load_pretrained_encoder
from src.data.preprocessing import load_preprocessed_h5
from scripts.BioPMAutoregressor import BioPMAutoregressor

def parse_args():
    p = argparse.ArgumentParser(
        description="Experimental: masked infilling with BioPM")
    p.add_argument("--data_dir", type=str, required=True,
                   help="Directory with preprocessed Data_MeLabel_*.h5 files")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to 50MR checkpoint.pt")
    p.add_argument("--save_path", type=str, required=True, help='Path to checkpoint')
    p.add_argument("--batch_size", type=int, default=64,
                   help="Batch size for token extraction and AR training")
    p.add_argument("--epochs", type=int, default=20,
                   help="Number of training epochs")
    p.add_argument("--hidden_dim", type=int, default=128,
                   help="GRU hidden size")
    p.add_argument("--num_layers", type=int, default=2,
                   help="Number of GRU layers")
    p.add_argument("--dropout", type=float, default=0.1,
                   help="GRU dropout")
    p.add_argument("--val_split", type=float, default=0.1,
                   help="Validation split fraction")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Learning rate")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed")
    p.add_argument("--max_windows", type=int, default=None,
                   help="Optional cap on preprocessed windows for faster experiments")
    p.add_argument("--mask_ratio", type=float, default=0.5,
                   help="Fraction of ME patches to mask (default: 0.5)")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--sample_idx", type=int, default=0,
                   help="Index of sample to experiment with")
    return p.parse_args()


class TokenDataset(Dataset):
    def __init__(self, tokens):
        self.tokens = tokens

    def __len__(self):
        return self.tokens.shape[0]

    def __getitem__(self, idx):
        return self.tokens[idx]

def loss_func(pred, target):
    loss = F.mse_loss(pred, target)
    return loss, loss.item()

@torch.no_grad()
def extract_tokens(model, X, pos_info, add_emb, device="cpu", batch_size=30):
    model.eval()
    tokens_list = []

    n = len(X)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)

        X_b = torch.from_numpy(X[start:end]).float().to(device)
        pos_b = torch.from_numpy(pos_info[start:end]).float().to(device)
        add_b = torch.from_numpy(add_emb[start:end]).float().to(device)
        mask_b = torch.zeros(X_b.shape[0], X_b.shape[1], device=device)
        z = model.encoder_acc(X_b, pos_b, mask_b, add_b)
        tokens_list.append(z.cpu())

    return torch.cat(tokens_list, dim=0)

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, total_mse = 0.0, 0.0
    n_batches = 0

    for tokens in loader:
        tokens = tokens.to(device)           
        x = tokens[:, :-1, :]                
        y = tokens[:, 1:, :]                 

        optimizer.zero_grad()
        pred, _ = model(x)
        loss, mse = loss_func(pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_mse += mse
        n_batches += 1

    return total_loss / n_batches, total_mse / n_batches

@torch.no_grad()
def eval_one_epoch(model, loader, device):
    model.eval()
    total_loss, total_mse = 0.0, 0.0
    n_batches = 0

    for tokens in loader:
        tokens = tokens.to(device)
        x = tokens[:, :-1, :]
        y = tokens[:, 1:, :]

        pred, _ = model(x)
        loss, mse = loss_func(pred, y)

        total_loss += loss.item()
        total_mse += mse
        n_batches += 1

    return total_loss / n_batches, total_mse / n_batches

@torch.no_grad()
def generate_tokens(model, seed_tokens, generate_steps, device="cpu", noise_std=0.0):
    model.eval()
    tokens = seed_tokens.to(device)
    h = None

    _, h = model(tokens, h0=h)
    current = tokens[:, -1:, :]   

    generated = [tokens]
    for _ in range(generate_steps):
        pred, h = model(current, h0=h)
        next_token = pred[:, -1:, :]

        if noise_std > 0:
            next_token = next_token + noise_std * torch.randn_like(next_token)

        generated.append(next_token)
        current = next_token

    return torch.cat(generated, dim=1)


def main():
    args = parse_args()
    if not 0.0 < args.val_split < 1.0:
        raise ValueError("--val_split must be between 0 and 1")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    (X, pos_info, add_emb, labels, pids,
     X_grav, raw_acc) = load_preprocessed_h5(args.data_dir)
    if args.max_windows is not None and X.shape[0] > args.max_windows:
        rng = np.random.default_rng(args.seed)
        subset_idx = np.sort(rng.choice(X.shape[0], size=args.max_windows, replace=False))
        X = X[subset_idx]
        pos_info = pos_info[subset_idx]
        add_emb = add_emb[subset_idx]
        labels = labels[subset_idx]
        pids = pids[subset_idx]
        X_grav = X_grav[subset_idx] if X_grav is not None else None
        raw_acc = raw_acc[subset_idx]
        print(f"Using a subset of {args.max_windows} windows for AR training")

    print("Loading pretrained BioPM encoder")
    model_biopm = load_pretrained_encoder(args.checkpoint, device=args.device)
    model_biopm.eval()
    print("Loaded pretrained BioPM encoder")

    print("Extracting token sequences...")
    tokens = extract_tokens(
        model_biopm, X, pos_info, add_emb,
        device=args.device, batch_size=args.batch_size
    )  

    print(f"Extracted tokens shape: {tuple(tokens.shape)}")

    dataset = TokenDataset(tokens)
    n_total = len(dataset)
    n_val = int(n_total * args.val_split)
    n_train = n_total - n_val
    if n_train <= 0 or n_val <= 0:
        raise ValueError("Not enough token windows for the requested validation split")
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = args.device
    token_dim = tokens.shape[-1]
    ar_model = BioPMAutoregressor(
        token_dim=token_dim,
        hidden_dim = args.hidden_dim,
        num_layers = args.num_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(ar_model.parameters(), lr=args.lr)

    best_val = float('inf')


    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_mse = train_one_epoch(ar_model, train_loader, optimizer, device)
        va_loss, va_mse = eval_one_epoch(ar_model, val_loader, device)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={tr_loss:.6f} train_mse={tr_mse:.6f} | "
            f"val_loss={va_loss:.6f} val_mse={va_mse:.6f}"
        )

        if va_loss < best_val:
            best_val = va_loss
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            torch.save(
                {
                    "model_state_dict": ar_model.state_dict(),
                    "token_dim": token_dim,
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers,
                    "dropout": args.dropout,
                    "best_val_loss": best_val,
                },
                args.save_path,
            )
            print(f"  Saved best model to: {args.save_path}")
    
if __name__ == "__main__":
    main()

'''
Use as:
checkpoint = torch.load("path/to/checkpoints/biopm_gru_autoreg.pt", map_location="cpu")
model = BioPMAutoregressor(
    token_dim=checkpoint["token_dim"],
    hidden_dim=checkpoint["hidden_dim"],
    num_layers=checkpoint["num_layers"],
    dropout=checkpoint["dropout"],
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

seed_len = min(16, tokens.shape[1] - 1)
seed = tokens[0:1, :seed_len, :]
synthetic = generate_tokens(
    ar_model,
    seed_tokens=seed,
    generate_steps=tokens.shape[1] - seed_len,
    device=args.device,
    noise_std=0.01,
)
'''

import time
import torch
from generation import *
import h5py
from src.data.preprocessing import load_preprocessed_h5_by_file
# python3 ./scripts/index.py --data_dir './pre_data/'


def log_time(task_name, start_time):
    elapsed = time.time() - start_time

    if elapsed >= 60:
        minutes = elapsed / 60
        print(f"[TIME] {task_name} took {minutes:.2f} minutes")
    else:
        print(f"[TIME] {task_name} took {elapsed:.2f} seconds")


def parse_args():
    p = argparse.ArgumentParser(
        description="Experimental: masked infilling with BioPM"
    )
    p.add_argument("--data_dir", type=str, required=True,
                   help="Directory with preprocessed Data_MeLabel_*.h5 files")
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
    p.add_argument("--device", type=str, default=None,
                   help="Device to use: cpu, cuda, or mps. If not set, auto-detect.")
    p.add_argument("--sample_idx", type=int, default=0,
                   help="Index of sample to experiment with")
    return p.parse_args()


def generate_synthetic_data(args, model_biopm, gen_model, name, data, device='cpu'):
    start = time.time()
    (
        X, pos_info, add_emb, labels, pids,
        X_grav, raw_acc
    ) = data
    log_time("Loading preprocessed H5 data", start)


    if args.max_windows is not None and X.shape[0] > args.max_windows:
        start = time.time()

        rng = np.random.default_rng(args.seed)
        subset_idx = np.sort(
            rng.choice(X.shape[0], size=args.max_windows, replace=False)
        )

        X = X[subset_idx]
        pos_info = pos_info[subset_idx]
        add_emb = add_emb[subset_idx]
        labels = labels[subset_idx]
        pids = pids[subset_idx]
        X_grav = X_grav[subset_idx] if X_grav is not None else None
        raw_acc = raw_acc[subset_idx]

        print(f"Using a subset of {args.max_windows} windows for AR training")
        log_time("Subsetting windows", start)

    start = time.time()
    print("Extracting token sequences...")
    tokens = extract_tokens(
        model_biopm,
        X,
        pos_info,
        add_emb,
        device=device,
        batch_size=args.batch_size
    )
    print(f"Extracted tokens shape: {tuple(tokens.shape)}")
    log_time("Extracting token sequences", start)

    print(tokens.shape)

    start = time.time()


    num_windows = tokens.shape[0]
    synthetic_windows = []
    for i in range(num_windows):
        start = time.time()
        seed_len = min(16, tokens.shape[1] - 1)
        seed = tokens[i:i+1, :seed_len, :].to(device)

        synthetic = generate_token(
            gen_model,
            seed_tokens=seed,
            generate_steps=tokens.shape[1],
            device=device,
            noise_std=0.01,
        )
        synthetic_windows.append(synthetic.cpu())
    synthetic_windows = torch.cat(synthetic_windows,dim=0)
    print(synthetic_windows.shape)


    with h5py.File(f'./synthetic_data/Synthetic_MeLabel_{name}.h5', "w") as f:
        f.create_dataset("synthetic_tokens", data=synthetic_windows.numpy())
        f.create_dataset("source_labels", data=labels)
        f.create_dataset("source_subject_ids", data=pids)

    log_time("Generating synthetic tokens", start)

def main():  
    total_start = time.time()

    start = time.time()
    args = parse_args()
    log_time("Argument parsing", start)


    start = time.time()
    if args.device is not None:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Hello")
    print(f"Using device: {device}")
    log_time("Device setup", start)


    start = time.time()

    data_dict = load_preprocessed_h5_by_file(args.data_dir)

    start = time.time()
    biopm_chkpt = "./checkpoints/checkpoint.pt"
    print("Loading checkpoint")
    model_biopm = load_pretrained_encoder(biopm_chkpt, device=device)
    model_biopm.eval()
    print("Checkpoint Loaded")
    log_time("Loading BioPM checkpoint", start)

    start = time.time()
    gen_chkpt = torch.load(
        "./checkpoints/biopm_gru_autoreg.pt",
        map_location="cpu"
    )

    gen_model = BioPMAutoregressor(
        token_dim=gen_chkpt["token_dim"],
        hidden_dim=gen_chkpt["hidden_dim"],
        num_layers=gen_chkpt["num_layers"],
        dropout=gen_chkpt["dropout"],
    )

    gen_model.load_state_dict(gen_chkpt["model_state_dict"])
    gen_model.to(device)
    gen_model.eval()
    log_time("Loading autoregressive checkpoint", start)

    for name,data in data_dict.items():
        start = time.time()
        print(f'Generating file for {name}')
        generate_synthetic_data(args, model_biopm, gen_model, name, data, device)
        print(f'Finished generating file for {name}')
        log_time("Loading BioPM checkpoint", start)
    log_time("Total script runtime", total_start)
    

if __name__ == "__main__":
    main()
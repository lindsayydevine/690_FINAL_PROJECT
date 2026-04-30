import inspect

import torch
import torch.nn as nn

class BioPMAutoregressor(nn.Module):
    def __init__(self, token_dim=64, hidden_dim=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=token_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_proj = nn.Linear(hidden_dim, token_dim)

    def forward(self, x, h0=None):
        h, h_n = self.gru(x, h0)
        pred = self.output_proj(h)
        return pred, h_n


def _torch_load_compat(path, map_location="cpu"):
    """Load checkpoints without triggering the new torch.load warning when possible."""
    load_kwargs = {"map_location": map_location}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = True
    return torch.load(path, **load_kwargs)


def load_autoregressor_checkpoint(path, map_location="cpu"):
    """
    Restore a GRU token autoregressor from the checkpoint format used in this repo.

    Returns:
        model:      BioPMAutoregressor in eval mode
        checkpoint: raw checkpoint dict with metadata such as best_val_loss
    """
    checkpoint = _torch_load_compat(path, map_location=map_location)
    required_keys = {
        "model_state_dict",
        "token_dim",
        "hidden_dim",
        "num_layers",
        "dropout",
    }
    missing = required_keys - set(checkpoint.keys())
    if missing:
        raise KeyError(f"Checkpoint is missing required keys: {sorted(missing)}")

    model = BioPMAutoregressor(
        token_dim=checkpoint["token_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        num_layers=checkpoint["num_layers"],
        dropout=checkpoint["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(map_location)
    model.eval()
    return model, checkpoint

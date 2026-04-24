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
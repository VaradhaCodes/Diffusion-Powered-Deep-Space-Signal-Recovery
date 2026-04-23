"""Zhu et al. 2023 baseline model — faithful PyTorch port from the paper.

Architecture verified against Table 1 (Radio Science 2023):

  CNN path (channel-first input B × 2 × 800):
    Conv1d(2→64, k=11, same) → BN → ReLU
    Conv1d(64→8, k=11, same) → BN → ReLU
    MaxPool1d(2) → ReLU
    Flatten → (B, 3200)

  Bi-LSTM path (input transposed to B × 800 × 2):
    BiLSTM(2→32 per dir, sum-merge) → Tanh → Dropout(0.2)   → (B,800,32)
    BiLSTM(32→32 per dir, sum-merge) → ReLU                  → (B,800,32)
    Flatten → (B, 25600)

  Head  [Table 1 layer 13 confirms concat = 28,800]:
    Cat(3200, 25600) → FC(28800→2048) → ReLU → Dropout(0.08)
    FC(2048→1024) → ReLU → Dropout(0.08)
    FC(1024→100) → Sigmoid

Loss: MSE  |  Adam lr=1e-3  |  Batch 512  |  Epochs 40
"""

import torch
import torch.nn as nn


class _BiLSTMSumMerge(nn.Module):
    """Bidirectional LSTM where forward and backward outputs are summed.

    Output shape: (B, T, hidden_size)  — same dim as unidirectional.
    Matches Keras Bidirectional(LSTM(units), merge_mode='sum').
    """
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self._init_weights()

    def _init_weights(self):
        for name, p in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p.data)
            elif "bias" in name:
                p.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(x)                         # (B, T, 2*hidden)
        return h[:, :, :h.shape[2]//2] + h[:, :, h.shape[2]//2:]  # sum merge → (B, T, hidden)


class ZhuBaseline(nn.Module):
    def __init__(self, dropout_lstm: float = 0.2, dropout_fc: float = 0.08):
        super().__init__()

        # CNN branch
        self.cnn = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=11, padding=5),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 8, kernel_size=11, padding=5),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.ReLU(),
        )  # → (B, 8, 400) → flatten (B, 3200)

        # Bi-LSTM branch with sum-merge (output dim = hidden_size, not 2×)
        self.bilstm1 = _BiLSTMSumMerge(input_size=2, hidden_size=32)
        self.lstm_drop = nn.Dropout(dropout_lstm)
        self.bilstm2 = _BiLSTMSumMerge(input_size=32, hidden_size=32)

        # Head  (3200 + 25600 = 28800, matching Table 1 layer 13)
        self.head = nn.Sequential(
            nn.Linear(3200 + 25600, 2048),
            nn.ReLU(),
            nn.Dropout(dropout_fc),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_fc),
            nn.Linear(1024, 100),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cnn_feat = self.cnn(x).flatten(1)               # (B, 3200)

        xt = x.permute(0, 2, 1)                         # (B, 800, 2)
        h1 = torch.tanh(self.bilstm1(xt))               # (B, 800, 32)
        h1 = self.lstm_drop(h1)
        h2 = torch.relu(self.bilstm2(h1))               # (B, 800, 32)
        lstm_feat = h2.flatten(1)                        # (B, 25600)

        return self.head(torch.cat([cnn_feat, lstm_feat], dim=1))  # (B, 100)

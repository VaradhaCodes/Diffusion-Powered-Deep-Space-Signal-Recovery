"""Zhu et al. dataset loader.

Data layout (per-sample CSV files):
  train_dataset/data_awgn/mod_data/mod_signal_N.csv     → (800,2)
  train_dataset/data_awgn/label_data/label_signal_N.csv → (100,1)
  train_dataset/data_kb2/...
  test_dataset/test_data/<condition>/mod_signal_NNNN.csv
  test_dataset/test_label/<condition>/label_signal_NNNN.csv

Actual counts: 21000 AWGN + 21000 KB2 = 42000 train; 6 × 1400 = 8400 test.
x shape: (2, 800) float32   [channel first: I=row0, Q=row1]
y shape: (100,)  float32
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset, Subset

ZHU_ROOT = "/mnt/c/EM Project/data_set/data_set"

TEST_CONDITIONS = [
    "Awgn_Tb0d3",
    "Awgn_Tb0d5",
    "kb2_Tb0d3_m1d2",
    "kb2_Tb0d3_m1d4",
    "kb2_Tb0d5_m1d2",
    "kb2_Tb0d5_m1d4",
]


def _sniff_padding(path):
    first = sorted(os.listdir(path))[0]
    # e.g. mod_signal_0001.csv → 4, mod_signal_1.csv → 0
    stem = os.path.splitext(first)[0]
    num_part = stem.split("_")[-1]
    return len(num_part) if num_part[0] == "0" else 0


class _CSVPairDataset(Dataset):
    """One split dir of per-sample CSV pairs."""

    def __init__(self, iq_dir, lbl_dir, channel_label: int = -1):
        pad = _sniff_padding(iq_dir)
        fmt = f"{{:0{pad}d}}" if pad else "{:d}"
        n_iq = len(glob.glob(os.path.join(iq_dir, "*.csv")))
        self.iq_paths = [os.path.join(iq_dir, f"mod_signal_{fmt.format(i+1)}.csv") for i in range(n_iq)]
        self.lbl_paths = [os.path.join(lbl_dir, f"label_signal_{fmt.format(i+1)}.csv") for i in range(n_iq)]
        self.channel_label = channel_label  # 0=awgn, 1=kb2, -1=unknown

    def __len__(self):
        return len(self.iq_paths)

    def __getitem__(self, idx):
        iq = np.loadtxt(self.iq_paths[idx], delimiter=",", dtype=np.float32)  # (800,2)
        lbl = np.loadtxt(self.lbl_paths[idx], delimiter=",", dtype=np.float32)  # (100,) or (100,1)
        x = iq.T.copy()        # (2,800)
        y = lbl.reshape(100)   # (100,)
        return torch.from_numpy(x), torch.from_numpy(y)


def zhu_train_dataset(val_frac: float = 0.2, seed: int = 42):
    """Returns (train_ds, val_ds) over all 42 000 train samples."""
    awgn = _CSVPairDataset(
        os.path.join(ZHU_ROOT, "train_dataset/data_awgn/mod_data"),
        os.path.join(ZHU_ROOT, "train_dataset/data_awgn/label_data"),
        channel_label=0,
    )
    kb2 = _CSVPairDataset(
        os.path.join(ZHU_ROOT, "train_dataset/data_kb2/mod_data"),
        os.path.join(ZHU_ROOT, "train_dataset/data_kb2/label_data"),
        channel_label=1,
    )
    full = ConcatDataset([awgn, kb2])
    n = len(full)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    split = int(n * (1 - val_frac))
    return Subset(full, idx[:split].tolist()), Subset(full, idx[split:].tolist())


def zhu_test_dataset(condition: str | None = None):
    """Returns Dataset for one or all test conditions.

    condition: None → all 8400 samples; else one of TEST_CONDITIONS.
    """
    conditions = [condition] if condition else TEST_CONDITIONS
    parts = []
    for cond in conditions:
        parts.append(_CSVPairDataset(
            os.path.join(ZHU_ROOT, f"test_dataset/test_data/{cond}"),
            os.path.join(ZHU_ROOT, f"test_dataset/test_label/{cond}"),
        ))
    return ConcatDataset(parts) if len(parts) > 1 else parts[0]

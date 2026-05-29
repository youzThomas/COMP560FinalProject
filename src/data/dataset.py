"""Mill dataset loader with known / unknown splits for open-world training.

The UC Berkeley Mill dataset is provided as a MATLAB struct array where each cut
carries several 1-D sensor channels. Cuts are annotated with a tool-wear class in
``labels_with_tool_class.csv`` (0 = Healthy, 1 = Degraded, 2 = Failed).

For the Object Newness framework we treat ``class 2`` (Failed) as the held-out
novel category. The training set only contains ``known_classes`` (0, 1); the
validation and test sets include all classes so we can measure both known-class
recall and unknown-class discovery.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import scipy.io as sio
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


@dataclass
class WindowedArrays:
    X: np.ndarray          # [N, T, C]
    y: np.ndarray          # [N] original tool_class labels
    cut_no: np.ndarray     # [N] cut index (for group-aware diagnostics)


def _load_windows(
    mat_path: str | Path,
    labels_csv: str | Path,
    window_size: int,
    stride: int,
    drop_bad_cuts: Sequence[int] | None,
) -> WindowedArrays:
    data = sio.loadmat(str(mat_path), struct_as_record=True)["mill"]
    df = pd.read_csv(labels_csv)
    if drop_bad_cuts:
        df = df.drop(index=list(drop_bad_cuts), errors="ignore")

    signal_names = data.dtype.names[7:]

    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    cut_list: list[int] = []

    for _, row in df.iterrows():
        cut_no = int(row["cut_no"])
        tool_class = int(row["tool_class"])
        if cut_no >= data.shape[1]:
            continue
        cut = data[0, cut_no]
        try:
            signals = np.column_stack([cut[s].flatten() for s in signal_names])
        except Exception:
            continue
        if signals.shape[0] < window_size:
            continue
        for start in range(0, signals.shape[0] - window_size + 1, stride):
            X_list.append(signals[start : start + window_size])
            y_list.append(tool_class)
            cut_list.append(cut_no)

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.int64)
    cuts = np.asarray(cut_list, dtype=np.int64)
    return WindowedArrays(X=X, y=y, cut_no=cuts)


def _minmax_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return X.min(axis=(0, 1)), X.max(axis=(0, 1))


def _minmax_apply(X: np.ndarray, mn: np.ndarray, mx: np.ndarray) -> np.ndarray:
    return (X - mn) / (mx - mn + 1e-8)


class MillWindowDataset(Dataset):
    """Windowed tensors produced from the Mill .mat file.

    Each sample carries the original tool_class in ``y_orig`` and a remapped
    known-class index in ``y`` (``-1`` when the sample belongs to an unknown
    class — downstream code uses this to compute open-world metrics).
    """

    def __init__(
        self,
        X: np.ndarray,
        y_orig: np.ndarray,
        known_to_idx: dict[int, int],
    ) -> None:
        self.X = torch.from_numpy(X).float()
        self.y_orig = torch.from_numpy(y_orig).long()
        self.y = torch.tensor(
            [known_to_idx.get(int(c), -1) for c in y_orig], dtype=torch.long
        )

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "x": self.X[idx],        # [T, C]
            "y": self.y[idx],        # known index or -1
            "y_orig": self.y_orig[idx],
        }


def build_dataloaders(cfg) -> dict:
    """Build train / val / test dataloaders + metadata.

    Training data only includes ``known_classes``. Validation and test contain
    both known and unknown samples.
    """
    data_cfg = cfg.data
    arrays = _load_windows(
        mat_path=data_cfg.mat_path,
        labels_csv=data_cfg.labels_csv,
        window_size=int(data_cfg.window_size),
        stride=int(data_cfg.stride),
        drop_bad_cuts=data_cfg.get("drop_bad_cuts", []),
    )
    X, y = arrays.X, arrays.y

    known_classes = list(map(int, data_cfg.known_classes))
    unknown_classes = list(map(int, data_cfg.unknown_classes))
    known_to_idx = {c: i for i, c in enumerate(known_classes)}

    # Split in two stages: first hold out unknown-only test pool, then split
    # known samples into train/val/test. Finally the unknowns are merged into
    # val/test so the open-world metrics reflect mixed inputs.
    is_known = np.isin(y, known_classes)
    X_known, y_known = X[is_known], y[is_known]
    X_unknown, y_unknown = X[~is_known], y[~is_known]

    val_frac = float(data_cfg.val_fraction)
    test_frac = float(data_cfg.test_fraction)
    assert 0.0 < val_frac + test_frac < 1.0, "val+test fractions must be in (0, 1)"

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X_known, y_known,
        test_size=val_frac + test_frac,
        random_state=cfg.seed,
        stratify=y_known,
    )
    rel_test = test_frac / (val_frac + test_frac)
    X_val_k, X_te_k, y_val_k, y_te_k = train_test_split(
        X_tmp, y_tmp,
        test_size=rel_test,
        random_state=cfg.seed + 1,
        stratify=y_tmp,
    )

    # Split unknowns roughly in half between val and test.
    if len(X_unknown) > 1:
        X_val_u, X_te_u, y_val_u, y_te_u = train_test_split(
            X_unknown, y_unknown,
            test_size=0.5,
            random_state=cfg.seed + 2,
            stratify=y_unknown if len(np.unique(y_unknown)) > 1 else None,
        )
    elif len(X_unknown) == 1:
        X_val_u, y_val_u = X_unknown, y_unknown
        X_te_u, y_te_u = np.empty((0, *X_unknown.shape[1:]), dtype=X.dtype), np.empty(0, dtype=y.dtype)
    else:
        X_val_u = np.empty((0, *X.shape[1:]), dtype=X.dtype)
        y_val_u = np.empty(0, dtype=y.dtype)
        X_te_u = X_val_u
        y_te_u = y_val_u

    # Fit min-max on training (known only) data and apply everywhere.
    mn, mx = _minmax_fit(X_tr)
    X_tr = _minmax_apply(X_tr, mn, mx)
    X_val = _minmax_apply(np.concatenate([X_val_k, X_val_u], axis=0), mn, mx)
    y_val = np.concatenate([y_val_k, y_val_u], axis=0)
    X_te = _minmax_apply(np.concatenate([X_te_k, X_te_u], axis=0), mn, mx)
    y_te = np.concatenate([y_te_k, y_te_u], axis=0)

    train_ds = MillWindowDataset(X_tr, y_tr, known_to_idx)
    val_ds = MillWindowDataset(X_val, y_val, known_to_idx)
    test_ds = MillWindowDataset(X_te, y_te, known_to_idx)

    batch = int(data_cfg.batch_size)
    workers = int(data_cfg.get("num_workers", 0))
    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds, batch_size=batch, shuffle=True, num_workers=workers,
        pin_memory=pin, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=pin,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=pin,
    )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "n_channels": int(X.shape[-1]),
        "window_size": int(X.shape[1]),
        "known_classes": known_classes,
        "unknown_classes": unknown_classes,
        "known_to_idx": known_to_idx,
        "scaler": {"min": mn.astype(np.float32), "max": mx.astype(np.float32)},
        "class_counts": {
            "train": {int(k): int(v) for k, v in zip(*np.unique(y_tr, return_counts=True))},
            "val": {int(k): int(v) for k, v in zip(*np.unique(y_val, return_counts=True))},
            "test": {int(k): int(v) for k, v in zip(*np.unique(y_te, return_counts=True))},
        },
    }

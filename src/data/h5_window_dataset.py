"""
Streaming HDF5 dataset utilities for large preprocessed BioPM window corpora.

These datasets avoid loading every preprocessed window into RAM at once.
They are a better fit for Capture24-scale decoder training.
"""

from __future__ import annotations

import bisect
import os
import re
from typing import Dict, List, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


ME_PATTERN = re.compile(r"Data_MeLabel_.*\.h5$")
ACC_PATTERN = re.compile(r"Data_AccLabel_.*\.h5$")


def list_me_h5_files(data_root: str) -> List[str]:
    """Return sorted Data_MeLabel files from a directory."""
    files = []
    for fname in sorted(os.listdir(data_root)):
        if ME_PATTERN.match(fname):
            files.append(os.path.join(data_root, fname))
    if not files:
        raise FileNotFoundError(f"No Data_MeLabel_*.h5 files found in {data_root}")
    return files


def list_acc_h5_files(data_root: str) -> List[str]:
    """Return sorted Data_AccLabel files from a directory."""
    files = []
    for fname in sorted(os.listdir(data_root)):
        if ACC_PATTERN.match(fname):
            files.append(os.path.join(data_root, fname))
    if not files:
        raise FileNotFoundError(f"No Data_AccLabel_*.h5 files found in {data_root}")
    return files


def parse_subject_id(path: str) -> int:
    """Extract a numeric subject id from a BioPM per-subject filename."""
    fname = os.path.basename(path)
    match = re.search(r"Data_(?:MeLabel|AccLabel)_([A-Za-z]*)(\d+)\.h5$", fname)
    if not match:
        raise ValueError(f"Could not parse subject id from {path}")
    return int(match.group(2))


def list_stage2_file_pairs(data_root: str) -> List[tuple[str, str]]:
    """
    Match Data_MeLabel and Data_AccLabel files by subject id.

    Returns:
      List of (me_path, acc_path) tuples sorted by subject id.
    """
    me_by_subject = {
        parse_subject_id(path): path for path in list_me_h5_files(data_root)
    }
    acc_by_subject = {
        parse_subject_id(path): path for path in list_acc_h5_files(data_root)
    }

    missing_acc = sorted(set(me_by_subject) - set(acc_by_subject))
    missing_me = sorted(set(acc_by_subject) - set(me_by_subject))
    if missing_acc or missing_me:
        raise FileNotFoundError(
            "Could not form complete Me/Acc file pairs. "
            f"Missing acc subjects: {missing_acc}; missing me subjects: {missing_me}"
        )

    return [
        (me_by_subject[subject_id], acc_by_subject[subject_id])
        for subject_id in sorted(me_by_subject)
    ]


class Stage1WindowDataset(Dataset):
    """
    Lazily streams x_acc_filt windows from preprocessed HDF5 files.

    Each item returns:
      x_acc_filt: (L, 38) float tensor
      label:      scalar float tensor
      pid:        scalar long tensor
    """

    def __init__(self, file_paths: Sequence[str], max_windows: int | None = None):
        self.file_paths = list(file_paths)
        if not self.file_paths:
            raise ValueError("Stage1WindowDataset requires at least one HDF5 file")

        self.subject_ids = [parse_subject_id(path) for path in self.file_paths]
        self.lengths = []
        for path in self.file_paths:
            with h5py.File(path, "r") as hf:
                self.lengths.append(int(hf["x_acc_filt"].shape[0]))

        self.cumulative = np.cumsum(self.lengths, dtype=np.int64)
        self.max_windows = min(int(max_windows), int(self.cumulative[-1])) \
            if max_windows is not None else None
        self._handles: Dict[str, h5py.File] = {}

    def __len__(self) -> int:
        return self.max_windows if self.max_windows is not None else int(self.cumulative[-1])

    def _get_handle(self, path: str) -> h5py.File:
        handle = self._handles.get(path)
        if handle is None:
            handle = h5py.File(path, "r")
            self._handles[path] = handle
        return handle

    def _locate_index(self, idx: int):
        file_idx = bisect.bisect_right(self.cumulative, idx)
        prev = 0 if file_idx == 0 else int(self.cumulative[file_idx - 1])
        local_idx = idx - prev
        return file_idx, local_idx

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        file_idx, local_idx = self._locate_index(idx)
        path = self.file_paths[file_idx]
        pid = self.subject_ids[file_idx]
        hf = self._get_handle(path)

        x_acc_filt = np.array(hf["x_acc_filt"][local_idx], dtype=np.float32)
        label = np.float32(hf["window_label"][local_idx])

        return {
            "x_acc_filt": torch.from_numpy(x_acc_filt),
            "label": torch.tensor(label, dtype=torch.float32),
            "pid": torch.tensor(pid, dtype=torch.long),
        }

    def close(self):
        """Close any lazily-opened HDF5 handles."""
        for handle in self._handles.values():
            try:
                handle.close()
            except Exception:
                pass
        self._handles.clear()

    def __del__(self):
        self.close()


class Stage2WindowDataset(Dataset):
    """
    Lazily stream paired x_acc_filt and window_acc_filt_gravity windows.

    Each item returns:
      x_acc_filt:               (L, 38) float tensor
      window_acc_filt_gravity:  (300, 6) float tensor
      label:                    scalar float tensor
      pid:                      scalar long tensor
    """

    def __init__(self, file_pairs: Sequence[tuple[str, str]],
                 max_windows: int | None = None):
        self.file_pairs = list(file_pairs)
        if not self.file_pairs:
            raise ValueError("Stage2WindowDataset requires at least one file pair")

        self.subject_ids = [parse_subject_id(me_path) for me_path, _ in self.file_pairs]
        self.lengths = []
        for me_path, acc_path in self.file_pairs:
            with h5py.File(me_path, "r") as me_hf, h5py.File(acc_path, "r") as acc_hf:
                n_me = int(me_hf["x_acc_filt"].shape[0])
                n_acc = int(acc_hf["window_acc_filt_gravity"].shape[0])
                if n_me != n_acc:
                    raise ValueError(
                        f"Mismatched window counts for {me_path} and {acc_path}: "
                        f"{n_me} vs {n_acc}"
                    )
                self.lengths.append(n_me)

        self.cumulative = np.cumsum(self.lengths, dtype=np.int64)
        self.max_windows = min(int(max_windows), int(self.cumulative[-1])) \
            if max_windows is not None else None
        self._handles: Dict[str, h5py.File] = {}

    def __len__(self) -> int:
        return self.max_windows if self.max_windows is not None else int(self.cumulative[-1])

    def _get_handle(self, path: str) -> h5py.File:
        handle = self._handles.get(path)
        if handle is None:
            handle = h5py.File(path, "r")
            self._handles[path] = handle
        return handle

    def _locate_index(self, idx: int):
        file_idx = bisect.bisect_right(self.cumulative, idx)
        prev = 0 if file_idx == 0 else int(self.cumulative[file_idx - 1])
        local_idx = idx - prev
        return file_idx, local_idx

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        file_idx, local_idx = self._locate_index(idx)
        me_path, acc_path = self.file_pairs[file_idx]
        pid = self.subject_ids[file_idx]
        me_hf = self._get_handle(me_path)
        acc_hf = self._get_handle(acc_path)

        x_acc_filt = np.array(me_hf["x_acc_filt"][local_idx], dtype=np.float32)
        window_acc_filt_gravity = np.array(
            acc_hf["window_acc_filt_gravity"][local_idx], dtype=np.float32)
        label = np.float32(me_hf["window_label"][local_idx])

        return {
            "x_acc_filt": torch.from_numpy(x_acc_filt),
            "window_acc_filt_gravity": torch.from_numpy(window_acc_filt_gravity),
            "label": torch.tensor(label, dtype=torch.float32),
            "pid": torch.tensor(pid, dtype=torch.long),
        }

    def close(self):
        """Close any lazily-opened HDF5 handles."""
        for handle in self._handles.values():
            try:
                handle.close()
            except Exception:
                pass
        self._handles.clear()

    def __del__(self):
        self.close()

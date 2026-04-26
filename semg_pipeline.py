"""
Modular sEMG training pipeline for NinaPro DB2 (Exercise 3).

Design goals
------------
- Prevent data leakage by construction.
- Keep architecture/preprocessing pluggable.
- Support two protocols:
  1) RepCV (within-subject): train/val/test repetitions per subject.
  2) LOSO (cross-subject): held-out test subject + independent val subject.

Note: This module defines pipeline structure only. It does not auto-run training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.io
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset


# -----------------------------
# Configs
# -----------------------------


@dataclass
class BaseConfig:
    base_path: str
    num_subjects: int = 40
    in_channels: int = 12
    num_classes: int = 24
    window_size: int = 400
    stride: int = 200
    purity_threshold: float = 0.80
    batch_size: int = 256
    num_workers: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class RepCVSplit:
    train_reps: Sequence[int] = (1, 3, 4)
    val_reps: Sequence[int] = (6,)
    test_reps: Sequence[int] = (2, 5)


@dataclass
class RunConfig(BaseConfig):
    protocol: str = "repcv"  # repcv | loso
    repcv_split: RepCVSplit = field(default_factory=RepCVSplit)
    epochs: int = 50
    lr: float = 3e-4
    weight_decay: float = 1e-2
    label_smoothing: float = 0.1
    grad_clip: float = 1.0


# -----------------------------
# Data I/O and preprocessing
# -----------------------------


def load_subject(base_path: str, subject_id: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one subject MAT file and return (emg, labels, repetitions)."""
    mat_path = Path(base_path) / f"DB2_s{subject_id}" / f"DB2_s{subject_id}" / f"S{subject_id}_E2_A1.mat"
    if not mat_path.exists():
        raise FileNotFoundError(f"Missing file: {mat_path}")

    mat = scipy.io.loadmat(str(mat_path), squeeze_me=True)
    emg = mat.get("emg", mat.get("EMG"))
    if emg is None:
        raise KeyError(f"No EMG field in {mat_path}")

    emg = np.nan_to_num(emg.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if emg.ndim == 2 and emg.shape[0] == 12 and emg.shape[1] != 12:
        emg = emg.T
    if emg.shape[1] != 12:
        raise ValueError(f"Unexpected emg shape {emg.shape} for subject {subject_id}")

    label_key = "restimulus" if "restimulus" in mat else "stimulus"
    rep_key = "rerepetition" if "rerepetition" in mat else "repetition"

    raw_labels = mat[label_key].astype(np.int32).ravel()
    reps = mat[rep_key].astype(np.int32).ravel()

    # map active classes to contiguous ids
    labels = np.zeros_like(raw_labels)
    uniq = np.unique(raw_labels[raw_labels > 0])
    for new_id, old_id in enumerate(sorted(uniq), start=1):
        labels[raw_labels == old_id] = new_id

    n = min(len(emg), len(labels), len(reps))
    return emg[:n], labels[:n], reps[:n]


def create_windows(data: np.ndarray, labels: np.ndarray, window: int, stride: int, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    """Simple sliding window with majority-vote purity threshold."""
    n = len(data)
    max_w = (n - window) // stride + 1
    if max_w <= 0:
        return np.empty((0, window, data.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.int64)

    xs: List[np.ndarray] = []
    ys: List[int] = []
    for i in range(max_w):
        s = i * stride
        e = s + window
        y_seg = labels[s:e]
        vals, counts = np.unique(y_seg, return_counts=True)
        best = int(vals[np.argmax(counts)])
        purity = counts.max() / window
        if purity >= threshold:
            xs.append(data[s:e])
            ys.append(best)

    if not xs:
        return np.empty((0, window, data.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.stack(xs).astype(np.float32), np.array(ys, dtype=np.int64)


class Preprocessor:
    """Inject your own preprocessing by subclassing or replacing this callable."""

    def __call__(self, emg: np.ndarray) -> np.ndarray:
        return emg.astype(np.float32)


# -----------------------------
# Splitting (leak-safe)
# -----------------------------


def split_repcv(reps: np.ndarray, split: RepCVSplit) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    tr = np.isin(reps, list(split.train_reps))
    va = np.isin(reps, list(split.val_reps))
    te = np.isin(reps, list(split.test_reps))
    if np.any(va & te) or np.any(tr & te) or np.any(tr & va):
        raise ValueError("RepCV split overlap detected.")
    return tr, va, te


def loso_subject_sets(subject_ids: Sequence[int], test_subject: int, val_subject: int) -> Tuple[List[int], int, int]:
    if test_subject == val_subject:
        raise ValueError("LOSO requires val_subject != test_subject")
    train_subjects = [s for s in subject_ids if s not in (test_subject, val_subject)]
    return train_subjects, val_subject, test_subject


# -----------------------------
# Dataset builders
# -----------------------------


def to_tensor_dataset(x: np.ndarray, y: np.ndarray) -> TensorDataset:
    t = torch.from_numpy(x).float().permute(0, 2, 1)  # (N,C,T)
    mu = t.mean(dim=2, keepdim=True)
    sd = t.std(dim=2, keepdim=True) + 1e-6
    t = (t - mu) / sd
    return TensorDataset(t, torch.from_numpy(y).long())


def build_loader(ds: Dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


# -----------------------------
# Trainer (validation-driven only)
# -----------------------------


class Trainer:
    """
    Ensures checkpoint selection is based ONLY on validation accuracy.
    Test is evaluated only after training.
    """

    def __init__(self, model: nn.Module, cfg: RunConfig):
        self.model = model.to(cfg.device)
        self.cfg = cfg
        self.criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
        self.optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    def _one_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        loss_sum = 0.0
        n = 0
        for xb, yb in loader:
            xb, yb = xb.to(self.cfg.device), yb.to(self.cfg.device)
            self.optim.zero_grad(set_to_none=True)
            loss = self.criterion(self.model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.optim.step()
            loss_sum += float(loss.item())
            n += 1
        return loss_sum / max(n, 1)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        total = 0
        correct = 0
        for xb, yb in loader:
            xb, yb = xb.to(self.cfg.device), yb.to(self.cfg.device)
            pred = self.model(xb).argmax(1)
            correct += int((pred == yb).sum().item())
            total += int(yb.size(0))
        return 100.0 * correct / max(total, 1)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, float]:
        best_val = -1.0
        best_state = None
        for _ in range(self.cfg.epochs):
            self._one_epoch(train_loader)
            val_acc = self.evaluate(val_loader)
            if val_acc > best_val:
                best_val = val_acc
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return {"best_val_acc": best_val}


# -----------------------------
# Extension hooks
# -----------------------------


ModelFactory = Callable[[RunConfig], nn.Module]
PreprocessFactory = Callable[[], Preprocessor]


@dataclass
class ArchitectureConfig:
    """
    Variant switches so you can ablate architecture pieces easily.

    Baseline (concat): use_branch_se=False, use_cross_scale=False
    BranchSE only    : use_branch_se=True,  use_cross_scale=False
    CrossScale only  : use_branch_se=False, use_cross_scale=True
    Full model       : use_branch_se=True,  use_cross_scale=True
    """

    compressed: bool = False
    use_branch_se: bool = True
    use_cross_scale: bool = True
    kernels: Tuple[int, int, int] = (7, 15, 31)
    num_classes: int = 24
    in_channels: int = 12
    reduction: int = 8


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.mean(-1)  # [B,C]
        z = torch.nn.functional.gelu(self.fc1(z))
        z = torch.sigmoid(self.fc2(z))
        return x * z.unsqueeze(-1)


class BranchSEBlock(nn.Module):
    """Depthwise-separable branch with optional SE."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        use_branch_se: bool = True,
        reduction: int = 8,
    ):
        super().__init__()
        pad = kernel_size // 2
        self.dw = nn.Conv1d(in_ch, in_ch, kernel_size, padding=pad, groups=in_ch, bias=False)
        self.pw = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.use_branch_se = use_branch_se
        self.se = SEBlock(out_ch, reduction=reduction) if use_branch_se else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dw(x)
        out = self.pw(out)
        out = self.bn(out)
        out = torch.nn.functional.gelu(out)
        out = self.se(out)
        return out


class CrossScaleGating(nn.Module):
    """Adaptive per-branch channel gates. Can be disabled as identity-concat."""

    def __init__(self, channels_per_branch: int, num_branches: int = 3, enabled: bool = True):
        super().__init__()
        total = channels_per_branch * num_branches
        self.enabled = enabled
        self.num_branches = num_branches
        if enabled:
            self.fc1 = nn.Linear(total, max(total // 2, 8))
            self.fc2 = nn.Linear(max(total // 2, 8), total)
        else:
            self.fc1 = nn.Identity()
            self.fc2 = nn.Identity()

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        if not self.enabled:
            return torch.cat(feats, dim=1)

        global_vec = torch.cat([f.mean(-1) for f in feats], dim=1)
        gates = torch.nn.functional.gelu(self.fc1(global_vec))
        gates = torch.sigmoid(self.fc2(gates))
        b = feats[0].shape[0]
        c = feats[0].shape[1]
        gates = gates.view(b, self.num_branches, c)
        out = []
        for i, f in enumerate(feats):
            out.append(f * gates[:, i, :].unsqueeze(-1))
        return torch.cat(out, dim=1)


class DepthSepResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.dw = nn.Conv1d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False)
        self.pw = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.res_conv = nn.Conv1d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else None
        self.res_bn = nn.BatchNorm1d(out_ch) if in_ch != out_ch else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = self.bn(self.pw(self.dw(x)))
        if self.res_conv is not None:
            res = self.res_bn(self.res_conv(res))
        return self.act(out + res)


class MSTSEMGNet(nn.Module):
    """
    Variant-friendly MSTSEMGNet.
    Toggle BranchSE, CrossScale, and compressed width from ArchitectureConfig.
    """

    def __init__(self, arch: ArchitectureConfig):
        super().__init__()
        ch0, ch1 = arch.in_channels, 32
        branch_ch = 48 if not arch.compressed else 32
        out_ch = branch_ch * len(arch.kernels)
        ds1_out = 192 if not arch.compressed else 128
        ds2_out = 128 if not arch.compressed else 64
        ds3_out = 64 if not arch.compressed else 32
        head_h = 128 if not arch.compressed else 64
        head2 = 64 if not arch.compressed else 32

        self.spatial = nn.Conv1d(ch0, ch1, 1, bias=False)
        self.bn0 = nn.BatchNorm1d(ch1)
        self.branches = nn.ModuleList(
            [
                BranchSEBlock(
                    ch1,
                    branch_ch,
                    k,
                    use_branch_se=arch.use_branch_se,
                    reduction=arch.reduction,
                )
                for k in arch.kernels
            ]
        )
        self.gating = CrossScaleGating(
            channels_per_branch=branch_ch,
            num_branches=len(arch.kernels),
            enabled=arch.use_cross_scale,
        )
        self.ds1 = DepthSepResBlock(out_ch, ds1_out)
        self.ds2 = DepthSepResBlock(ds1_out, ds2_out)
        self.ds3 = DepthSepResBlock(ds2_out, ds3_out)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(ds3_out, head_h)
        self.bn1 = nn.BatchNorm1d(head_h)
        self.fc2 = nn.Linear(head_h, head2)
        self.bn2 = nn.BatchNorm1d(head2)
        self.fc3 = nn.Linear(head2, arch.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.gelu(self.bn0(self.spatial(x)))
        feats = [b(x) for b in self.branches]
        x = self.gating(feats)
        x = self.ds1(x)
        x = self.ds2(x)
        x = self.ds3(x)
        x = self.gap(x).squeeze(-1)
        x = torch.nn.functional.gelu(self.bn1(self.fc1(x)))
        x = torch.nn.functional.dropout(x, p=0.5, training=self.training)
        x = torch.nn.functional.gelu(self.bn2(self.fc2(x)))
        x = torch.nn.functional.dropout(x, p=0.5, training=self.training)
        return self.fc3(x)


def build_architecture_variants() -> Dict[str, ArchitectureConfig]:
    """
    Predefined variants for your ablation table.
    """
    return {
        "baseline_concat": ArchitectureConfig(
            compressed=False, use_branch_se=False, use_cross_scale=False
        ),
        "branch_se_only": ArchitectureConfig(
            compressed=False, use_branch_se=True, use_cross_scale=False
        ),
        "cross_scale_only": ArchitectureConfig(
            compressed=False, use_branch_se=False, use_cross_scale=True
        ),
        "branch_se_plus_cross_scale": ArchitectureConfig(
            compressed=False, use_branch_se=True, use_cross_scale=True
        ),
        "compressed_full": ArchitectureConfig(
            compressed=True, use_branch_se=True, use_cross_scale=True
        ),
    }


def make_model_from_arch(cfg: RunConfig, arch: ArchitectureConfig) -> nn.Module:
    _ = cfg
    return MSTSEMGNet(arch)


def run_protocol(cfg: RunConfig, make_model: ModelFactory, make_preprocessor: PreprocessFactory) -> None:
    """
    Skeleton entrypoint:
      - You plug your model and preprocessing strategy.
      - You implement desired logging/checkpointing around this.
      - No auto-execution in this file.
    """
    _ = cfg, make_model, make_preprocessor
    raise NotImplementedError("Wire your specific RepCV/LOSO loop here using the above building blocks.")

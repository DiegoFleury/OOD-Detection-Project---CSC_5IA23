"""
Neural Collapse Analysis Module
================================
Computes NC1–NC4 metrics across training checkpoints for a ResNet model.

Usage (from notebook):
    from src.neural_collapse.nc_analysis import (
        load_checkpoints_and_analyze,
        plot_nc_evolution,
        plot_nc_individual,
        save_metrics_yaml,
        NCMetricsTracker,
    )

    tracker = load_checkpoints_and_analyze(
        checkpoint_dir="checkpoints/",
        model_class=ResNet18,
        loader=train_loader,
        device="cuda",
        num_classes=100,
    )
    fig = plot_nc_evolution(tracker, save_dir="figures/nc/")
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional, Type

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from scipy.sparse.linalg import svds
from tqdm import tqdm


# ==========================================================================
# NCMetricsTracker — stores NC metrics across epochs
# ==========================================================================

@dataclass
class NCMetricsTracker:
    """Accumulates Neural Collapse metrics over training epochs."""

    epochs: List[int] = field(default_factory=list)
    accuracy: List[float] = field(default_factory=list)
    loss: List[float] = field(default_factory=list)

    # NC1: within-class variability collapse
    Sw_invSb: List[float] = field(default_factory=list)

    # NC2: convergence to simplex ETF
    norm_M_CoV: List[float] = field(default_factory=list)   # equinorm (means)
    norm_W_CoV: List[float] = field(default_factory=list)   # equinorm (classifier)
    cos_M: List[float] = field(default_factory=list)        # equiangularity (means)
    cos_W: List[float] = field(default_factory=list)        # equiangularity (classifier)

    # NC3: self-duality  (W ≈ M)
    W_M_dist: List[float] = field(default_factory=list)

    # NC4: simplification to NCC
    NCC_mismatch: List[float] = field(default_factory=list)

    # ------------------------------------------------------------------
    def summary(self) -> str:
        """Return a human-readable summary of the tracked metrics."""
        lines = [
            "=" * 60,
            "NEURAL COLLAPSE METRICS SUMMARY",
            "=" * 60,
            f"Epochs analyzed: {len(self.epochs)}",
        ]
        if self.epochs:
            lines.append(f"Epoch range: {self.epochs[0]} → {self.epochs[-1]}")
        if self.accuracy:
            lines.append(f"Final accuracy: {self.accuracy[-1]:.4f}")
        if self.Sw_invSb:
            lines.append(f"Final NC1 Tr[Σ_W Σ_B† / C]: {self.Sw_invSb[-1]:.4f}")
        if self.norm_M_CoV:
            lines.append(f"Final NC2 equinorm (M CoV): {self.norm_M_CoV[-1]:.4f}")
        if self.cos_M:
            lines.append(f"Final NC2 equiangularity (cos M): {self.cos_M[-1]:.4f}")
        if self.W_M_dist:
            lines.append(f"Final NC3 (W-M dist): {self.W_M_dist[-1]:.4f}")
        if self.NCC_mismatch:
            lines.append(f"Final NC4 (NCC mismatch): {self.NCC_mismatch[-1]:.4f}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Export all metrics as a plain dictionary."""
        return {
            "epochs": list(self.epochs),
            "accuracy": list(self.accuracy),
            "loss": list(self.loss),
            "Sw_invSb": list(self.Sw_invSb),
            "norm_M_CoV": list(self.norm_M_CoV),
            "norm_W_CoV": list(self.norm_W_CoV),
            "cos_M": list(self.cos_M),
            "cos_W": list(self.cos_W),
            "W_M_dist": list(self.W_M_dist),
            "NCC_mismatch": list(self.NCC_mismatch),
        }


# ==========================================================================
# Feature extraction helpers
# ==========================================================================

class _FeatureHook:
    """Forward-hook that captures the *input* to the last linear layer."""

    def __init__(self):
        self.features: Optional[torch.Tensor] = None

    def __call__(self, module, inp, out):
        self.features = inp[0].detach()


def _find_classifier(model: nn.Module) -> nn.Linear:
    """Find the last Linear layer (classifier head) of the model.

    Tries common attribute names used by different ResNet implementations:
    ``fc``, ``linear``, ``classifier``, ``head``.
    Falls back to scanning all modules for the last ``nn.Linear``.
    """
    for attr in ("fc", "linear", "classifier", "head"):
        layer = getattr(model, attr, None)
        if isinstance(layer, nn.Linear):
            return layer

    # Fallback: find last nn.Linear in the model
    last_linear = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last_linear = m
    if last_linear is not None:
        return last_linear

    raise RuntimeError(
        "Could not find the classifier (nn.Linear) layer in the model. "
        "Please ensure the model has an attribute named 'fc', 'linear', "
        "'classifier', or 'head'."
    )


def _load_checkpoint(path: str, model: nn.Module, device: str) -> None:
    """Load a checkpoint into *model*, handling various checkpoint formats."""
    ckpt = torch.load(path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict):
        # Try common keys
        for key in ("model_state_dict", "state_dict", "model"):
            if key in ckpt:
                model.load_state_dict(ckpt[key])
                return
        # Maybe the dict IS the state_dict
        try:
            model.load_state_dict(ckpt)
        except Exception:
            raise RuntimeError(
                f"Cannot parse checkpoint at {path}. "
                f"Keys found: {list(ckpt.keys())}"
            )
    else:
        # Assume it's a raw state_dict
        model.load_state_dict(ckpt)


# ==========================================================================
# Core NC metric computation for a single checkpoint
# ==========================================================================

@torch.no_grad()
def _compute_nc_metrics(
    model: nn.Module,
    classifier: nn.Linear,
    hook: _FeatureHook,
    loader: torch.utils.data.DataLoader,
    device: str,
    num_classes: int,
) -> dict:
    """Run a full forward pass over *loader* and return NC1–NC4 metrics."""
    model.eval()

    N = [0] * num_classes
    mean = [None] * num_classes
    Sw = 0.0  # within-class covariance (accumulated)

    loss_sum = 0.0
    total_samples = 0
    net_correct = 0
    NCC_match_net = 0

    # ---- Pass 1: compute class means & loss ----
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        h = hook.features.view(images.shape[0], -1)  # (B, D)

        loss_sum += F.cross_entropy(outputs, targets, reduction="sum").item()
        total_samples += targets.shape[0]

        for c in range(num_classes):
            idxs = (targets == c).nonzero(as_tuple=True)[0]
            if len(idxs) == 0:
                continue
            h_c = h[idxs]
            if mean[c] is None:
                mean[c] = torch.zeros(h.shape[1], device=device)
            mean[c] += h_c.sum(dim=0)
            N[c] += len(idxs)

    # Finalize means
    for c in range(num_classes):
        if N[c] > 0 and mean[c] is not None:
            mean[c] /= N[c]
        else:
            mean[c] = torch.zeros_like(mean[0] if mean[0] is not None else torch.zeros(1))

    M = torch.stack(mean)  # (C, D)

    # ---- Pass 2: within-class covariance, accuracy, NCC ----
    D = M.shape[1]
    Sw = torch.zeros(D, D, device=device)

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        h = hook.features.view(images.shape[0], -1)

        # Accuracy
        preds = outputs.argmax(dim=1)
        net_correct += (preds == targets).sum().item()

        # NCC prediction
        # distances: (B, C)
        dists = torch.cdist(h, M)  # (B, C)
        ncc_preds = dists.argmin(dim=1)
        NCC_match_net += (ncc_preds == preds).sum().item()

        for c in range(num_classes):
            idxs = (targets == c).nonzero(as_tuple=True)[0]
            if len(idxs) == 0:
                continue
            h_c = h[idxs]
            z = h_c - mean[c].unsqueeze(0)
            Sw += z.T @ z

    total_N = sum(N)
    Sw /= total_N

    # ---- Derived quantities ----
    M_T = M.T  # (D, C)
    muG = M_T.mean(dim=1, keepdim=True)  # global mean (D, 1)
    M_ = M_T - muG  # centered class means (D, C)
    Sb = (M_ @ M_.T) / num_classes  # between-class covariance

    W = classifier.weight.detach().to(device)  # (C, D)

    # NC2: equinorm
    M_norms = torch.norm(M_, dim=0)  # per-class mean norms
    W_norms = torch.norm(W, dim=1)   # per-class classifier norms

    norm_M_CoV = (torch.std(M_norms) / torch.mean(M_norms)).item()
    norm_W_CoV = (torch.std(W_norms) / torch.mean(W_norms)).item()

    # NC2: equiangularity (coherence)
    def coherence(V: torch.Tensor, K: int) -> float:
        """V: (D, K) columns. Measures deviation from simplex ETF."""
        G = V.T @ V
        G += torch.ones(K, K, device=device) / (K - 1)
        G -= torch.diag(torch.diag(G))
        return torch.norm(G, p=1).item() / (K * (K - 1))

    cos_M_val = coherence(M_ / (M_norms + 1e-12), num_classes)
    cos_W_val = coherence(W.T / (W_norms.unsqueeze(0) + 1e-12), num_classes)

    # NC1: Tr{Σ_W @ Σ_B†} / C  (Moore-Penrose pseudoinverse)
    Sw_np = Sw.cpu().float().numpy()
    Sb_np = Sb.cpu().float().numpy()
    try:
        k = min(num_classes - 1, D - 1)
        eigvec, eigval, _ = svds(Sb_np.astype(np.float64), k=k)
        inv_Sb = eigvec @ np.diag(eigval ** (-1)) @ eigvec.T
        nc1 = np.trace(Sw_np @ inv_Sb) / num_classes
    except Exception:
        nc1 = float("nan")

    # NC3: ||W^T - M_||^2 (Frobenius, normalized)
    normalized_M = M_ / (torch.norm(M_, "fro") + 1e-12)
    normalized_W = W.T / (torch.norm(W.T, "fro") + 1e-12)
    W_M_dist = (torch.norm(normalized_W - normalized_M) ** 2).item()

    # NC4: NCC mismatch
    ncc_mismatch = 1.0 - NCC_match_net / total_N

    return {
        "accuracy": net_correct / total_N,
        "loss": loss_sum / total_N,
        "Sw_invSb": float(nc1),
        "norm_M_CoV": norm_M_CoV,
        "norm_W_CoV": norm_W_CoV,
        "cos_M": cos_M_val,
        "cos_W": cos_W_val,
        "W_M_dist": W_M_dist,
        "NCC_mismatch": ncc_mismatch,
    }


# ==========================================================================
# Main analysis entry point
# ==========================================================================

def load_checkpoints_and_analyze(
    checkpoint_dir: str,
    model_class: Type[nn.Module],
    loader: torch.utils.data.DataLoader,
    device: str,
    num_classes: int,
    checkpoint_pattern: str = "resnet18_cifar100_*.pth",
    epoch_regex: str = r"epoch(\d+)",
    verbose: bool = True,
) -> NCMetricsTracker:
    """Load every checkpoint matching *pattern*, compute NC metrics, return tracker.

    Parameters
    ----------
    checkpoint_dir : str
        Directory containing ``.pth`` checkpoint files.
    model_class : type
        Model constructor — called as ``model_class(num_classes=num_classes)``.
    loader : DataLoader
        Training (or validation) data loader used for feature extraction.
    device : str
        ``"cuda"`` or ``"cpu"``.
    num_classes : int
        Number of classes (e.g. 100 for CIFAR-100).
    checkpoint_pattern : str
        Glob pattern for checkpoint files.
    epoch_regex : str
        Regex with one capture group to extract the epoch number from filenames.
    verbose : bool
        If True, print progress.

    Returns
    -------
    NCMetricsTracker
        Populated tracker with per-epoch NC metrics.
    """
    # Discover & sort checkpoints
    paths = glob.glob(os.path.join(checkpoint_dir, checkpoint_pattern))
    if not paths:
        raise FileNotFoundError(
            f"No checkpoints matching '{checkpoint_pattern}' in {checkpoint_dir}"
        )

    def _extract_epoch(p: str) -> int:
        m = re.search(epoch_regex, os.path.basename(p))
        return int(m.group(1)) if m else 0

    paths = sorted(paths, key=_extract_epoch)

    # Build model once, reuse across checkpoints
    model = model_class(num_classes=num_classes).to(device)
    classifier = _find_classifier(model)
    hook = _FeatureHook()
    classifier.register_forward_hook(hook)

    tracker = NCMetricsTracker()

    for ckpt_path in (tqdm(paths, desc="Analyzing checkpoints") if verbose else paths):
        epoch = _extract_epoch(ckpt_path)
        _load_checkpoint(ckpt_path, model, device)

        metrics = _compute_nc_metrics(
            model, classifier, hook, loader, device, num_classes
        )

        tracker.epochs.append(epoch)
        for key in (
            "accuracy", "loss", "Sw_invSb",
            "norm_M_CoV", "norm_W_CoV", "cos_M", "cos_W",
            "W_M_dist", "NCC_mismatch",
        ):
            getattr(tracker, key).append(metrics[key])

        if verbose:
            print(
                f"  Epoch {epoch:>4d} | "
                f"Acc {metrics['accuracy']:.4f} | "
                f"NC1 {metrics['Sw_invSb']:.4f} | "
                f"NC3 {metrics['W_M_dist']:.4f} | "
                f"NC4 {metrics['NCC_mismatch']:.4f}"
            )

    return tracker


# ==========================================================================
# Plotting
# ==========================================================================

def plot_nc_evolution(
    tracker: NCMetricsTracker,
    save_dir: Optional[str] = None,
) -> plt.Figure:
    """Plot all NC metrics in a single 2×3 figure. Returns the Figure."""
    epochs = tracker.epochs

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Neural Collapse Evolution During Training", fontsize=16, y=1.02)

    # (0,0) NC1 — Activation Collapse
    ax = axes[0, 0]
    ax.semilogy(epochs, tracker.Sw_invSb, "b-o", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"$\mathrm{Tr}(\Sigma_W \Sigma_B^{\dagger}) / C$")
    ax.set_title("NC1: Activation Collapse")
    ax.grid(True, alpha=0.3)

    # (0,1) NC2 — Equinorm
    ax = axes[0, 1]
    ax.plot(epochs, tracker.norm_M_CoV, "m-o", markersize=3, label="Class Means")
    ax.plot(epochs, tracker.norm_W_CoV, "g-s", markersize=3, label="Classifiers")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Std / Mean of Norms")
    ax.set_title("NC2: Equinorm")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (0,2) NC2 — Equiangularity
    ax = axes[0, 2]
    ax.plot(epochs, tracker.cos_M, "g-o", markersize=3, label="Class Means")
    ax.plot(epochs, tracker.cos_W, "c-s", markersize=3, label="Classifiers")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"Avg $|\cos\theta + 1/(C-1)|$")
    ax.set_title("NC2: Maximal Equiangularity")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (1,0) NC3 — Self-Duality
    ax = axes[1, 0]
    ax.plot(epochs, tracker.W_M_dist, "c-o", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"$\|W^T - \bar{M}\|^2$ (normalized)")
    ax.set_title("NC3: Self-Duality (W ≈ M)")
    ax.grid(True, alpha=0.3)

    # (1,1) NC4 — NCC
    ax = axes[1, 1]
    ax.plot(epochs, tracker.NCC_mismatch, "r-o", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Proportion Mismatch")
    ax.set_title("NC4: Convergence to NCC")
    ax.grid(True, alpha=0.3)

    # (1,2) Training accuracy
    ax = axes[1, 2]
    ax.plot(epochs, [a * 100 for a in tracker.accuracy], "k-o", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Training Accuracy")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "nc_evolution_all.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f" Saved: {path}")

    return fig


def plot_nc_individual(
    tracker: NCMetricsTracker,
    save_dir: Optional[str] = None,
) -> dict:
    """Generate one figure per NC metric. Returns dict[name → Figure]."""
    epochs = tracker.epochs
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    specs = [
        ("NC1_activation_collapse", "NC1: Activation Collapse",
         r"$\mathrm{Tr}(\Sigma_W \Sigma_B^{\dagger}) / C$",
         [(tracker.Sw_invSb, "b-o", r"$\mathrm{Tr}(\Sigma_W \Sigma_B^{\dagger})/C$")], True),
        ("NC2_equinorm", "NC2: Equinorm", "Std / Mean of Norms",
         [(tracker.norm_M_CoV, "m-o", "Class Means"),
          (tracker.norm_W_CoV, "g-s", "Classifiers")], False),
        ("NC2_equiangularity", "NC2: Maximal Equiangularity",
         r"Avg $|\cos\theta + 1/(C-1)|$",
         [(tracker.cos_M, "g-o", "Class Means"),
          (tracker.cos_W, "c-s", "Classifiers")], False),
        ("NC3_self_duality", "NC3: Self-Duality (W ≈ M)",
         r"$\|W^T - \bar{M}\|^2$",
         [(tracker.W_M_dist, "c-o", "W-M distance")], False),
        ("NC4_NCC_convergence", "NC4: Convergence to NCC",
         "Proportion Mismatch",
         [(tracker.NCC_mismatch, "r-o", "NCC mismatch")], False),
    ]

    figs = {}
    for name, title, ylabel, series, use_log in specs:
        fig, ax = plt.subplots(figsize=(8, 5))
        for data, style, label in series:
            if use_log:
                ax.semilogy(epochs, data, style, markersize=3, label=label)
            else:
                ax.plot(epochs, data, style, markersize=3, label=label)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        if save_dir:
            path = os.path.join(save_dir, f"{name}.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")

        figs[name] = fig

    if save_dir:
        print(f" Saved {len(figs)} individual figures to {save_dir}")

    return figs


# ==========================================================================
# Serialization
# ==========================================================================

def save_metrics_yaml(tracker: NCMetricsTracker, path: str) -> None:
    """Save the tracker metrics to a YAML file."""
    data = tracker.to_dict()

    # Convert NaN → None for clean YAML output
    for key, values in data.items():
        if isinstance(values, list):
            data[key] = [
                v if not (isinstance(v, float) and np.isnan(v)) else None
                for v in values
            ]

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    print(f" Metrics saved to: {path}")

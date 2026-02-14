"""
Neural Collapse OOD Detection Module (NECO)
=============================================
Implements NC5 (ID/OOD Orthogonality) and the NECO score from:

    Ben Ammar et al., "NECO: Neural Collapse Based Out-of-Distribution Detection"
    ICLR 2024.  (arXiv:2310.06823)

Usage (from notebook)::

    from src.neural_collapse.nc_ood import (
        load_checkpoints_and_analyze_ood,
        compute_neco_scores,
        evaluate_ood_detection,
        plot_nc5_convergence,
        plot_neco_distributions,
        plot_ood_summary,
        save_ood_metrics_yaml,
        NCOODTracker,
    )

    # 1. Track NC5 across epochs (+ NC1/NC2 with OOD as extra class)
    ood_tracker = load_checkpoints_and_analyze_ood(
        checkpoint_dir="checkpoints/",
        model_class=ResNet18,
        id_loader=train_loader,
        ood_loader=ood_loader,
        device="cuda",
        num_classes=100,
    )

    # 2. Compute NECO score on final model for OOD detection
    results = compute_neco_scores(
        model=model,
        id_loader=test_loader,
        ood_loader=ood_loader,
        device="cuda",
        num_classes=100,
        id_train_loader=train_loader,  # for PCA estimation
    )

    # 3. Evaluate
    metrics = evaluate_ood_detection(results["id_scores"], results["ood_scores"])
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from scipy.sparse.linalg import svds
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

# Reuse helpers from nc_analysis
from .nc_analysis import _FeatureHook, _find_classifier, _load_checkpoint


# ==========================================================================
# Data containers
# ==========================================================================

@dataclass
class NCOODTracker:
    """Tracks NC metrics in the presence of OOD data across epochs.

    Attributes
    ----------
    epochs : list of int
        Epoch numbers analysed.
    nc5_orthodev : list of float
        NC5 ID/OOD orthogonality deviation (eq. 5 of NECO paper).
        Lower → more orthogonal → better for NECO-based OOD detection.
    Sw_invSb_id_only : list of float
        NC1 computed on ID data only.
    Sw_invSb_id_ood : list of float
        NC1 computed treating OOD as an extra class.
    nc2_equiang_id_only : list of float
        NC2 equiangularity on ID only.
    nc2_equiang_id_ood : list of float
        NC2 equiangularity treating OOD as extra class.
    nc2_equinorm_id : list of float
        NC2 equinorm (class means CoV) on ID only.
    """

    epochs: List[int] = field(default_factory=list)

    # NC5
    nc5_orthodev: List[float] = field(default_factory=list)

    # NC1
    Sw_invSb_id_only: List[float] = field(default_factory=list)
    Sw_invSb_id_ood: List[float] = field(default_factory=list)

    # NC2 equiangularity
    nc2_equiang_id_only: List[float] = field(default_factory=list)
    nc2_equiang_id_ood: List[float] = field(default_factory=list)

    # NC2 equinorm (ID only)
    nc2_equinorm_id: List[float] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "NC OOD METRICS SUMMARY (NECO)",
            "=" * 60,
            f"Epochs analyzed: {len(self.epochs)}",
        ]
        if self.epochs:
            lines.append(f"Epoch range: {self.epochs[0]} → {self.epochs[-1]}")
        if self.nc5_orthodev:
            lines.append(f"Final NC5 OrthoDev: {self.nc5_orthodev[-1]:.6f}")
        if self.Sw_invSb_id_only:
            lines.append(
                f"Final NC1 (ID only): {self.Sw_invSb_id_only[-1]:.4f}"
            )
        if self.Sw_invSb_id_ood:
            lines.append(
                f"Final NC1 (ID+OOD): {self.Sw_invSb_id_ood[-1]:.4f}"
            )
        if self.nc2_equiang_id_only:
            lines.append(
                f"Final NC2 equiangularity (ID): {self.nc2_equiang_id_only[-1]:.4f}"
            )
        if self.nc2_equiang_id_ood:
            lines.append(
                f"Final NC2 equiangularity (ID+OOD): {self.nc2_equiang_id_ood[-1]:.4f}"
            )
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "epochs": list(self.epochs),
            "nc5_orthodev": list(self.nc5_orthodev),
            "Sw_invSb_id_only": list(self.Sw_invSb_id_only),
            "Sw_invSb_id_ood": list(self.Sw_invSb_id_ood),
            "nc2_equiang_id_only": list(self.nc2_equiang_id_only),
            "nc2_equiang_id_ood": list(self.nc2_equiang_id_ood),
            "nc2_equinorm_id": list(self.nc2_equinorm_id),
        }


@dataclass
class NECOResult:
    """Holds the output of a NECO evaluation on a single checkpoint.

    Attributes
    ----------
    id_scores : np.ndarray
        NECO scores for ID test samples (higher → more ID-like).
    ood_scores : np.ndarray
        NECO scores for OOD samples.
    auroc : float
        Area under ROC (ID=positive).
    fpr95 : float
        False Positive Rate at 95% True Positive Rate.
    pca_dim : int
        Number of principal components used.
    """

    id_scores: np.ndarray
    ood_scores: np.ndarray
    auroc: float
    fpr95: float
    pca_dim: int


# ==========================================================================
# Core: extract features from a loader
# ==========================================================================

@torch.no_grad()
def _extract_features(
    model: nn.Module,
    hook: _FeatureHook,
    loader: torch.utils.data.DataLoader,
    device: str,
    max_samples: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward-pass all data through *model* and collect features.

    Returns
    -------
    features : (N, D)  penultimate layer activations
    labels   : (N,)    class labels
    logits   : (N, C)  raw network outputs
    """
    model.eval()
    all_h, all_y, all_logits = [], [], []
    n = 0

    for images, targets in loader:
        images = images.to(device)
        outputs = model(images)
        h = hook.features.view(images.shape[0], -1)

        all_h.append(h.cpu())
        all_y.append(targets)
        all_logits.append(outputs.cpu())

        n += images.shape[0]
        if max_samples is not None and n >= max_samples:
            break

    features = torch.cat(all_h, dim=0)
    labels = torch.cat(all_y, dim=0)
    logits = torch.cat(all_logits, dim=0)

    if max_samples is not None:
        features = features[:max_samples]
        labels = labels[:max_samples]
        logits = logits[:max_samples]

    return features, labels, logits


# ==========================================================================
# NC5: ID/OOD Orthogonality (eq. 5)
# ==========================================================================

def _compute_nc5(
    id_means: torch.Tensor,
    ood_global_mean: torch.Tensor,
) -> float:
    """Compute NC5 OrthoDev (eq. 5 from NECO paper).

    Parameters
    ----------
    id_means : (C, D)
        Per-class means of ID data (NOT centered by global mean).
    ood_global_mean : (D,)
        Global mean of OOD features.

    Returns
    -------
    float
        Average absolute cosine similarity between each ID class mean
        and the OOD global mean.  Should → 0 if NC5 is satisfied.
    """
    # eq.5: Avg_c |<µ_c, µ^OOD_G> / (||µ_c|| * ||µ^OOD_G||)|
    mu_ood_norm = torch.norm(ood_global_mean)
    if mu_ood_norm < 1e-12:
        return 0.0

    cosines = []
    for c in range(id_means.shape[0]):
        mu_c = id_means[c]
        mu_c_norm = torch.norm(mu_c)
        if mu_c_norm < 1e-12:
            continue
        cos = torch.abs(torch.dot(mu_c, ood_global_mean) / (mu_c_norm * mu_ood_norm))
        cosines.append(cos.item())

    return float(np.mean(cosines)) if cosines else 0.0


# ==========================================================================
# NC metrics with OOD as extra class
# ==========================================================================

def _coherence(V: torch.Tensor, K: int, device: str) -> float:
    """Equiangularity measure: deviation from simplex ETF.

    V : (D, K) column-normalized vectors.
    """
    G = V.T @ V
    G += torch.ones(K, K, device=device) / (K - 1)
    G -= torch.diag(torch.diag(G))
    return torch.norm(G, p=1).item() / (K * (K - 1))


@torch.no_grad()
def _compute_nc_ood_metrics(
    model: nn.Module,
    hook: _FeatureHook,
    id_loader: torch.utils.data.DataLoader,
    ood_loader: torch.utils.data.DataLoader,
    device: str,
    num_classes: int,
) -> dict:
    """Compute NC1, NC2, NC5 metrics with OOD data treated as an extra class.

    This implements the analysis from Section 4 and Appendix D of the NECO paper:
    - NC1 on ID only and on ID+OOD (OOD = class C)
    - NC2 equiangularity on ID only and on ID+OOD
    - NC2 equinorm on ID class means
    - NC5 OrthoDev (eq. 5)
    """
    model.eval()
    C = num_classes
    C_total = C + 1  # OOD as extra class

    # --- Pass 1: class means ---
    N_id = [0] * C
    mean_id = [None] * C

    # ID features
    for images, targets in id_loader:
        images, targets = images.to(device), targets.to(device)
        _ = model(images)
        h = hook.features.view(images.shape[0], -1)

        for c in range(C):
            idxs = (targets == c).nonzero(as_tuple=True)[0]
            if len(idxs) == 0:
                continue
            h_c = h[idxs]
            if mean_id[c] is None:
                mean_id[c] = torch.zeros(h.shape[1], device=device)
            mean_id[c] += h_c.sum(dim=0)
            N_id[c] += len(idxs)

    D = None
    for c in range(C):
        if N_id[c] > 0 and mean_id[c] is not None:
            mean_id[c] /= N_id[c]
            D = mean_id[c].shape[0]
        else:
            mean_id[c] = torch.zeros_like(
                mean_id[0] if mean_id[0] is not None else torch.zeros(1)
            )

    if D is None:
        raise RuntimeError("No ID features extracted. Check loader and model.")

    # OOD features (single "class")
    ood_sum = torch.zeros(D, device=device)
    N_ood = 0
    for images, targets in ood_loader:
        images = images.to(device)
        _ = model(images)
        h = hook.features.view(images.shape[0], -1)
        ood_sum += h.sum(dim=0)
        N_ood += images.shape[0]

    mean_ood = ood_sum / max(N_ood, 1)

    # Stack all means:  M_id = (C, D),  M_all = (C+1, D)
    M_id = torch.stack(mean_id)            # (C, D)
    M_all = torch.cat([M_id, mean_ood.unsqueeze(0)], dim=0)  # (C+1, D)

    # --- NC5: OrthoDev (eq. 5) ---
    nc5 = _compute_nc5(M_id, mean_ood)

    # --- Centered class means ---
    # ID only
    muG_id = M_id.mean(dim=0, keepdim=True)
    M_id_centered = (M_id - muG_id).T  # (D, C)

    # ID + OOD
    muG_all = M_all[:C].mean(dim=0, keepdim=True)  # global mean from ID only
    M_all_centered = (M_all - muG_all).T  # (D, C+1)

    # --- NC2: equinorm (ID only) ---
    M_norms_id = torch.norm(M_id_centered, dim=0)
    nc2_equinorm_id = (torch.std(M_norms_id) / (torch.mean(M_norms_id) + 1e-12)).item()

    # --- NC2: equiangularity ---
    # ID only
    M_id_normed = M_id_centered / (M_norms_id.unsqueeze(0) + 1e-12)
    nc2_equiang_id = _coherence(M_id_normed, C, device)

    # ID + OOD
    M_norms_all = torch.norm(M_all_centered, dim=0)
    M_all_normed = M_all_centered / (M_norms_all.unsqueeze(0) + 1e-12)
    nc2_equiang_id_ood = _coherence(M_all_normed, C_total, device)

    # --- Pass 2: within-class covariance ---
    Sw_id = torch.zeros(D, D, device=device)
    Sw_all = torch.zeros(D, D, device=device)

    # ID within-class cov
    for images, targets in id_loader:
        images, targets = images.to(device), targets.to(device)
        _ = model(images)
        h = hook.features.view(images.shape[0], -1)

        for c in range(C):
            idxs = (targets == c).nonzero(as_tuple=True)[0]
            if len(idxs) == 0:
                continue
            z = h[idxs] - mean_id[c].unsqueeze(0)
            Sw_id += z.T @ z

    # OOD within-class cov (single class)
    Sw_ood = torch.zeros(D, D, device=device)
    for images, targets in ood_loader:
        images = images.to(device)
        _ = model(images)
        h = hook.features.view(images.shape[0], -1)
        z = h - mean_ood.unsqueeze(0)
        Sw_ood += z.T @ z

    total_N_id = sum(N_id)
    total_N_all = total_N_id + N_ood

    Sw_id_avg = Sw_id / max(total_N_id, 1)
    Sw_all_avg = (Sw_id + Sw_ood) / max(total_N_all, 1)

    # --- NC1: Tr{Σ_W Σ_B†} / C ---
    def _nc1(Sw_np, Sb_np, n_classes):
        try:
            k = min(n_classes - 1, D - 1)
            eigvec, eigval, _ = svds(Sb_np.astype(np.float64), k=k)
            inv_Sb = eigvec @ np.diag(eigval ** (-1)) @ eigvec.T
            return float(np.trace(Sw_np @ inv_Sb) / n_classes)
        except Exception:
            return float("nan")

    Sb_id_np = (M_id_centered @ M_id_centered.T / C).cpu().float().numpy()
    Sb_all_np = (M_all_centered @ M_all_centered.T / C_total).cpu().float().numpy()

    nc1_id = _nc1(Sw_id_avg.cpu().float().numpy(), Sb_id_np, C)
    nc1_all = _nc1(Sw_all_avg.cpu().float().numpy(), Sb_all_np, C_total)

    return {
        "nc5_orthodev": nc5,
        "Sw_invSb_id_only": nc1_id,
        "Sw_invSb_id_ood": nc1_all,
        "nc2_equiang_id_only": nc2_equiang_id,
        "nc2_equiang_id_ood": nc2_equiang_id_ood,
        "nc2_equinorm_id": nc2_equinorm_id,
    }


# ==========================================================================
# NECO Score (eq. 6)
# ==========================================================================

@torch.no_grad()
def compute_neco_scores(
    model: nn.Module,
    id_loader: torch.utils.data.DataLoader,
    ood_loader: torch.utils.data.DataLoader,
    device: str,
    num_classes: int,
    id_train_loader: Optional[torch.utils.data.DataLoader] = None,
    pca_dim: Optional[int] = None,
    use_maxlogit: bool = False,
    max_train_samples: int = 50_000,
) -> NECOResult:
    """Compute NECO scores for ID and OOD data.

    Implements eq. 6:  ``NECO(x) = ||P h(x)|| / ||h(x)||``

    Parameters
    ----------
    model : nn.Module
        Trained model.
    id_loader : DataLoader
        ID *test* data for evaluation.
    ood_loader : DataLoader
        OOD data for evaluation.
    device : str
        ``"cuda"`` or ``"cpu"``.
    num_classes : int
        Number of ID classes.
    id_train_loader : DataLoader, optional
        ID *training* data for PCA estimation.  If None, uses ``id_loader``.
    pca_dim : int, optional
        Number of principal components.  If None, defaults to ``num_classes - 1``
        (= dimension of the Simplex ETF subspace).
    use_maxlogit : bool
        If True, multiply NECO score by max logit (recommended for ViT).
    max_train_samples : int
        Maximum samples for PCA fitting (to limit memory usage).

    Returns
    -------
    NECOResult
        Contains ``id_scores``, ``ood_scores``, ``auroc``, ``fpr95``, ``pca_dim``.
    """
    model.to(device).eval()
    classifier = _find_classifier(model)
    hook = _FeatureHook()
    handle = classifier.register_forward_hook(hook)

    try:
        # --- Step 1: Fit PCA on ID training features ---
        train_loader = id_train_loader if id_train_loader is not None else id_loader

        train_features, _, _ = _extract_features(
            model, hook, train_loader, device, max_samples=max_train_samples
        )

        d = pca_dim if pca_dim is not None else (num_classes - 1)
        d = min(d, train_features.shape[1] - 1, train_features.shape[0] - 1)

        pca = PCA(n_components=d)
        pca.fit(train_features.numpy())

        # Projection matrix P: (d, D)
        P = torch.tensor(pca.components_, dtype=torch.float32)

        # --- Step 2: Score ID test samples ---
        id_features, _, id_logits = _extract_features(
            model, hook, id_loader, device
        )
        id_scores = _neco_score(id_features, P, id_logits if use_maxlogit else None)

        # --- Step 3: Score OOD samples ---
        ood_features, _, ood_logits = _extract_features(
            model, hook, ood_loader, device
        )
        ood_scores = _neco_score(ood_features, P, ood_logits if use_maxlogit else None)

    finally:
        handle.remove()

    # --- Step 4: Evaluate ---
    metrics = evaluate_ood_detection(id_scores, ood_scores)

    return NECOResult(
        id_scores=id_scores,
        ood_scores=ood_scores,
        auroc=metrics["auroc"],
        fpr95=metrics["fpr95"],
        pca_dim=d,
    )


def _neco_score(
    features: torch.Tensor,
    P: torch.Tensor,
    logits: Optional[torch.Tensor] = None,
) -> np.ndarray:
    """Compute NECO score: ||P h|| / ||h||, optionally × MaxLogit.

    Parameters
    ----------
    features : (N, D)
    P : (d, D)  PCA projection matrix
    logits : (N, C), optional.  If given, multiply score by max logit.

    Returns
    -------
    np.ndarray of shape (N,)
    """
    # ||P h||  for each sample
    projected = features @ P.T  # (N, d)
    proj_norms = torch.norm(projected, dim=1)  # (N,)

    # ||h||
    full_norms = torch.norm(features, dim=1)  # (N,)

    scores = proj_norms / (full_norms + 1e-12)  # (N,)

    if logits is not None:
        max_logits = logits.max(dim=1).values  # (N,)
        scores = scores * max_logits

    return scores.numpy()


# ==========================================================================
# OOD evaluation metrics
# ==========================================================================

def evaluate_ood_detection(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
) -> Dict[str, float]:
    """Compute AUROC and FPR95 for OOD detection.

    Convention: ID = positive (higher NECO score).

    Parameters
    ----------
    id_scores : (N_id,)  scores for ID test samples
    ood_scores : (N_ood,) scores for OOD samples

    Returns
    -------
    dict with keys ``"auroc"`` and ``"fpr95"`` (both in [0, 1]).
    """
    labels = np.concatenate([
        np.ones(len(id_scores)),
        np.zeros(len(ood_scores)),
    ])
    scores = np.concatenate([id_scores, ood_scores])

    auroc = roc_auc_score(labels, scores)

    # FPR at 95% TPR
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.searchsorted(tpr, 0.95, side="left")
    fpr95 = fpr[min(idx, len(fpr) - 1)]

    return {"auroc": float(auroc), "fpr95": float(fpr95)}


# ==========================================================================
# Main entry point: analyze across checkpoints
# ==========================================================================

def load_checkpoints_and_analyze_ood(
    checkpoint_dir: str,
    model_class: Type[nn.Module],
    id_loader: torch.utils.data.DataLoader,
    ood_loader: torch.utils.data.DataLoader,
    device: str,
    num_classes: int,
    checkpoint_pattern: str = "resnet18_cifar100_*.pth",
    epoch_regex: str = r"epoch(\d+)",
    verbose: bool = True,
) -> NCOODTracker:
    """Analyze NC metrics (incl. NC5) across training checkpoints.

    Parameters
    ----------
    checkpoint_dir : str
        Directory containing ``.pth`` checkpoint files.
    model_class : type
        Model constructor — called as ``model_class(num_classes=num_classes)``.
    id_loader : DataLoader
        In-distribution data loader (training set).
    ood_loader : DataLoader
        Out-of-distribution data loader.
    device : str
        ``"cuda"`` or ``"cpu"``.
    num_classes : int
        Number of ID classes (e.g. 100 for CIFAR-100).
    checkpoint_pattern, epoch_regex : str
        Used to discover and sort checkpoint files.
    verbose : bool
        Print per-epoch progress.

    Returns
    -------
    NCOODTracker
        Populated tracker with NC5 and NC1/NC2 (ID-only & ID+OOD) per epoch.
    """
    paths = glob.glob(os.path.join(checkpoint_dir, checkpoint_pattern))
    if not paths:
        raise FileNotFoundError(
            f"No checkpoints matching '{checkpoint_pattern}' in {checkpoint_dir}"
        )

    def _extract_epoch(p: str) -> int:
        m = re.search(epoch_regex, os.path.basename(p))
        return int(m.group(1)) if m else 0

    paths = sorted(paths, key=_extract_epoch)

    model = model_class(num_classes=num_classes).to(device)
    classifier = _find_classifier(model)
    hook = _FeatureHook()
    classifier.register_forward_hook(hook)

    tracker = NCOODTracker()

    iterator = tqdm(paths, desc="NC+OOD Analysis") if verbose else paths
    for ckpt_path in iterator:
        epoch = _extract_epoch(ckpt_path)
        _load_checkpoint(ckpt_path, model, device)

        metrics = _compute_nc_ood_metrics(
            model, hook, id_loader, ood_loader, device, num_classes
        )

        tracker.epochs.append(epoch)
        for key in (
            "nc5_orthodev",
            "Sw_invSb_id_only", "Sw_invSb_id_ood",
            "nc2_equiang_id_only", "nc2_equiang_id_ood",
            "nc2_equinorm_id",
        ):
            getattr(tracker, key).append(metrics[key])

        if verbose:
            print(
                f"  Epoch {epoch:>4d} | "
                f"NC5 {metrics['nc5_orthodev']:.6f} | "
                f"NC1(ID) {metrics['Sw_invSb_id_only']:.4f} | "
                f"NC1(+OOD) {metrics['Sw_invSb_id_ood']:.4f} | "
                f"NC2eq(ID) {metrics['nc2_equiang_id_only']:.4f} | "
                f"NC2eq(+OOD) {metrics['nc2_equiang_id_ood']:.4f}"
            )

    return tracker


# ==========================================================================
# Plotting
# ==========================================================================

def plot_nc5_convergence(
    tracker: NCOODTracker,
    ood_name: str = "OOD",
    id_name: str = "ID",
    save_dir: Optional[str] = None,
) -> plt.Figure:
    """Plot NC5 OrthoDev convergence (cf. Fig 1 / D.11 of NECO paper).

    Returns the figure object.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        tracker.epochs, tracker.nc5_orthodev,
        "b-o", markersize=4, linewidth=2,
        label=f"{id_name} / {ood_name}",
    )
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("NC5: ID/OOD OrthoDev", fontsize=12)
    ax.set_title("NC5: Convergence to ID/OOD Orthogonality", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "nc5_convergence.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"💾 Saved: {path}")

    return fig


def plot_ood_summary(
    tracker: NCOODTracker,
    ood_name: str = "OOD",
    save_dir: Optional[str] = None,
) -> plt.Figure:
    """Plot 2×3 grid: NC5, NC1 (ID vs ID+OOD), NC2 equiang, NC2 equinorm.

    Mirrors the style of ``plot_nc_evolution`` from ``nc_analysis.py``
    but focused on the ID/OOD interaction.
    """
    epochs = tracker.epochs
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"Neural Collapse with OOD Data ({ood_name})", fontsize=16, y=1.02
    )

    # (0,0) NC5 OrthoDev
    ax = axes[0, 0]
    ax.plot(epochs, tracker.nc5_orthodev, "b-o", markersize=3)
    ax.set_ylabel("NC5 OrthoDev")
    ax.set_title("NC5: ID/OOD Orthogonality")
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)

    # (0,1) NC1 ID only
    ax = axes[0, 1]
    ax.semilogy(epochs, tracker.Sw_invSb_id_only, "b-o", markersize=3, label="ID only")
    ax.semilogy(epochs, tracker.Sw_invSb_id_ood, "r-s", markersize=3, label="ID+OOD")
    ax.set_ylabel(r"$\mathrm{Tr}(\Sigma_W \Sigma_B^{\dagger}) / C$")
    ax.set_title("NC1: Activation Collapse")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (0,2) NC2 equiangularity
    ax = axes[0, 2]
    ax.plot(epochs, tracker.nc2_equiang_id_only, "g-o", markersize=3, label="ID only")
    ax.plot(epochs, tracker.nc2_equiang_id_ood, "m-s", markersize=3, label="ID+OOD")
    ax.set_ylabel(r"Avg $|\cos\theta + 1/(C-1)|$")
    ax.set_title("NC2: Equiangularity")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (1,0) NC2 equinorm (ID only)
    ax = axes[1, 0]
    ax.plot(epochs, tracker.nc2_equinorm_id, "m-o", markersize=3)
    ax.set_ylabel("Std / Mean of Norms")
    ax.set_title("NC2: Equinorm (ID)")
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)

    # (1,1) empty — information box
    ax = axes[1, 1]
    ax.axis("off")
    info_lines = [
        f"Final NC5 OrthoDev: {tracker.nc5_orthodev[-1]:.6f}",
        f"Final NC1 (ID):     {tracker.Sw_invSb_id_only[-1]:.4f}",
        f"Final NC1 (ID+OOD): {tracker.Sw_invSb_id_ood[-1]:.4f}",
        f"Final NC2 eq (ID):  {tracker.nc2_equiang_id_only[-1]:.4f}",
        f"Final NC2 eq (+OOD):{tracker.nc2_equiang_id_ood[-1]:.4f}",
    ]
    ax.text(
        0.1, 0.5, "\n".join(info_lines),
        transform=ax.transAxes, fontsize=12, verticalalignment="center",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )
    ax.set_title("Summary")

    # (1,2) reserved
    axes[1, 2].axis("off")

    fig.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "nc_ood_summary.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"💾 Saved: {path}")

    return fig


def plot_neco_distributions(
    result: NECOResult,
    id_name: str = "ID",
    ood_name: str = "OOD",
    save_dir: Optional[str] = None,
) -> plt.Figure:
    """Plot NECO score histograms for ID vs OOD (cf. Fig E.16/E.17 in paper).

    Parameters
    ----------
    result : NECOResult
        Output of ``compute_neco_scores()``.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax = axes[0]
    bins = np.linspace(
        min(result.ood_scores.min(), result.id_scores.min()),
        max(result.ood_scores.max(), result.id_scores.max()),
        80,
    )
    ax.hist(result.id_scores, bins=bins, alpha=0.6, density=True,
            color="teal", label=id_name)
    ax.hist(result.ood_scores, bins=bins, alpha=0.6, density=True,
            color="gray", label=ood_name)
    ax.set_xlabel("NECO score")
    ax.set_ylabel("Density")
    ax.set_title(f"NECO Score Distribution (d={result.pca_dim})")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # ROC Curve
    ax = axes[1]
    labels = np.concatenate([
        np.ones(len(result.id_scores)),
        np.zeros(len(result.ood_scores)),
    ])
    scores = np.concatenate([result.id_scores, result.ood_scores])
    fpr, tpr, _ = roc_curve(labels, scores)
    ax.plot(fpr, tpr, "b-", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(
        f"ROC Curve  (AUROC={result.auroc:.4f}, FPR95={result.fpr95:.4f})"
    )
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "neco_distributions.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"💾 Saved: {path}")

    return fig


def plot_neco_pca_2d(
    model: nn.Module,
    id_loader: torch.utils.data.DataLoader,
    ood_loader: torch.utils.data.DataLoader,
    device: str,
    num_classes: int,
    id_name: str = "ID",
    ood_name: str = "OOD",
    max_samples: int = 2000,
    save_dir: Optional[str] = None,
) -> plt.Figure:
    """PCA projection (2D) of ID + OOD features (cf. Fig 2 / D.14 in paper).

    Shows how ID clusters separate from OOD data near the origin.
    """
    model.to(device).eval()
    classifier = _find_classifier(model)
    hook = _FeatureHook()
    handle = classifier.register_forward_hook(hook)

    try:
        id_feats, id_labels, _ = _extract_features(
            model, hook, id_loader, device, max_samples=max_samples
        )
        ood_feats, _, _ = _extract_features(
            model, hook, ood_loader, device, max_samples=max_samples
        )
    finally:
        handle.remove()

    # Fit PCA on ID features
    pca = PCA(n_components=2)
    id_proj = pca.fit_transform(id_feats.numpy())
    ood_proj = pca.transform(ood_feats.numpy())

    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot ID per class (use colormap)
    unique_labels = np.unique(id_labels.numpy())
    cmap = plt.cm.get_cmap("tab20", len(unique_labels))
    for i, c in enumerate(unique_labels):
        mask = id_labels.numpy() == c
        ax.scatter(
            id_proj[mask, 0], id_proj[mask, 1],
            c=[cmap(i % 20)], s=8, alpha=0.5,
        )

    # Plot OOD
    ax.scatter(
        ood_proj[:, 0], ood_proj[:, 1],
        c="gray", s=8, alpha=0.3, marker="x",
        label=f"{ood_name} (OOD)",
    )

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(f"PCA — {id_name} (colored) vs {ood_name} (gray)")
    ax.legend(fontsize=11, markerscale=3)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "neco_pca_2d.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"💾 Saved: {path}")

    return fig


# ==========================================================================
# Serialization
# ==========================================================================

def save_ood_metrics_yaml(tracker: NCOODTracker, path: str) -> None:
    """Save OOD tracker metrics to YAML."""
    data = tracker.to_dict()

    for key, values in data.items():
        if isinstance(values, list):
            data[key] = [
                v if not (isinstance(v, float) and np.isnan(v)) else None
                for v in values
            ]

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    print(f"💾 OOD metrics saved to: {path}")

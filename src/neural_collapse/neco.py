"""
NECO: Neural Collapse Based Out-of-Distribution Detection
==========================================================
Implements the NECO score (eq. 6), baseline OOD methods (MSP, MaxLogit, Energy),
evaluation metrics (AUROC, FPR95), and visualisation utilities.

Reference:
    Ben Ammar et al., "NECO: Neural Collapse Based Out-of-Distribution Detection"
    ICLR 2024.  (arXiv:2310.06823)  —  eq. 6, Section 5.

Usage (from notebook)::

    from src.neural_collapse.neco import (
        compute_neco_scores,
        compute_baseline_scores,
        evaluate_ood_detection,
        plot_neco_distributions,
        plot_neco_pca_2d,
        NECOResult,
    )

    result = compute_neco_scores(
        model=model,
        id_loader=test_loader,
        ood_loader=svhn_loader,
        device="cuda",
        num_classes=100,
        id_train_loader=train_loader,
    )
    print(f"AUROC: {result.auroc:.4f}  FPR95: {result.fpr95:.4f}")
    plot_neco_distributions(result, id_name="CIFAR-100", ood_name="SVHN")
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, roc_curve

from .nc_analysis import _FeatureHook, _find_classifier


# ==========================================================================
# Data container
# ==========================================================================

@dataclass
class NECOResult:
    """Holds the output of a NECO evaluation.

    Attributes
    ----------
    id_scores : np.ndarray   (N_id,)   — higher → more ID-like
    ood_scores : np.ndarray  (N_ood,)
    auroc : float             in [0,1]
    fpr95 : float             FPR at 95 % TPR
    pca_dim : int             number of principal components used
    """

    id_scores: np.ndarray
    ood_scores: np.ndarray
    auroc: float
    fpr95: float
    pca_dim: int


# ==========================================================================
# Feature extraction
# ==========================================================================

@torch.no_grad()
def _extract_features(
    model: nn.Module,
    hook: _FeatureHook,
    loader: torch.utils.data.DataLoader,
    device: str,
    max_samples: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward-pass *loader* and collect penultimate features.

    Returns
    -------
    features : (N, D)
    labels   : (N,)
    logits   : (N, C)
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
# NECO Score  (eq. 6)
# ==========================================================================

def _neco_score(
    features: torch.Tensor,
    P: torch.Tensor,
    logits: Optional[torch.Tensor] = None,
) -> np.ndarray:
    """NECO(x) = ||P h(x)|| / ||h(x)||, optionally × MaxLogit.

    Parameters
    ----------
    features : (N, D)
    P : (d, D)   PCA projection matrix
    logits : (N, C), optional — multiply by max logit when given.

    Returns
    -------
    np.ndarray (N,)
    """
    projected = features @ P.T               # (N, d)
    proj_norms = torch.norm(projected, dim=1) # (N,)
    full_norms = torch.norm(features, dim=1)  # (N,)

    scores = proj_norms / (full_norms + 1e-12)

    if logits is not None:
        max_logits = logits.max(dim=1).values
        scores = scores * max_logits

    return scores.numpy()


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
        ID *test* data.
    ood_loader : DataLoader
        OOD data.
    device : str
    num_classes : int
    id_train_loader : DataLoader, optional
        ID *training* data for PCA.  Defaults to *id_loader*.
    pca_dim : int, optional
        Number of principal components.  Default: ``num_classes - 1``.
    use_maxlogit : bool
        Multiply NECO by max logit (recommended for ViT).
    max_train_samples : int
        Cap samples for PCA fitting.

    Returns
    -------
    NECOResult
    """
    model.to(device).eval()
    classifier = _find_classifier(model)
    hook = _FeatureHook()
    handle = classifier.register_forward_hook(hook)

    try:
        # Step 1: fit PCA on ID training features
        train_loader = id_train_loader if id_train_loader is not None else id_loader
        train_features, _, _ = _extract_features(
            model, hook, train_loader, device, max_samples=max_train_samples
        )

        d = pca_dim if pca_dim is not None else (num_classes - 1)
        d = min(d, train_features.shape[1] - 1, train_features.shape[0] - 1)

        pca = PCA(n_components=d)
        pca.fit(train_features.numpy())
        P = torch.tensor(pca.components_, dtype=torch.float32)   # (d, D)

        # Step 2: score ID test
        id_features, _, id_logits = _extract_features(
            model, hook, id_loader, device
        )
        id_scores = _neco_score(
            id_features, P, id_logits if use_maxlogit else None
        )

        # Step 3: score OOD
        ood_features, _, ood_logits = _extract_features(
            model, hook, ood_loader, device
        )
        ood_scores = _neco_score(
            ood_features, P, ood_logits if use_maxlogit else None
        )

    finally:
        handle.remove()

    # Step 4: evaluate
    metrics = evaluate_ood_detection(id_scores, ood_scores)

    return NECOResult(
        id_scores=id_scores,
        ood_scores=ood_scores,
        auroc=metrics["auroc"],
        fpr95=metrics["fpr95"],
        pca_dim=d,
    )


# ==========================================================================
# Baseline OOD scores  (MSP, MaxLogit, Energy)
# ==========================================================================

@torch.no_grad()
def compute_baseline_scores(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: str,
    method: str = "msp",
) -> np.ndarray:
    """Compute a simple baseline OOD score for every sample in *loader*.

    Parameters
    ----------
    method : {"msp", "maxlogit", "energy"}
        * msp       — Maximum Softmax Probability (Hendrycks & Gimpel 2017)
        * maxlogit  — Max Logit (Hendrycks et al. 2022)
        * energy    — Energy = logsumexp(logits) (Liu et al. 2020)

    Returns
    -------
    np.ndarray (N,)   Higher → more ID-like.
    """
    model.eval()
    all_scores = []

    for images, _ in loader:
        images = images.to(device)
        logits = model(images)

        if method == "msp":
            scores = torch.softmax(logits, dim=1).max(dim=1).values
        elif method == "maxlogit":
            scores = logits.max(dim=1).values
        elif method == "energy":
            scores = torch.logsumexp(logits, dim=1)
        else:
            raise ValueError(
                f"Unknown method '{method}'. Choose from: msp, maxlogit, energy."
            )

        all_scores.append(scores.cpu())

    return torch.cat(all_scores).numpy()


# ==========================================================================
# Evaluation metrics
# ==========================================================================

def evaluate_ood_detection(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
) -> Dict[str, float]:
    """Compute AUROC and FPR95 for OOD detection.

    Convention: ID = positive class (higher score).

    Returns
    -------
    dict  ``{"auroc": …, "fpr95": …}``  both in [0, 1].
    """
    labels = np.concatenate([
        np.ones(len(id_scores)),
        np.zeros(len(ood_scores)),
    ])
    scores = np.concatenate([id_scores, ood_scores])

    auroc = roc_auc_score(labels, scores)

    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.searchsorted(tpr, 0.95, side="left")
    fpr95 = fpr[min(idx, len(fpr) - 1)]

    return {"auroc": float(auroc), "fpr95": float(fpr95)}


# ==========================================================================
# Plotting
# ==========================================================================

def plot_neco_distributions(
    result: NECOResult,
    id_name: str = "ID",
    ood_name: str = "OOD",
    save_dir: Optional[str] = None,
) -> plt.Figure:
    """NECO score histograms + ROC curve (cf. Fig E.16/E.17 in paper)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax = axes[0]
    lo = min(result.ood_scores.min(), result.id_scores.min())
    hi = max(result.ood_scores.max(), result.id_scores.max())
    bins = np.linspace(lo, hi, 80)
    ax.hist(result.id_scores, bins=bins, alpha=0.6, density=True,
            color="teal", label=id_name)
    ax.hist(result.ood_scores, bins=bins, alpha=0.6, density=True,
            color="gray", label=ood_name)
    ax.set_xlabel("NECO score")
    ax.set_ylabel("Density")
    ax.set_title(f"NECO Score Distribution (d={result.pca_dim})")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # ROC
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
        path = os.path.join(
            save_dir,
            f"neco_distributions_{ood_name.lower().replace('-', '')}.png",
        )
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
    """PCA 2-D projection of ID + OOD features (cf. Fig 2 / D.14 in paper)."""
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

    pca = PCA(n_components=2)
    id_proj = pca.fit_transform(id_feats.numpy())
    ood_proj = pca.transform(ood_feats.numpy())

    fig, ax = plt.subplots(figsize=(8, 8))

    unique_labels = np.unique(id_labels.numpy())
    cmap = plt.cm.get_cmap("tab20", len(unique_labels))
    for i, c in enumerate(unique_labels):
        mask = id_labels.numpy() == c
        ax.scatter(
            id_proj[mask, 0], id_proj[mask, 1],
            c=[cmap(i % 20)], s=8, alpha=0.5,
        )

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
        path = os.path.join(
            save_dir,
            f"neco_pca_2d_{ood_name.lower().replace('-', '')}.png",
        )
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"💾 Saved: {path}")

    return fig


def plot_pca_dim_sweep(
    results_by_dim: Dict[int, NECOResult],
    ood_name: str = "OOD",
    save_dir: Optional[str] = None,
) -> plt.Figure:
    """Plot AUROC & FPR95 vs PCA dimension (cf. Fig C.5/C.6 in paper).

    Parameters
    ----------
    results_by_dim : dict[int, NECOResult]
        Keyed by PCA dimension.
    """
    dims = sorted(results_by_dim.keys())
    aurocs = [results_by_dim[d].auroc * 100 for d in dims]
    fpr95s = [results_by_dim[d].fpr95 * 100 for d in dims]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(dims, aurocs, "b-o", markersize=4)
    ax.set_xlabel("PCA dimension (d)", fontsize=12)
    ax.set_ylabel("AUROC (%)", fontsize=12)
    ax.set_title(f"AUROC vs PCA Dimension ({ood_name})", fontsize=14)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(dims, fpr95s, "r-s", markersize=4)
    ax.set_xlabel("PCA dimension (d)", fontsize=12)
    ax.set_ylabel("FPR95 (%)", fontsize=12)
    ax.set_title(f"FPR95 vs PCA Dimension ({ood_name})", fontsize=14)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(
            save_dir,
            f"neco_pca_sweep_{ood_name.lower().replace('-', '')}.png",
        )
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"💾 Saved: {path}")

    return fig

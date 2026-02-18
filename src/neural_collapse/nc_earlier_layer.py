"""
Neural Collapse on Earlier Layers (NC1–NC5)
=============================================
Analyses how Neural Collapse emerges progressively through network depth.
At early layers the features live in high-dimensional spatial maps; we apply
Global Average Pooling (GAP) to obtain a feature vector per sample and then
compute the standard NC metrics.

Metrics computed at each layer:
    NC1  — Activation collapse:   Tr(Σ_W Σ_B†) / C
    NC2  — Equinorm:              CoV(‖μ̃_c‖)
    NC2  — Equiangularity:        coherence(M̃)
    NC3  — Self-duality (W ≈ M):  ‖W^T − M̃‖²_F  (penultimate only)
    NC4  — NCC agreement:         1 − Acc(NCC vs network)  (ALL layers)
    NC5  — ID/OOD orthogonality:  Avg_c |cos(μ_c, μ^OOD_G)|  (ALL layers, requires OOD data)

Key insight (Papyan et al. 2020, Ben Ammar et al. 2024):
    NC forms *last-to-first* — the penultimate layer collapses first, and
    collapse propagates backward through the network during extended training.

Usage (from notebook)::

    from src.neural_collapse.nc_earlier_layer import (
        analyze_layers_single_checkpoint,
        analyze_layers_across_checkpoints,
        plot_nc_by_layer,
        plot_nc_layers_across_epochs,
        plot_nc_heatmap,
        LayerNCResult,
        LayerNCTracker,
    )

    # Single checkpoint — all NC metrics per layer
    results = analyze_layers_single_checkpoint(
        model=model, loader=train_loader,
        device="cuda", num_classes=100,
        ood_loader=svhn_loader,          # optional, for NC5
    )
    plot_nc_by_layer(results)

    # Across training — collapse propagation
    tracker = analyze_layers_across_checkpoints(
        checkpoint_dir="checkpoints/", model_class=ResNet18,
        loader=train_loader, device="cuda", num_classes=100,
        ood_loader=svhn_loader,
    )
    plot_nc_layers_across_epochs(tracker)
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from scipy.sparse.linalg import svds
from tqdm import tqdm

from .nc_analysis import _find_classifier, _load_checkpoint


# ==========================================================================
# Layer-wise hook — captures output of any named sub-module
# ==========================================================================

class _MultiLayerHook:
    """Registers forward hooks on multiple named sub-modules and stores
    the output of each, GAP-pooled to (B, D) when spatial."""

    def __init__(self):
        self.features: Dict[str, torch.Tensor] = {}
        self._handles: list = []

    def register(self, model: nn.Module, layer_names: List[str]) -> None:
        """Attach hooks.  *layer_names* are dot-separated attribute paths
        (e.g. ``"layer1"``, ``"layer2"``)."""
        self.remove()
        for name in layer_names:
            module = model
            for attr in name.split("."):
                module = getattr(module, attr)

            def _make_hook(n):
                def hook(_mod, _inp, out):
                    if out.dim() == 4:
                        pooled = out.mean(dim=[2, 3])   # GAP: (B,C,H,W) → (B,C)
                    else:
                        pooled = out
                    self.features[n] = pooled.detach()
                return hook

            h = module.register_forward_hook(_make_hook(name))
            self._handles.append(h)

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self.features.clear()


# ==========================================================================
# Data containers
# ==========================================================================

@dataclass
class LayerNCResult:
    """NC1–NC5 metrics for a single layer at a single checkpoint.

    Attributes
    ----------
    layer_name : str
        E.g. ``"layer1"``, ``"layer4"``, ``"penultimate"``.
    feature_dim : int
        Dimension D of the (GAP-pooled) feature vector at this layer.
    nc1 : float
        Tr(Σ_W Σ_B†) / C — within-class variability collapse.
    nc2_equinorm : float
        CoV of class-mean norms (lower → more equinorm).
    nc2_equiangularity : float
        Coherence measure (lower → closer to simplex ETF).
    nc3_w_m_dist : float or None
        ‖W^T − M̃‖²_F (normalized).  Only at penultimate (needs classifier W).
    nc4_ncc_mismatch : float
        1 − agreement(NCC, network) at this layer.  Defined at ALL layers.
    nc5_orthodev : float or None
        Avg_c |cos(μ_c, μ^OOD_G)| at this layer.  None if no OOD data given.
    """

    layer_name: str
    feature_dim: int
    nc1: float
    nc2_equinorm: float
    nc2_equiangularity: float
    nc3_w_m_dist: Optional[float] = None
    nc4_ncc_mismatch: Optional[float] = None
    nc5_orthodev: Optional[float] = None


@dataclass
class LayerNCTracker:
    """Tracks per-layer NC1–NC5 across training epochs.

    ``data[layer_name][metric_name]`` → list of floats (one per epoch).
    """

    epochs: List[int] = field(default_factory=list)
    layer_names: List[str] = field(default_factory=list)
    data: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)

    # All metrics tracked across epochs (NC3 only for penultimate)
    _ALL_METRICS = (
        "nc1", "nc2_equinorm", "nc2_equiangularity",
        "nc4_ncc_mismatch", "nc5_orthodev",
    )

    def _ensure_layer(self, name: str) -> None:
        if name not in self.data:
            self.data[name] = {m: [] for m in self._ALL_METRICS}

    def append(self, layer_name: str, metrics: dict) -> None:
        self._ensure_layer(layer_name)
        for key in self._ALL_METRICS:
            val = metrics.get(key)
            self.data[layer_name][key].append(
                val if val is not None else float("nan")
            )

    def summary(self) -> str:
        lines = [
            "=" * 80,
            "LAYER-WISE NEURAL COLLAPSE SUMMARY  (NC1–NC5)",
            "=" * 80,
            f"Epochs analyzed: {len(self.epochs)}",
        ]
        if self.epochs:
            lines.append(f"Epoch range: {self.epochs[0]} → {self.epochs[-1]}")
        lines.append("")
        header = (f"{'Layer':<14s} {'NC1':>8s} {'Equinorm':>9s} {'Equiang':>9s}"
                  f" {'NC4':>8s} {'NC5':>8s}")
        lines.append(header)
        lines.append("-" * len(header))
        for name in self.layer_names:
            d = self.data.get(name, {})
            def _last(key):
                vals = d.get(key, [])
                return vals[-1] if vals else float("nan")
            nc1 = _last("nc1")
            eq  = _last("nc2_equinorm")
            ea  = _last("nc2_equiangularity")
            nc4 = _last("nc4_ncc_mismatch")
            nc5 = _last("nc5_orthodev")
            nc4s = f"{nc4:>8.4f}" if not np.isnan(nc4) else "     —"
            nc5s = f"{nc5:>8.4f}" if not np.isnan(nc5) else "     —"
            lines.append(
                f"{name:<14s} {nc1:>8.4f} {eq:>9.4f} {ea:>9.4f} {nc4s} {nc5s}"
            )
        lines.append("=" * 80)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "epochs": list(self.epochs),
            "layer_names": list(self.layer_names),
            "data": {
                layer: {k: list(v) for k, v in metrics.items()}
                for layer, metrics in self.data.items()
            },
        }


# ==========================================================================
# NC5: ID/OOD orthogonality for a given set of means
# ==========================================================================

def _compute_nc5(
    id_means: torch.Tensor,
    ood_global_mean: torch.Tensor,
) -> float:
    """NC5 OrthoDev = Avg_c |cos(µ_c, µ^OOD_G)|.  Should → 0."""
    mu_ood_norm = torch.norm(ood_global_mean)
    if mu_ood_norm < 1e-12:
        return 0.0
    cosines = []
    for c in range(id_means.shape[0]):
        mu_c_norm = torch.norm(id_means[c])
        if mu_c_norm < 1e-12:
            continue
        cos = torch.abs(
            torch.dot(id_means[c], ood_global_mean) / (mu_c_norm * mu_ood_norm)
        )
        cosines.append(cos.item())
    return float(np.mean(cosines)) if cosines else 0.0


# ==========================================================================
# Core: compute NC1–NC5 on pre-extracted features for ONE layer
# ==========================================================================

def _compute_layer_nc(
    features_by_class: Dict[int, torch.Tensor],
    num_classes: int,
    device: str,
    classifier_W: Optional[torch.Tensor] = None,
    network_preds: Optional[torch.Tensor] = None,
    all_features: Optional[torch.Tensor] = None,
    ood_global_mean: Optional[torch.Tensor] = None,
) -> dict:
    """Compute NC1–NC5 on already-collected per-class features.

    Parameters
    ----------
    features_by_class : dict[int, (N_c, D)]
    num_classes : int
    device : str
    classifier_W : (C, D), optional — if given + dim matches → NC3.
    network_preds : (N,), optional — if given with *all_features* → NC4.
    all_features : (N, D), optional
    ood_global_mean : (D,), optional — if given → NC5.

    Returns
    -------
    dict with nc1, nc2_equinorm, nc2_equiangularity,
    [nc3_w_m_dist], [nc4_ncc_mismatch], [nc5_orthodev].
    """
    C = num_classes

    # ---- Class means ----
    means = []
    for c in range(C):
        if c in features_by_class and features_by_class[c].shape[0] > 0:
            means.append(features_by_class[c].mean(dim=0))
        else:
            means.append(
                torch.zeros_like(means[0]) if means
                else torch.zeros(1, device=device)
            )

    M = torch.stack(means)  # (C, D)
    D = M.shape[1]

    # Centered class means
    muG = M.mean(dim=0, keepdim=True)
    M_centered = (M - muG).T  # (D, C)

    # Between-class covariance
    Sb = (M_centered @ M_centered.T) / C

    # Within-class covariance
    Sw = torch.zeros(D, D, device=device)
    total_N = 0
    for c in range(C):
        if c not in features_by_class or features_by_class[c].shape[0] == 0:
            continue
        z = features_by_class[c] - means[c].unsqueeze(0)
        Sw += z.T @ z
        total_N += features_by_class[c].shape[0]
    if total_N > 0:
        Sw /= total_N

    # ---- NC1: Tr{Σ_W Σ_B†} / C ----
    Sw_np = Sw.cpu().float().numpy()
    Sb_np = Sb.cpu().float().numpy()
    try:
        k = min(C - 1, D - 1)
        if k < 1:
            nc1 = float("nan")
        else:
            eigvec, eigval, _ = svds(Sb_np.astype(np.float64), k=k)
            inv_Sb = eigvec @ np.diag(eigval ** (-1)) @ eigvec.T
            nc1 = float(np.trace(Sw_np @ inv_Sb) / C)
    except Exception:
        nc1 = float("nan")

    # ---- NC2: equinorm ----
    M_norms = torch.norm(M_centered, dim=0)
    nc2_equinorm = (torch.std(M_norms) / (torch.mean(M_norms) + 1e-12)).item()

    # ---- NC2: equiangularity ----
    M_normed = M_centered / (M_norms.unsqueeze(0) + 1e-12)
    G = M_normed.T @ M_normed
    G += torch.ones(C, C, device=device) / (C - 1)
    G -= torch.diag(torch.diag(G))
    nc2_equiang = torch.norm(G, p=1).item() / (C * (C - 1))

    result = {
        "nc1": nc1,
        "nc2_equinorm": nc2_equinorm,
        "nc2_equiangularity": nc2_equiang,
    }

    # ---- NC3: ||W^T − M̃||² (normalized Frobenius) — penultimate only ----
    if classifier_W is not None:
        W = classifier_W.to(device)
        if W.shape[1] == D:
            norm_M = M_centered / (torch.norm(M_centered, "fro") + 1e-12)
            norm_W = W.T / (torch.norm(W.T, "fro") + 1e-12)
            result["nc3_w_m_dist"] = (torch.norm(norm_W - norm_M) ** 2).item()

    # ---- NC4: NCC mismatch — ALL layers ----
    if network_preds is not None and all_features is not None:
        if all_features.shape[1] == D:
            dists = torch.cdist(all_features, M)  # (N, C)
            ncc_preds = dists.argmin(dim=1)
            result["nc4_ncc_mismatch"] = (
                1.0 - (ncc_preds == network_preds).float().mean().item()
            )

    # ---- NC5: ID/OOD orthogonality — ALL layers (if OOD given) ----
    if ood_global_mean is not None and ood_global_mean.shape[0] == D:
        result["nc5_orthodev"] = _compute_nc5(M, ood_global_mean)

    return result


# ==========================================================================
# Single-checkpoint analysis
# ==========================================================================

RESNET_LAYERS = ["layer1", "layer2", "layer3", "layer4"]


@torch.no_grad()
def analyze_layers_single_checkpoint(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: str,
    num_classes: int,
    layer_names: Optional[List[str]] = None,
    include_penultimate: bool = True,
    ood_loader: Optional[torch.utils.data.DataLoader] = None,
) -> List[LayerNCResult]:
    """Compute NC1–NC5 at each layer of the network.

    Parameters
    ----------
    model : nn.Module
        A model with named sub-modules (e.g. ``layer1`` … ``layer4``).
    loader : DataLoader
        ID training (or validation) data.
    device : str
    num_classes : int
    layer_names : list[str], optional
        Default: ``["layer1","layer2","layer3","layer4"]``.
    include_penultimate : bool
        Also analyse penultimate (pre-classifier) features.
    ood_loader : DataLoader, optional
        OOD data — if given, NC5 is computed at every layer.

    Returns
    -------
    list[LayerNCResult]  ordered shallow → deep.
    """
    model.to(device).eval()

    if layer_names is None:
        layer_names = list(RESNET_LAYERS)

    classifier = _find_classifier(model)
    classifier_W = classifier.weight.detach()

    # ---- Setup hooks ----
    multi_hook = _MultiLayerHook()
    multi_hook.register(model, layer_names)

    class _PenHook:
        """Captures the input to the classifier (= penultimate features)."""
        def __init__(self):
            self.features = None
        def __call__(self, mod, inp, out):
            self.features = inp[0].detach()

    pen_hook = _PenHook()
    pen_handle = (
        classifier.register_forward_hook(pen_hook)
        if include_penultimate else None
    )

    # ---- Collect ID features per class, per layer ----
    # layer_features[name][class] → list[Tensor]
    layer_features: Dict[str, Dict[int, List[torch.Tensor]]] = {
        name: {c: [] for c in range(num_classes)} for name in layer_names
    }
    # Also store raw (uncategorised) features per layer for NC4
    layer_all_features: Dict[str, List[torch.Tensor]] = {
        name: [] for name in layer_names
    }

    pen_class_features: Dict[int, List[torch.Tensor]] = {
        c: [] for c in range(num_classes)
    }
    pen_all_features: List[torch.Tensor] = []
    all_net_preds: List[torch.Tensor] = []

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        all_net_preds.append(preds.cpu())

        # Intermediate layers (already GAP-pooled by hook)
        for name in layer_names:
            h = multi_hook.features[name]  # (B, D_layer)
            layer_all_features[name].append(h.cpu())
            for c in range(num_classes):
                idxs = (targets == c).nonzero(as_tuple=True)[0]
                if len(idxs) > 0:
                    layer_features[name][c].append(h[idxs].cpu())

        # Penultimate
        if include_penultimate and pen_hook.features is not None:
            h_pen = pen_hook.features.view(images.shape[0], -1)
            pen_all_features.append(h_pen.cpu())
            for c in range(num_classes):
                idxs = (targets == c).nonzero(as_tuple=True)[0]
                if len(idxs) > 0:
                    pen_class_features[c].append(h_pen[idxs].cpu())

    # Cleanup ID hooks
    multi_hook.remove()
    if pen_handle is not None:
        pen_handle.remove()

    # Concatenate network predictions
    cat_net_preds = torch.cat(all_net_preds)  # (N,)

    # ---- Collect OOD features per layer (for NC5) ----
    # ood_global_means[layer_name] → (D,)
    ood_global_means: Dict[str, torch.Tensor] = {}

    if ood_loader is not None:
        # Re-register hooks for OOD pass
        multi_hook_ood = _MultiLayerHook()
        multi_hook_ood.register(model, layer_names)

        pen_hook_ood = _PenHook()
        pen_handle_ood = (
            classifier.register_forward_hook(pen_hook_ood)
            if include_penultimate else None
        )

        ood_sums: Dict[str, torch.Tensor] = {}
        ood_pen_sum: Optional[torch.Tensor] = None
        ood_n = 0

        for images, _ in ood_loader:
            images = images.to(device)
            _ = model(images)
            ood_n += images.shape[0]

            for name in layer_names:
                h = multi_hook_ood.features[name]
                if name not in ood_sums:
                    ood_sums[name] = torch.zeros(h.shape[1], device=device)
                ood_sums[name] += h.sum(dim=0)

            if include_penultimate and pen_hook_ood.features is not None:
                h_pen = pen_hook_ood.features.view(images.shape[0], -1)
                if ood_pen_sum is None:
                    ood_pen_sum = torch.zeros(h_pen.shape[1], device=device)
                ood_pen_sum += h_pen.sum(dim=0)

        multi_hook_ood.remove()
        if pen_handle_ood is not None:
            pen_handle_ood.remove()

        if ood_n > 0:
            for name in layer_names:
                ood_global_means[name] = (ood_sums[name] / ood_n).cpu()
            if ood_pen_sum is not None:
                ood_global_means["penultimate"] = (ood_pen_sum / ood_n).cpu()

    # ---- Compute NC1–NC5 per layer ----
    results: List[LayerNCResult] = []

    for name in layer_names:
        feats_by_class = {
            c: torch.cat(layer_features[name][c]).to(device)
            if layer_features[name][c] else torch.empty(0, device=device)
            for c in range(num_classes)
        }
        D_layer = next(
            (f.shape[1] for f in feats_by_class.values() if f.shape[0] > 0), 0
        )
        cat_all = torch.cat(layer_all_features[name]).to(device)

        ood_mean = (
            ood_global_means[name].to(device)
            if name in ood_global_means else None
        )

        metrics = _compute_layer_nc(
            feats_by_class, num_classes, device,
            classifier_W=None,                   # NC3 only at penultimate
            network_preds=cat_net_preds.to(device),
            all_features=cat_all,
            ood_global_mean=ood_mean,
        )

        results.append(LayerNCResult(
            layer_name=name,
            feature_dim=D_layer,
            nc1=metrics["nc1"],
            nc2_equinorm=metrics["nc2_equinorm"],
            nc2_equiangularity=metrics["nc2_equiangularity"],
            nc3_w_m_dist=None,
            nc4_ncc_mismatch=metrics.get("nc4_ncc_mismatch"),
            nc5_orthodev=metrics.get("nc5_orthodev"),
        ))

    # ---- Penultimate (NC3 + NC4 + NC5) ----
    if include_penultimate:
        pen_by_class = {
            c: torch.cat(pen_class_features[c]).to(device)
            if pen_class_features[c] else torch.empty(0, device=device)
            for c in range(num_classes)
        }
        D_pen = next(
            (f.shape[1] for f in pen_by_class.values() if f.shape[0] > 0), 0
        )
        cat_pen = torch.cat(pen_all_features).to(device) if pen_all_features else None

        ood_mean_pen = (
            ood_global_means["penultimate"].to(device)
            if "penultimate" in ood_global_means else None
        )

        metrics = _compute_layer_nc(
            pen_by_class, num_classes, device,
            classifier_W=classifier_W,
            network_preds=cat_net_preds.to(device),
            all_features=cat_pen,
            ood_global_mean=ood_mean_pen,
        )

        results.append(LayerNCResult(
            layer_name="penultimate",
            feature_dim=D_pen,
            nc1=metrics["nc1"],
            nc2_equinorm=metrics["nc2_equinorm"],
            nc2_equiangularity=metrics["nc2_equiangularity"],
            nc3_w_m_dist=metrics.get("nc3_w_m_dist"),
            nc4_ncc_mismatch=metrics.get("nc4_ncc_mismatch"),
            nc5_orthodev=metrics.get("nc5_orthodev"),
        ))

    return results


# ==========================================================================
# Multi-checkpoint analysis
# ==========================================================================

def analyze_layers_across_checkpoints(
    checkpoint_dir: str,
    model_class: Type[nn.Module],
    loader: torch.utils.data.DataLoader,
    device: str,
    num_classes: int,
    layer_names: Optional[List[str]] = None,
    ood_loader: Optional[torch.utils.data.DataLoader] = None,
    checkpoint_pattern: str = "resnet18_cifar100_*.pth",
    epoch_regex: str = r"epoch(\d+)",
    verbose: bool = True,
) -> LayerNCTracker:
    """Compute per-layer NC1–NC5 across training checkpoints.

    Parameters
    ----------
    checkpoint_dir : str
    model_class : type
    loader : DataLoader
        ID training data.
    device : str
    num_classes : int
    layer_names : list[str], optional
    ood_loader : DataLoader, optional
        OOD data — if given, NC5 is computed at every layer.
    checkpoint_pattern, epoch_regex : str
    verbose : bool

    Returns
    -------
    LayerNCTracker
    """
    if layer_names is None:
        layer_names = list(RESNET_LAYERS)
    all_names = layer_names + ["penultimate"]

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
    tracker = LayerNCTracker(layer_names=all_names)

    iterator = tqdm(paths, desc="Layer NC Analysis") if verbose else paths
    for ckpt_path in iterator:
        epoch = _extract_epoch(ckpt_path)
        _load_checkpoint(ckpt_path, model, device)

        layer_results = analyze_layers_single_checkpoint(
            model, loader, device, num_classes,
            layer_names=layer_names,
            include_penultimate=True,
            ood_loader=ood_loader,
        )

        tracker.epochs.append(epoch)
        for lr in layer_results:
            tracker.append(lr.layer_name, {
                "nc1": lr.nc1,
                "nc2_equinorm": lr.nc2_equinorm,
                "nc2_equiangularity": lr.nc2_equiangularity,
                "nc4_ncc_mismatch": lr.nc4_ncc_mismatch,
                "nc5_orthodev": lr.nc5_orthodev,
            })

        if verbose:
            parts = [f"Epoch {epoch:>4d}"]
            for lr in layer_results:
                nc4s = f"NC4={lr.nc4_ncc_mismatch:.2f}" if lr.nc4_ncc_mismatch is not None else ""
                nc5s = f"NC5={lr.nc5_orthodev:.3f}" if lr.nc5_orthodev is not None else ""
                parts.append(f"{lr.layer_name}:NC1={lr.nc1:.2f} {nc4s} {nc5s}".rstrip())
            print("  " + " | ".join(parts))

    return tracker


# ==========================================================================
# Plotting — single checkpoint
# ==========================================================================

def plot_nc_by_layer(
    results: List[LayerNCResult],
    title_suffix: str = "",
    save_dir: Optional[str] = None,
) -> plt.Figure:
    """Bar chart of NC1–NC5 across layers (single checkpoint).

    Layout: 2 rows × 3 cols.
        (0,0) NC1   (0,1) NC2 equinorm   (0,2) NC2 equiangularity
        (1,0) NC4   (1,1) NC5            (1,2) NC3 info box
    """
    names = [r.layer_name for r in results]
    dims = [r.feature_dim for r in results]
    labels = [f"{n}\n(D={d})" for n, d in zip(names, dims)]
    x = np.arange(len(names))
    colors = plt.cm.viridis(np.linspace(0.25, 0.85, len(names)))

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))

    # ---- Row 0: NC1, NC2 equinorm, NC2 equiangularity ----
    def _bar(ax, vals, ylabel, title):
        ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=13)
        ax.grid(axis="y", alpha=0.3)

    _bar(axes[0, 0], [r.nc1 for r in results],
         r"$\mathrm{Tr}(\Sigma_W \Sigma_B^{\dagger}) / C$",
         "NC1: Activation Collapse")

    _bar(axes[0, 1], [r.nc2_equinorm for r in results],
         "Std / Mean of Norms", "NC2: Equinorm")

    _bar(axes[0, 2], [r.nc2_equiangularity for r in results],
         r"Avg $|\cos\theta + 1/(C-1)|$", "NC2: Equiangularity")

    # ---- Row 1: NC4, NC5, NC3 info ----
    nc4_vals = [r.nc4_ncc_mismatch if r.nc4_ncc_mismatch is not None else 0 for r in results]
    _bar(axes[1, 0], nc4_vals,
         "1 − NCC agreement", "NC4: NCC Mismatch")

    nc5_vals = [r.nc5_orthodev if r.nc5_orthodev is not None else 0 for r in results]
    has_nc5 = any(r.nc5_orthodev is not None for r in results)
    if has_nc5:
        _bar(axes[1, 1], nc5_vals,
             r"Avg$_c |\cos(\mu_c, \mu^{OOD}_G)|$",
             "NC5: ID/OOD Orthogonality")
    else:
        axes[1, 1].axis("off")
        axes[1, 1].text(
            0.5, 0.5, "NC5: no OOD data provided",
            transform=axes[1, 1].transAxes, ha="center", va="center",
            fontsize=12, color="gray",
        )

    # NC3 info box (penultimate only)
    ax = axes[1, 2]
    ax.axis("off")
    pen = next((r for r in results if r.layer_name == "penultimate"), None)
    info_lines = []
    if pen is not None:
        if pen.nc3_w_m_dist is not None:
            info_lines.append(f"NC3 (W ≈ M): {pen.nc3_w_m_dist:.4f}")
        if pen.nc4_ncc_mismatch is not None:
            info_lines.append(f"NC4 (NCC):   {pen.nc4_ncc_mismatch:.4f}")
        if pen.nc5_orthodev is not None:
            info_lines.append(f"NC5 (Orth):  {pen.nc5_orthodev:.6f}")
    if info_lines:
        ax.text(
            0.1, 0.5,
            "Penultimate layer:\n\n" + "\n".join(info_lines),
            transform=ax.transAxes, fontsize=12, va="center",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
        )
    ax.set_title("Summary (penultimate)", fontsize=13)

    fig.suptitle(
        f"Neural Collapse Across Network Depth (NC1–NC5){title_suffix}",
        fontsize=15, y=1.02,
    )
    fig.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "nc_by_layer.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f" Saved: {path}")

    return fig


# ==========================================================================
# Plotting — across checkpoints
# ==========================================================================

def plot_nc_layers_across_epochs(
    tracker: LayerNCTracker,
    save_dir: Optional[str] = None,
) -> plt.Figure:
    """Line plots of NC1–NC5 vs epoch, one line per layer.

    Layout: 2 rows × 3 cols (NC1, NC2eq, NC2ea, NC4, NC5, empty/info).
    """
    epochs = tracker.epochs
    layers = tracker.layer_names
    cmap = plt.cm.get_cmap("plasma", len(layers) + 1)

    metric_specs = [
        ("nc1", r"$\mathrm{Tr}(\Sigma_W \Sigma_B^{\dagger}) / C$",
         "NC1: Activation Collapse", True),
        ("nc2_equinorm", "Std / Mean of Norms",
         "NC2: Equinorm", False),
        ("nc2_equiangularity", r"Avg $|\cos\theta + 1/(C-1)|$",
         "NC2: Equiangularity", False),
        ("nc4_ncc_mismatch", "1 − NCC agreement",
         "NC4: NCC Mismatch", False),
        ("nc5_orthodev", r"Avg$_c |\cos(\mu_c, \mu^{OOD}_G)|$",
         "NC5: ID/OOD Orthogonality", False),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    flat_axes = [axes[0, 0], axes[0, 1], axes[0, 2],
                 axes[1, 0], axes[1, 1]]

    for ax, (metric_key, ylabel, title, use_log) in zip(flat_axes, metric_specs):
        any_data = False
        for i, layer in enumerate(layers):
            vals = tracker.data.get(layer, {}).get(metric_key, [])
            if not vals or all(np.isnan(v) for v in vals):
                continue
            any_data = True
            plot_fn = ax.semilogy if use_log else ax.plot
            plot_fn(
                epochs[:len(vals)], vals,
                "-o", markersize=3, linewidth=1.5,
                color=cmap(i), label=layer,
            )
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=13)
        if any_data:
            ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[1, 2].axis("off")

    fig.suptitle(
        "Neural Collapse (NC1–NC5) Across Layers During Training",
        fontsize=15, y=1.02,
    )
    fig.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "nc_layers_across_epochs.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f" Saved: {path}")

    return fig


def plot_nc_heatmap(
    tracker: LayerNCTracker,
    metric: str = "nc1",
    save_dir: Optional[str] = None,
) -> plt.Figure:
    """Heatmap: layers (y) × epochs (x) coloured by a chosen NC metric.

    Parameters
    ----------
    metric : {"nc1", "nc2_equinorm", "nc2_equiangularity",
              "nc4_ncc_mismatch", "nc5_orthodev"}
    """
    epochs = tracker.epochs
    layers = tracker.layer_names

    matrix = np.full((len(layers), len(epochs)), np.nan)
    for i, layer in enumerate(layers):
        vals = tracker.data.get(layer, {}).get(metric, [])
        for j, v in enumerate(vals):
            matrix[i, j] = v

    fig, ax = plt.subplots(figsize=(max(10, len(epochs) * 0.3), 4))

    use_log = metric in ("nc1",)
    data = np.log10(matrix + 1e-12) if use_log else matrix

    im = ax.imshow(data, aspect="auto", cmap="viridis_r", interpolation="nearest")
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers, fontsize=10)

    step = max(1, len(epochs) // 15)
    tick_idx = list(range(0, len(epochs), step))
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([str(epochs[i]) for i in tick_idx], fontsize=9)
    ax.set_xlabel("Epoch", fontsize=11)

    label = f"log₁₀({metric})" if use_log else metric
    metric_titles = {
        "nc1": "NC1: Activation Collapse",
        "nc2_equinorm": "NC2: Equinorm",
        "nc2_equiangularity": "NC2: Equiangularity",
        "nc4_ncc_mismatch": "NC4: NCC Mismatch",
        "nc5_orthodev": "NC5: ID/OOD Orthogonality",
    }
    ax.set_title(
        f"{metric_titles.get(metric, metric)} — Layer × Epoch",
        fontsize=13,
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(label, fontsize=10)
    fig.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"nc_heatmap_{metric}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f" Saved: {path}")

    return fig


# ==========================================================================
# Serialization
# ==========================================================================

def save_layer_metrics_yaml(tracker: LayerNCTracker, path: str) -> None:
    """Save layer-wise tracker metrics to YAML."""
    data = tracker.to_dict()
    for layer, metrics in data.get("data", {}).items():
        for key, values in metrics.items():
            data["data"][layer][key] = [
                v if not (isinstance(v, float) and np.isnan(v)) else None
                for v in values
            ]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    print(f" Layer metrics saved to: {path}")

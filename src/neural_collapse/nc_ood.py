from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from scipy.sparse.linalg import svds
from tqdm import tqdm
import plotly.graph_objects as go
import plotly.express as px


from .nc_analysis import _FeatureHook, _find_classifier, _load_checkpoint


# ==========================================================================
# Data container
# ==========================================================================

@dataclass
class NCOODTracker:

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
            "NC OOD METRICS SUMMARY",
            "=" * 60,
            f"Epochs analyzed: {len(self.epochs)}",
        ]
        if self.epochs:
            lines.append(f"Epoch range: {self.epochs[0]} → {self.epochs[-1]}")
        if self.nc5_orthodev:
            lines.append(f"Final NC5 OrthoDev: {self.nc5_orthodev[-1]:.6f}")
        if self.Sw_invSb_id_only:
            lines.append(f"Final NC1 (ID only): {self.Sw_invSb_id_only[-1]:.4f}")
        if self.Sw_invSb_id_ood:
            lines.append(f"Final NC1 (ID+OOD):  {self.Sw_invSb_id_ood[-1]:.4f}")
        if self.nc2_equiang_id_only:
            lines.append(f"Final NC2 equiang (ID):    {self.nc2_equiang_id_only[-1]:.4f}")
        if self.nc2_equiang_id_ood:
            lines.append(f"Final NC2 equiang (ID+OOD):{self.nc2_equiang_id_ood[-1]:.4f}")
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


# ==========================================================================
# NC5: ID/OOD Orthogonality  (eq. 5)
# ==========================================================================

def _compute_nc5(
    id_means: torch.Tensor,
    ood_global_mean: torch.Tensor,
) -> float:
    mu_ood_norm = torch.norm(ood_global_mean)
    if mu_ood_norm < 1e-12:
        return 0.0

    cosines = []
    for c in range(id_means.shape[0]):
        mu_c = id_means[c]
        mu_c_norm = torch.norm(mu_c)
        if mu_c_norm < 1e-12:
            continue
        cos = torch.abs(
            torch.dot(mu_c, ood_global_mean) / (mu_c_norm * mu_ood_norm)
        )
        cosines.append(cos.item())

    return float(np.mean(cosines)) if cosines else 0.0


# ==========================================================================
# Helpers
# ==========================================================================

def _coherence(V: torch.Tensor, K: int, device: str) -> float:
    G = V.T @ V
    G += torch.ones(K, K, device=device) / (K - 1)
    G -= torch.diag(torch.diag(G))
    return torch.norm(G, p=1).item() / (K * (K - 1))


# ==========================================================================
# Core metric computation for one checkpoint
# ==========================================================================

@torch.no_grad()
def _compute_nc_ood_metrics(
    model: nn.Module,
    hook: _FeatureHook,
    id_loader: torch.utils.data.DataLoader,
    ood_loader: torch.utils.data.DataLoader,
    device: str,
    num_classes: int,
) -> dict:

    model.eval()
    C = num_classes
    C_total = C + 1

    # ---- Pass 1: class means ----
    N_id = [0] * C
    mean_id = [None] * C

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

    # OOD global mean
    ood_sum = torch.zeros(D, device=device)
    N_ood = 0
    for images, _ in ood_loader:
        images = images.to(device)
        _ = model(images)
        h = hook.features.view(images.shape[0], -1)
        ood_sum += h.sum(dim=0)
        N_ood += images.shape[0]
    mean_ood = ood_sum / max(N_ood, 1)

    M_id = torch.stack(mean_id)                                   # (C, D)
    M_all = torch.cat([M_id, mean_ood.unsqueeze(0)], dim=0)       # (C+1, D)

    # ---- NC5 ----
    nc5 = _compute_nc5(M_id, mean_ood)

    # ---- Centered class means ----
    muG_id = M_id.mean(dim=0, keepdim=True)
    M_id_centered = (M_id - muG_id).T                             # (D, C)

    muG_all = M_all[:C].mean(dim=0, keepdim=True)                 # global mean from ID
    M_all_centered = (M_all - muG_all).T                          # (D, C+1)

    # ---- NC2: equinorm (ID only) ----
    M_norms_id = torch.norm(M_id_centered, dim=0)
    nc2_equinorm_id = (torch.std(M_norms_id) / (torch.mean(M_norms_id) + 1e-12)).item()

    # ---- NC2: equiangularity ----
    M_id_normed = M_id_centered / (M_norms_id.unsqueeze(0) + 1e-12)
    nc2_equiang_id = _coherence(M_id_normed, C, device)

    M_norms_all = torch.norm(M_all_centered, dim=0)
    M_all_normed = M_all_centered / (M_norms_all.unsqueeze(0) + 1e-12)
    nc2_equiang_id_ood = _coherence(M_all_normed, C_total, device)

    # ---- Pass 2: within-class covariance ----
    Sw_id = torch.zeros(D, D, device=device)
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

    Sw_ood = torch.zeros(D, D, device=device)
    for images, _ in ood_loader:
        images = images.to(device)
        _ = model(images)
        h = hook.features.view(images.shape[0], -1)
        z = h - mean_ood.unsqueeze(0)
        Sw_ood += z.T @ z

    total_N_id = sum(N_id)
    Sw_id_avg = Sw_id / max(total_N_id, 1)
    Sw_all_avg = (Sw_id + Sw_ood) / max(total_N_id + N_ood, 1)

    # ---- NC1 ----
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
# Main entry point
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
        path = os.path.join(
            save_dir,
            f"nc5_convergence_{ood_name.lower().replace('-', '')}.png",
        )
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f" Saved: {path}")

    return fig


def plot_ood_summary(
    tracker: NCOODTracker,
    ood_name: str = "OOD",
    save_dir: Optional[str] = None,
) -> plt.Figure:

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

    # (0,1) NC1
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

    # (1,0) NC2 equinorm
    ax = axes[1, 0]
    ax.plot(epochs, tracker.nc2_equinorm_id, "m-o", markersize=3)
    ax.set_ylabel("Std / Mean of Norms")
    ax.set_title("NC2: Equinorm (ID)")
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)

    # (1,1) info box
    ax = axes[1, 1]
    ax.axis("off")
    info = [
        f"Final NC5 OrthoDev: {tracker.nc5_orthodev[-1]:.6f}",
        f"Final NC1 (ID):     {tracker.Sw_invSb_id_only[-1]:.4f}",
        f"Final NC1 (ID+OOD): {tracker.Sw_invSb_id_ood[-1]:.4f}",
        f"Final NC2 eq (ID):  {tracker.nc2_equiang_id_only[-1]:.4f}",
        f"Final NC2 eq (+OOD):{tracker.nc2_equiang_id_ood[-1]:.4f}",
    ]
    ax.text(
        0.1, 0.5, "\n".join(info),
        transform=ax.transAxes, fontsize=12, verticalalignment="center",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )
    ax.set_title("Summary")

    axes[1, 2].axis("off")
    fig.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(
            save_dir,
            f"nc_ood_summary_{ood_name.lower().replace('-', '')}.png",
        )
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f" Saved: {path}")

    return fig


# ==========================================================================
# Serialization
# ==========================================================================

def save_ood_metrics_yaml(tracker: NCOODTracker, path: str) -> None:
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
    print(f" OOD metrics saved to: {path}")


# ==========================================================================
# Feature extraction (shared helper for PCA plots)
# ==========================================================================

@torch.no_grad()
def _extract_features(
    model: nn.Module,
    hook: _FeatureHook,
    loader: torch.utils.data.DataLoader,
    device: str,
    max_samples: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

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
# PCA 2-D projection (static, matplotlib)
# ==========================================================================

def plot_pca_2d(
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

    var_pct = pca.explained_variance_ratio_ * 100
    ax.set_xlabel(f"PC 1 ({var_pct[0]:.1f}%)")
    ax.set_ylabel(f"PC 2 ({var_pct[1]:.1f}%)")
    ax.set_title(f"PCA — {id_name} (colored) vs {ood_name} (gray)")
    ax.legend(fontsize=11, markerscale=3)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(
            save_dir,
            f"pca_2d_{ood_name.lower().replace('-', '')}.png",
        )
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f" Saved: {path}")

    return fig


# ==========================================================================
# PCA 3-D projection (interactive, Plotly)
# ==========================================================================

@torch.no_grad()
def plot_pca_3d_interactive(
    model: nn.Module,
    id_loader: torch.utils.data.DataLoader,
    ood_loader: torch.utils.data.DataLoader,
    device: str,
    num_classes: int,
    id_train_loader: Optional[torch.utils.data.DataLoader] = None,
    id_name: str = "ID",
    ood_name: str = "OOD",
    max_samples: int = 3000,
    show_class_means: bool = True,
    show_ood_mean: bool = True,
    save_path: Optional[str] = None,
):

    model.to(device).eval()
    classifier = _find_classifier(model)
    hook = _FeatureHook()
    handle = classifier.register_forward_hook(hook)

    try:
        # Fit PCA on training features
        train_ldr = id_train_loader if id_train_loader is not None else id_loader
        train_feats, _, _ = _extract_features(
            model, hook, train_ldr, device, max_samples=50_000
        )
        pca = PCA(n_components=3)
        pca.fit(train_feats.numpy())

        # Project ID
        id_feats, id_labels, _ = _extract_features(
            model, hook, id_loader, device, max_samples=max_samples
        )
        id_proj = pca.transform(id_feats.numpy())

        # Project OOD
        ood_feats, _, _ = _extract_features(
            model, hook, ood_loader, device, max_samples=max_samples
        )
        ood_proj = pca.transform(ood_feats.numpy())
    finally:
        handle.remove()

    # Variance explained
    var_pct = pca.explained_variance_ratio_ * 100

    # --- Build plotly figure ---
    fig = go.Figure()

    # ID samples (one trace per class, grouped in legend)
    unique_labels = np.unique(id_labels.numpy())
    palette = px.colors.qualitative.Alphabet
    if len(unique_labels) > len(palette):
        palette = palette * (len(unique_labels) // len(palette) + 1)

    for i, c in enumerate(unique_labels):
        mask = id_labels.numpy() == c
        fig.add_trace(go.Scatter3d(
            x=id_proj[mask, 0], y=id_proj[mask, 1], z=id_proj[mask, 2],
            mode="markers",
            marker=dict(size=2, color=palette[i % len(palette)], opacity=0.4),
            name=f"Class {c}",
            legendgroup="id",
            showlegend=(i < 5),
            hovertemplate=f"Class {c}<br>PC1=%{{x:.2f}}<br>PC2=%{{y:.2f}}<br>PC3=%{{z:.2f}}",
        ))

    # OOD samples
    fig.add_trace(go.Scatter3d(
        x=ood_proj[:, 0], y=ood_proj[:, 1], z=ood_proj[:, 2],
        mode="markers",
        marker=dict(size=2, color="gray", opacity=0.3, symbol="x"),
        name=f"{ood_name} (OOD)",
        legendgroup="ood",
        hovertemplate=f"{ood_name}<br>PC1=%{{x:.2f}}<br>PC2=%{{y:.2f}}<br>PC3=%{{z:.2f}}",
    ))

    # Class means
    if show_class_means:
        id_means_3d = []
        for c in unique_labels:
            mask = id_labels.numpy() == c
            id_means_3d.append(id_proj[mask].mean(axis=0))
        id_means_3d = np.array(id_means_3d)

        fig.add_trace(go.Scatter3d(
            x=id_means_3d[:, 0], y=id_means_3d[:, 1], z=id_means_3d[:, 2],
            mode="markers",
            marker=dict(size=6, color="black", symbol="diamond", opacity=0.9),
            name=f"{id_name} class means",
            hovertemplate="Class mean<br>PC1=%{x:.2f}<br>PC2=%{y:.2f}<br>PC3=%{z:.2f}",
        ))

    # OOD global mean + line from origin
    if show_ood_mean:
        ood_mean_3d = ood_proj.mean(axis=0)

        fig.add_trace(go.Scatter3d(
            x=[ood_mean_3d[0]], y=[ood_mean_3d[1]], z=[ood_mean_3d[2]],
            mode="markers",
            marker=dict(size=10, color="red", symbol="diamond"),
            name=f"{ood_name} global mean (mu_G^OOD)",
            hovertemplate=(
                f"{ood_name} global mean<br>"
                "PC1=%{x:.2f}<br>PC2=%{y:.2f}<br>PC3=%{z:.2f}"
            ),
        ))

        # Line from origin to OOD mean
        fig.add_trace(go.Scatter3d(
            x=[0, ood_mean_3d[0]], y=[0, ood_mean_3d[1]], z=[0, ood_mean_3d[2]],
            mode="lines",
            line=dict(color="red", width=4, dash="dash"),
            name="Origin -> mu_G^OOD",
            showlegend=False,
        ))

        # Origin marker
        fig.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[0],
            mode="markers",
            marker=dict(size=6, color="black", symbol="cross"),
            name="Origin",
        ))

    fig.update_layout(
        title=dict(
            text=(
                f"3D PCA - {id_name} vs {ood_name}<br>"
                f"<sub>Var explained: PC1={var_pct[0]:.1f}%, "
                f"PC2={var_pct[1]:.1f}%, PC3={var_pct[2]:.1f}%</sub>"
            ),
            x=0.5,
        ),
        scene=dict(
            xaxis_title=f"PC1 ({var_pct[0]:.1f}%)",
            yaxis_title=f"PC2 ({var_pct[1]:.1f}%)",
            zaxis_title=f"PC3 ({var_pct[2]:.1f}%)",
        ),
        width=900, height=700,
        legend=dict(itemsizing="constant"),
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.write_html(save_path)
        print(f" Saved: {save_path}")

    return fig

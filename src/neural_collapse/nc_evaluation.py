import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple


def extract_features(model, loader, device='cuda'):
    """
    Extract penultimate features and labels from a model over a dataloader.

    Uses [`ResNet18.get_features`](src/models/resnet.py).
    """
    model = model.to(device)
    model.eval()

    feats = []
    labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            f = model.get_features(x)  # [B, D]
            if isinstance(f, tuple):  # defensive
                f = f[1]
            feats.append(f.cpu())
            labels.append(y.cpu())

    feats = torch.cat(feats, dim=0)  # [N, D]
    labels = torch.cat(labels, dim=0)  # [N]
    return feats, labels


def compute_class_means(features: torch.Tensor, labels: torch.Tensor, num_classes: int = None) -> torch.Tensor:
    """Compute per-class mean vectors [K, D]."""
    if num_classes is None:
        num_classes = int(labels.max().item()) + 1

    D = features.size(1)
    means = torch.zeros((num_classes, D), dtype=features.dtype)
    counts = torch.zeros(num_classes, dtype=torch.long)

    for c in range(num_classes):
        mask = labels == c
        if mask.sum() == 0:
            continue
        means[c] = features[mask].mean(dim=0)
        counts[c] = mask.sum()

    return means, counts


def nc1_within_total_ratio(features: torch.Tensor, labels: torch.Tensor, means: torch.Tensor) -> float:
    """
    NC1: ratio of within-class scatter to total scatter.
    Returns scalar in [0,1], smaller is more collapsed.
    """
    N = features.size(0)
    global_mean = features.mean(dim=0)
    # within-class scatter
    within = 0.0
    for c in range(means.size(0)):
        mask = labels == c
        if mask.sum() == 0:
            continue
        diffs = features[mask] - means[c:c+1]
        within += (diffs ** 2).sum().item()
    within /= N
    # total scatter
    tot = ((features - global_mean) ** 2).sum().item() / N
    if tot == 0:
        return float('nan')
    return within / tot


def nc2_simplex_deviation(means: torch.Tensor) -> Dict[str, float]:
    """
    NC2: measure how close class means are to an equiangular simplex.
    Returns average off-diagonal cosine and deviation from target -1/(K-1).
    """
    K = means.size(0)
    # normalize means
    M = F.normalize(means, dim=1)  # [K, D]
    S = (M @ M.t()).cpu().numpy()  # [K, K]
    # ignore diagonal
    off_diag = S[~np.eye(K, dtype=bool)].reshape(K, K-1)
    avg_off = off_diag.mean()
    target = -1.0 / (K - 1)
    deviation = float(np.abs(avg_off - target))
    return {'avg_off_diagonal_cosine': float(avg_off), 'simplex_target': float(target), 'nc2_deviation': deviation}


def nc3_classifier_alignment(model, means: torch.Tensor) -> float:
    """
    NC3: alignment between classifier weight vectors and class means.

    Uses [`ResNet18.get_classifier_weights`](src/models/resnet.py).
    Returns mean cosine similarity across classes (higher = better alignment).
    """
    W = model.get_classifier_weights().cpu()  # [K, D]
    # ensure same ordering and dims
    K = min(W.size(0), means.size(0))
    Wn = F.normalize(W[:K], dim=1)
    Mn = F.normalize(means[:K], dim=1)
    cos = (Wn * Mn).sum(dim=1).cpu().numpy()
    return float(np.mean(cos))


def nc4_self_duality(model, means: torch.Tensor) -> float:
    """
    NC4: self-duality / per-class duality measure.

    Compute mean diagonal cosine of normalized classifier weights vs. means.
    Returns mean diagonal cosine (closer to 1 => strong self-duality).
    """
    W = model.get_classifier_weights().cpu()  # [K, D]
    K = min(W.size(0), means.size(0))
    Wn = F.normalize(W[:K], dim=1)
    Mn = F.normalize(means[:K], dim=1)
    diag_cos = (Wn * Mn).sum(dim=1).cpu().numpy()
    return float(np.mean(diag_cos))


def evaluate_nc(model, loader, device='cuda', num_classes: int = None) -> Dict[str, float]:
    """
    Full evaluation pipeline returning NC metrics dict:
      - nc1_ratio: within/total scatter (smaller is better)
      - nc2_avg_off: average off-diagonal cosine between class means
      - nc2_deviation: deviation from equiangular simplex target
      - nc3_alignment: mean cosine between classifier rows and class means
      - nc4_self_duality: mean diagonal cosine (weights vs means)
    """
    features, labels = extract_features(model, loader, device=device)
    if num_classes is None:
        num_classes = int(labels.max().item()) + 1

    means, counts = compute_class_means(features, labels, num_classes=num_classes)

    metrics = {}
    metrics['nc1_ratio'] = nc1_within_total_ratio(features, labels, means)
    nc2 = nc2_simplex_deviation(means)
    metrics['nc2_avg_off_diagonal_cosine'] = nc2['avg_off_diagonal_cosine']
    metrics['nc2_deviation'] = nc2['nc2_deviation']
    metrics['nc3_alignment'] = nc3_classifier_alignment(model, means)
    metrics['nc4_self_duality'] = nc4_self_duality(model, means)
    metrics['num_classes_present'] = int((counts > 0).sum().item())

    return metrics


if __name__ == "__main__":
    #Example usage (uncomment to run; uses helpers in the repo):
    from src.utils.training import load_model  # noqa: E402
    from src.data.datasets import get_cifar100_loaders  # noqa: E402
    model, _ = load_model('checkpoints/resnet18_cifar100_best.pth', device='cpu')
    train_loader, val_loader, test_loader = get_cifar100_loaders(batch_size=128, augment=False)
    metrics = evaluate_nc(model, val_loader, device='cpu')
    print(metrics)
    pass
#

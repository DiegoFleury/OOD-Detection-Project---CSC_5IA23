"""
OOD detection metrics (AUROC, FPR@95)
"""

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def compute_auroc(id_scores, ood_scores):
    """
    Compute AUROC for OOD detection
    
    Args:
        id_scores: scores for ID samples [N_id]
        ood_scores: scores for OOD samples [N_ood]
    
    Returns:
        auroc: float in [0.5, 1.0]
    """
    y_true = np.concatenate([
        np.zeros(len(id_scores)),  # ID = 0
        np.ones(len(ood_scores))   # OOD = 1
    ])
    y_scores = np.concatenate([id_scores, ood_scores])
    
    return roc_auc_score(y_true, y_scores)


def compute_fpr_at_tpr(id_scores, ood_scores, tpr_target=0.95):
    """
    Compute FPR when TPR = tpr_target
    
    Args:
        id_scores: scores for ID samples [N_id]
        ood_scores: scores for OOD samples [N_ood]
        tpr_target: target TPR (default 0.95)
    
    Returns:
        fpr: float, percentage of ID classified as OOD
    """
    y_true = np.concatenate([
        np.zeros(len(id_scores)),
        np.ones(len(ood_scores))
    ])
    y_scores = np.concatenate([id_scores, ood_scores])
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    # Find FPR at TPR closest to target
    idx = np.argmin(np.abs(tpr - tpr_target))
    
    return fpr[idx] * 100  # Return as percentage
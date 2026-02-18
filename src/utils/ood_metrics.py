import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def compute_auroc(id_scores, ood_scores):

    y_true = np.concatenate([
        np.zeros(len(id_scores)),  # ID = 0
        np.ones(len(ood_scores))   # OOD = 1
    ])
    y_scores = np.concatenate([id_scores, ood_scores])
    
    return roc_auc_score(y_true, y_scores)


def compute_fpr_at_tpr(id_scores, ood_scores, tpr_target=0.95):

    y_true = np.concatenate([
        np.zeros(len(id_scores)),
        np.ones(len(ood_scores))
    ])
    y_scores = np.concatenate([id_scores, ood_scores])
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    # Find FPR at TPR closest to target
    idx = np.argmin(np.abs(tpr - tpr_target))
    
    return fpr[idx] * 100  # Return as percentage
"""
Neural Collapse (NECO) OOD scorer
Requires terminal phase training
"""

import torch
import numpy as np
from sklearn.decomposition import PCA
from .base import BaseFeatureScorer


class NECOScorer(BaseFeatureScorer):
    """
    NECO (Neural Collapse for OOD detection)

    Based on the paper: "NECO: NEural Collapse Based Out-of-Distribution Detection"
    Published at ICLR 2024

    Leverages neural collapse properties in terminal phase:
    - Computes NECO(x) = ||P h(x)|| / ||h(x)||
    - Where P is the PCA projection matrix on the first d principal components

    Only works with models trained in Terminal Phase Training (TPT)
    """

    def __init__(self, model, device='cuda', n_components=None):
        super().__init__(model, device)
        self.pca = None
        self.n_components = n_components  # d in the paper
    
    def fit(self, train_loader, num_classes=100):
        """
        Fit PCA on training features to extract the Simplex ETF subspace

        Args:
            train_loader: DataLoader with (images, labels)
            num_classes: number of classes (not used, kept for compatibility)
        """
        features_list = []

        with torch.no_grad():
            for batch in train_loader:
                images, labels = batch
                images = images.to(self.device)

                features = self.get_penultimate_features(images)
                features_list.append(features.cpu())

        features = torch.cat(features_list, dim=0).numpy()  # [N, D]

        # Fit PCA on training features to identify the Simplex ETF subspace
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(features)
    
    
    def score(self, x):
        """
        Compute NECO OOD score

        Based on equation (6) from the paper:
        NECO(x) = ||P h(x)|| / ||h(x)||

        Where P is the PCA projection matrix (d principal components)

        Args:
            x: input batch [B, C, H, W]

        Returns:
            scores [B] - higher = more ID (lower = more OOD)
        """
        if self.pca is None:
            raise RuntimeError("Must call fit() before score()")

        features = self.get_penultimate_features(x)  # [B, D]
        features_np = features.cpu().numpy()

        # Project features onto the principal subspace (Simplex ETF subspace)
        features_projected = self.pca.transform(features_np)  # [B, d]

        # Compute norms
        norm_projected = np.linalg.norm(features_projected, axis=1)  # [B]
        norm_original = np.linalg.norm(features_np, axis=1)  # [B]

        # NECO score: ||P h(x)|| / ||h(x)||
        neco_scores = norm_projected / (norm_original + 1e-8)

        # Convert to torch tensor
        neco_scores = torch.from_numpy(neco_scores).float().to(self.device)

        return neco_scores
    
    def __repr__(self):
        return "NECO"

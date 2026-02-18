import torch
import numpy as np
from sklearn.decomposition import PCA
from .base import BaseFeatureScorer


class NECOScorer(BaseFeatureScorer):
    def __init__(self, model, device='cuda', n_components=None):
        super().__init__(model, device)
        self.n_components = n_components
        self.components = None  # [d, D] - principal components

    def fit(self, train_loader, num_classes=100):
        features_list = []
        with torch.no_grad():
            for images, _ in train_loader:
                features_list.append(
                    self.get_penultimate_features(images.to(self.device)).cpu()
                )
        features = torch.cat(features_list, dim=0).numpy()

        pca = PCA(n_components=self.n_components)
        pca.fit(features)

        self.components = torch.tensor(
            pca.components_, dtype=torch.float32
        ).to(self.device)  # [d, D]

    def score(self, x):
        features = self.get_penultimate_features(x)  # [B, D]

        # ||P h(x)||: project onto principal subspace and measure norm
        proj = features @ self.components.T  # [B, d]
        reconstructed = proj @ self.components  # [B, D]  (= P h(x))
        norm_projected = torch.norm(reconstructed, dim=1)  # [B]
        norm_original = torch.norm(features, dim=1)        # [B]

        # NECO = ||P h(x)|| / ||h(x)||  → higher = more ID
        # negate so that higher = more OOD
        return -(norm_projected / (norm_original + 1e-8))

    def __repr__(self):
        return f"NECO(n_components={self.n_components})"
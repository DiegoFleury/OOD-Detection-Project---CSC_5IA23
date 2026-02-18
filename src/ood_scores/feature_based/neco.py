import torch
import numpy as np
from sklearn.decomposition import PCA
from .base import BaseFeatureScorer


class NECOScorer(BaseFeatureScorer):
    def __init__(self, model, device='cuda', n_components=None):
        super().__init__(model, device)
        self.n_components = n_components
        self.components = None  # [d, D]

    def fit(self, train_loader, num_classes=100):
        features_list = []
        with torch.no_grad():
            for images, _ in train_loader:
                features_list.append(
                    self.get_penultimate_features(images.to(self.device)).cpu()
                )
        features = torch.cat(features_list, dim=0).numpy()

        # Default: use num_classes as subspace dimension (ETF has C-1 dims)
        n_comp = self.n_components if self.n_components is not None else num_classes
        n_comp = min(n_comp, features.shape[1], features.shape[0])

        pca = PCA(n_components=n_comp)
        pca.fit(features)

        self.components = torch.tensor(
            pca.components_, dtype=torch.float32
        ).to(self.device)  # [d, D]

    def score(self, x):
        features = self.get_penultimate_features(x)  # [B, D]

        proj = features @ self.components.T  # [B, d]
        reconstructed = proj @ self.components  # [B, D]
        norm_projected = torch.norm(reconstructed, dim=1)  # [B]
        norm_original = torch.norm(features, dim=1)        # [B]

        # NECO = ||Ph(x)|| / ||h(x)|| -> higher = more ID
        # Negate so higher = more OOD
        return -(norm_projected / (norm_original + 1e-8))

    def __repr__(self):
        return f"NECO(n_components={self.n_components})"
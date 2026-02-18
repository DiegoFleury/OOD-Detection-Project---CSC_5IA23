import torch
import numpy as np
from sklearn.decomposition import PCA
from .base import BaseFeatureScorer


class ViMScorer(BaseFeatureScorer):
    def __init__(self, model, device='cuda', dim_reduction=512):
        super().__init__(model, device)
        self.dim_reduction = dim_reduction
        self.principal_space = None
        self.null_space = None
        self.feature_mean = None
        self.alpha = None

    def fit(self, train_loader, num_classes=100):
        features_list = []
        max_logits_list = []

        with torch.no_grad():
            for images, _ in train_loader:
                images = images.to(self.device)
                features_list.append(self.get_penultimate_features(images).cpu())
                max_logits_list.append(self.get_logits(images).max(dim=1).values.cpu())

        features = torch.cat(features_list, dim=0)
        max_logits = torch.cat(max_logits_list, dim=0)

        self.feature_mean = features.mean(dim=0).to(self.device)
        features_centered = features - features.mean(dim=0)

        pca = PCA(n_components=self.dim_reduction)
        pca.fit(features_centered.numpy())
        self.principal_space = torch.tensor(
            pca.components_.T, dtype=torch.float32
        ).to(self.device)

        pca_full = PCA(n_components=features_centered.shape[1])
        pca_full.fit(features_centered.numpy())
        self.null_space = torch.tensor(
            pca_full.components_.T, dtype=torch.float32
        ).to(self.device)[:, self.dim_reduction:]

        with torch.no_grad():
            residual_norms = self._residual_norm(features_centered.to(self.device))

        self.alpha = max_logits.mean().item() / residual_norms.cpu().mean().item()

    def score(self, x):
        features = self.get_penultimate_features(x)
        logits = self.get_logits(x)
        centered = features - self.feature_mean

        virtual_logit = self.alpha * self._residual_norm(centered)
        augmented_logits = torch.cat([logits, virtual_logit.unsqueeze(1)], dim=1)

        return torch.softmax(augmented_logits, dim=1)[:, -1]

    def _residual_norm(self, centered_features):
        proj = centered_features @ self.principal_space
        residual = centered_features - proj @ self.principal_space.T
        return torch.norm(residual, dim=1)

    def __repr__(self):
        return f"ViM(dim={self.dim_reduction}, alpha={self.alpha:.4f})"


class ResidualScorer(BaseFeatureScorer):
    def __init__(self, model, device='cuda', dim_reduction=512):
        super().__init__(model, device)
        self.dim_reduction = dim_reduction
        self.principal_space = None
        self.feature_mean = None

    def fit(self, train_loader, **kwargs):
        features_list = []
        with torch.no_grad():
            for images, _ in train_loader:
                features_list.append(
                    self.get_penultimate_features(images.to(self.device)).cpu()
                )
        features = torch.cat(features_list, dim=0)
        self.feature_mean = features.mean(dim=0).to(self.device)

        pca = PCA(n_components=self.dim_reduction)
        pca.fit((features - features.mean(dim=0)).numpy())
        self.principal_space = torch.tensor(
            pca.components_.T, dtype=torch.float32
        ).to(self.device)

    def score(self, x):
        features = self.get_penultimate_features(x)
        centered = features - self.feature_mean
        proj = centered @ self.principal_space
        residual = centered - proj @ self.principal_space.T
        return torch.norm(residual, dim=1)

    def __repr__(self):
        return f"ResidualScorer(dim={self.dim_reduction})"
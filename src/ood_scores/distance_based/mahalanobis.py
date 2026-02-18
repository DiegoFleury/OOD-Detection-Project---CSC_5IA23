import torch
from .base import BaseDistanceScorer


class MahalanobisScorer(BaseDistanceScorer):
    def __init__(self, model, device='cuda'):
        super().__init__(model, device)
        self.class_means = None
        self.precision = None

    def fit(self, train_loader, num_classes=100):
        features_list = []
        labels_list = []

        with torch.no_grad():
            for images, labels in train_loader:
                features_list.append(self.get_penultimate_features(images.to(self.device)).cpu())
                labels_list.append(labels)

        features = torch.cat(features_list, dim=0)
        labels = torch.cat(labels_list, dim=0)

        self.class_means = torch.stack([
            features[labels == c].mean(dim=0) for c in range(num_classes)
        ]).to(self.device)

        centered = torch.cat([
            features[labels == c] - self.class_means[c].cpu()
            for c in range(num_classes)
        ], dim=0)

        cov = (centered.T @ centered) / len(centered) + torch.eye(features.shape[1]) * 1e-4
        self.precision = torch.linalg.inv(cov).to(self.device)

    def score(self, x):
        features = self.get_penultimate_features(x)

        diffs = features.unsqueeze(1) - self.class_means.unsqueeze(0)  # [B, C, D]
        distances = torch.einsum('bcd,dd,bcd->bc', diffs, self.precision, diffs)  # [B, C]

        return distances.min(dim=1).values

    def __repr__(self):
        return "MahalanobisScorer"
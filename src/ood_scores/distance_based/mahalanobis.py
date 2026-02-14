"""
Mahalanobis distance-based OOD scorer
Lee et al. (2018) - A Simple Unified Framework for Detecting Out-of-Distribution Samples
"""

import torch
import numpy as np
from .base import BaseDistanceScorer


class MahalanobisScorer(BaseDistanceScorer):
    """
    Mahalanobis distance-based OOD detection
    
    Score: min_c (x - mu_c)^T Sigma^{-1} (x - mu_c)
    where mu_c is class c mean, Sigma is tied covariance
    
    Higher score = more OOD (farther from all classes)
    """
    
    def __init__(self, model, device='cuda'):
        super().__init__(model, device)
        self.precision = None  # Inverse covariance matrix
    
    def fit(self, train_loader, num_classes=100):
        """
        Compute class means and tied covariance from training data
        
        Args:
            train_loader: DataLoader with (images, labels)
            num_classes: number of classes
        """
        #print("Computing Mahalanobis statistics...")
        
        # Collect features and labels
        features_list = []
        labels_list = []
        
        with torch.no_grad():
            for batch in train_loader:
                images, labels = batch
                images = images.to(self.device)
                
                features = self.get_penultimate_features(images)
                features_list.append(features.cpu())
                labels_list.append(labels)
        
        features = torch.cat(features_list, dim=0)  # [N, D]
        labels = torch.cat(labels_list, dim=0)  # [N]
        
        # Compute class means
        class_means = []
        for c in range(num_classes):
            mask = (labels == c)
            if mask.sum() > 0:
                class_mean = features[mask].mean(dim=0)
                class_means.append(class_mean)
            else:
                class_means.append(torch.zeros(features.shape[1]))
        
        self.class_means = torch.stack(class_means).to(self.device)  # [C, D]
        
        # Compute tied covariance
        centered_features = []
        for c in range(num_classes):
            mask = (labels == c)
            if mask.sum() > 0:
                centered = features[mask] - self.class_means[c].cpu()
                centered_features.append(centered)
        
        centered_features = torch.cat(centered_features, dim=0)
        cov = (centered_features.T @ centered_features) / len(centered_features)
        
        # Add small regularization for numerical stability
        cov += torch.eye(cov.shape[0]) * 1e-4
        
        # Compute precision (inverse covariance)
        self.precision = torch.linalg.inv(cov).to(self.device)
        
        print(f"Computed statistics for {num_classes} classes")
        print(f"Feature dimension: {features.shape[1]}")
    
    def score(self, x):
        """
        Compute Mahalanobis distance OOD score
        
        Args:
            x: input batch [B, C, H, W]
        
        Returns:
            scores [B] - higher = more OOD
        """
        if self.class_means is None or self.precision is None:
            raise RuntimeError("Must call fit() before score()")
        
        features = self.get_penultimate_features(x)  # [B, D]
        
        # Compute Mahalanobis distance to each class
        min_distances = []
        
        for class_mean in self.class_means:
            diff = features - class_mean.unsqueeze(0)  # [B, D]
            # Mahalanobis distance: sqrt((x - mu)^T Sigma^{-1} (x - mu))
            dist = torch.sum(diff @ self.precision * diff, dim=1)
            min_distances.append(dist)
        
        min_distances = torch.stack(min_distances, dim=1)  # [B, C]
        
        scores = min_distances.min(dim=1)[0]
        
        return scores
    
    def __repr__(self):
        return "Mahalanobis"

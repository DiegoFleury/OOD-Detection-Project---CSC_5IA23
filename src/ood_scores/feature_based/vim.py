"""
Virtual-logit Matching (ViM) OOD scorer
Wang et al. (2022) - ViM: Out-Of-Distribution with Virtual-Logit Matching
"""

import torch
import numpy as np
from sklearn.decomposition import PCA
from .base import BaseFeatureScorer


class ViMScorer(BaseFeatureScorer):
    """
    ViM (Virtual-logit Matching)
    
    Uses PCA to find principal subspace and residual subspace
    OOD samples have larger residuals (outside principal subspace)
    """
    
    def __init__(self, model, device='cuda', dim_reduction=512):
        super().__init__(model, device)
        self.dim_reduction = dim_reduction
        self.principal_space = None  # Principal components
        self.feature_mean = None
    
    def fit(self, train_loader, num_classes=100):
        """
        Compute principal subspace from training features
        
        Args:
            train_loader: DataLoader with (images, labels)
            num_classes: number of classes
        """
        print("Computing ViM principal subspace...")
        
        # Collect all training features
        features_list = []
        
        with torch.no_grad():
            for batch in train_loader:
                images, _ = batch
                images = images.to(self.device)
                
                features = self.get_penultimate_features(images)
                features_list.append(features.cpu())
        
        features = torch.cat(features_list, dim=0)  # [N, D]
        
        # Compute mean
        self.feature_mean = features.mean(dim=0).to(self.device)

        # Apply PCA using sklearn
        pca = PCA(n_components=self.dim_reduction)
        pca.fit(features.numpy())

        # Store principal components (transposed to match our usage)
        self.principal_space = torch.tensor(pca.components_.T, dtype=torch.float32).to(self.device)  # [D, d]
        
        #print(f"Reduced from {features.shape[1]} to {self.dim_reduction} dimensions")
    
    def score(self, x):
        """
        Compute ViM OOD score
        
        Args:
            x: input batch [B, C, H, W]
        
        Returns:
            scores [B] - higher = more OOD
        """
        if self.principal_space is None:
            raise RuntimeError("Must call fit() before score()")
        
        features = self.get_penultimate_features(x)  # [B, D]
        centered = features - self.feature_mean
        
        # Project onto principal subspace
        proj = centered @ self.principal_space  # [B, d]
        reconstructed = proj @ self.principal_space.T  # [B, D]
        
        # Residual (orthogonal component)
        residual = centered - reconstructed
        
        # Score = norm of residual (larger = more OOD)
        scores = torch.norm(residual, dim=1)
        
        return scores
    
    def __repr__(self):
        return f"ViM(dim={self.dim_reduction})"

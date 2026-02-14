"""
Base class for feature-based OOD scorers
"""

import torch
from ..base import BaseOODScorer


class BaseFeatureScorer(BaseOODScorer):
    """
    Base class for feature-based OOD detection methods
    
    These methods analyze feature space properties (subspaces, collapse, etc.)
    """
    
    def __init__(self, model, device='cuda'):
        super().__init__(model, device)
    
    def fit(self, train_loader, num_classes=100):
        """
        Fit scorer to training data
        
        Args:
            train_loader: DataLoader with (images, labels)
            num_classes: number of classes
        """
        raise NotImplementedError("Subclasses must implement fit()")
    
    def get_penultimate_features(self, x):
        """Extract penultimate layer features"""
        with torch.no_grad():
            features = self.model.get_features(x)
        return features

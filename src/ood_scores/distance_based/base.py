"""
Base class for distance-based OOD scorers
"""

import torch
from ..base import BaseOODScorer


class BaseDistanceScorer(BaseOODScorer):
    """
    Base class for distance-based OOD detection methods
    
    These methods compute distances from class prototypes/distributions
    Require class statistics from training data
    """
    
    def __init__(self, model, device='cuda'):
        super().__init__(model, device)
        self.class_means = None
        self.class_covs = None
    
    def fit(self, train_loader, num_classes=100):
        """
        Compute class statistics from training data
        
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

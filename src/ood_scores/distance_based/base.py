import torch
from ..base import BaseOODScorer


class BaseDistanceScorer(BaseOODScorer):
    
    def __init__(self, model, device='cuda'):
        super().__init__(model, device)
        self.class_means = None
        self.class_covs = None
    
    def fit(self, train_loader, num_classes=100):
        pass
    
    def get_penultimate_features(self, x):
        with torch.no_grad():
            features = self.model.get_features(x)
        return features

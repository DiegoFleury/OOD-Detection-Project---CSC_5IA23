import torch
from ..base import BaseOODScorer


class BaseFeatureScorer(BaseOODScorer):
    def __init__(self, model, device='cuda'):
        super().__init__(model, device)

    def fit(self, train_loader, num_classes=100):
        raise NotImplementedError("Subclasses must implement fit()")

    def get_penultimate_features(self, x):
        with torch.no_grad():
            return self.model.get_features(x)

    def get_logits(self, x):
        with torch.no_grad():
            return self.model(x)
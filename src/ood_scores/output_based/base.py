import torch
from ..base import BaseOODScorer


class BaseOutputScorer(BaseOODScorer):
    def __init__(self, model, device='cuda'):
        super().__init__(model, device)
    
    def get_logits(self, x):
        logits = self.model(x)
        return logits

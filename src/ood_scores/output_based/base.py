"""
Base class for output-based OOD scorers
"""

import torch
from ..base import BaseOODScorer


class BaseOutputScorer(BaseOODScorer):
    """
    Base class for output-based OOD detection methods
    
    These methods only use the model's output (logits/softmax)
    """
    
    def __init__(self, model, device='cuda'):
        super().__init__(model, device)
    
    def get_logits(self, x):
        """Get model logits"""
        with torch.no_grad():
            logits = self.model(x)
        return logits

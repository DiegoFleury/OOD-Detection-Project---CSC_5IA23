"""
Maximum Logit OOD scorer
"""

import torch
from .base import BaseOutputScorer


class MaxLogitScorer(BaseOutputScorer):
    """
    Maximum Logit score
    
    Score: -max(logits)
    Higher score = more OOD (lower max logit)
    """
    
    def __init__(self, model, device='cuda'):
        super().__init__(model, device)
    
    def score(self, x):
        """
        Compute MaxLogit OOD score
        
        Args:
            x: input batch [B, C, H, W]
        
        Returns:
            scores [B] - higher = more OOD
        """
        logits = self.get_logits(x)
        max_logit = logits.max(dim=1)[0]
        
        # Return negative max logit
        return -max_logit
    
    def __repr__(self):
        return "MaxLogit"

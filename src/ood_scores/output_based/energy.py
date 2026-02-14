"""
Energy-based OOD scorer
Liu et al. (2020) - Energy-based Out-of-distribution Detection
"""

import torch
from .base import BaseOutputScorer


class EnergyScorer(BaseOutputScorer):
    """
    Energy score
    
    Score: -T * log(sum(exp(logits / T)))
    where T is temperature (default 1.0)
    
    Higher score = more OOD
    """
    
    def __init__(self, model, device='cuda', temperature=1.0):
        super().__init__(model, device)
        self.temperature = temperature
    
    def score(self, x):
        """
        Compute Energy OOD score
        
        Args:
            x: input batch [B, C, H, W]
        
        Returns:
            scores [B] - higher = more OOD
        """
        logits = self.get_logits(x)
        
        # Energy = -T * log(sum(exp(logits / T)))
        energy = -self.temperature * torch.logsumexp(logits / self.temperature, dim=1)
        
        return energy
    
    def __repr__(self):
        return f"Energy(T={self.temperature})"

"""
Maximum Softmax Probability (MSP) OOD scorer
Baseline method from Hendrycks & Gimpel (2017)
"""

import torch
import torch.nn.functional as F
from .base import BaseOutputScorer


class MSPScorer(BaseOutputScorer):
    """
    Maximum Softmax Probability (MSP)
    
    Score: -max(softmax(logits))
    Higher score = more OOD (lower confidence)
    """
    
    def __init__(self, model, device='cuda'):
        super().__init__(model, device)
    
    def score(self, x):
        """
        Compute MSP OOD score
        
        Args:
            x: input batch [B, C, H, W]
        
        Returns:
            scores [B] - higher = more OOD
        """
        logits = self.get_logits(x)
        probs = F.softmax(logits, dim=1)
        confidence = probs.max(dim=1)[0]
        
        # Return negative confidence (high confidence = low OOD score)
        return -confidence
    
    def __repr__(self):
        return "MSP"

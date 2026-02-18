import torch
from .base import BaseOutputScorer


class MaxLogitScorer(BaseOutputScorer):
    
    def __init__(self, model, device='cuda'):
        super().__init__(model, device)
    
    def score(self, x):

        logits = self.get_logits(x)
        max_logit = logits.max(dim=1)[0]
        
        # Return negative max logit
        return -max_logit
    
    def __repr__(self):
        return "MaxLogit"

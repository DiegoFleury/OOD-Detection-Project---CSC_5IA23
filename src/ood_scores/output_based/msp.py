import torch.nn.functional as F
from .base import BaseOutputScorer


class MSPScorer(BaseOutputScorer):
    
    def __init__(self, model, device='cuda'):
        super().__init__(model, device)
    
    def score(self, x):

        logits = self.get_logits(x)
        probs = F.softmax(logits, dim=1)
        confidence = probs.max(dim=1)[0]
        
        # Return negative confidence (high confidence = low OOD score)
        return -confidence
    
    def __repr__(self):
        return "MSP"

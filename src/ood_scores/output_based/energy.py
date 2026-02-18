import torch
from .base import BaseOutputScorer


class EnergyScorer(BaseOutputScorer):
    
    def __init__(self, model, device='cuda', temperature=1.0):
        super().__init__(model, device)
        self.temperature = temperature
    
    def score(self, x):

        logits = self.get_logits(x)
        
        # Energy = -T * log(sum(exp(logits / T)))
        energy = -self.temperature * torch.logsumexp(logits / self.temperature, dim=1)
        
        return energy
    
    def __repr__(self):
        return f"Energy(T={self.temperature})"

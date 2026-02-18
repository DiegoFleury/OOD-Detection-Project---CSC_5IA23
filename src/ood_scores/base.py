import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class BaseOODScorer(ABC):
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    @abstractmethod
    def score(self, x: torch.Tensor) -> torch.Tensor:
        pass
    
    def score_loader(self, loader) -> np.ndarray:

        scores = []
        
        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (tuple, list)):
                    x = batch[0]
                else:
                    x = batch
                
                x = x.to(self.device)
                batch_scores = self.score(x)
                scores.append(batch_scores.cpu().numpy())
        
        return np.concatenate(scores)
    
    def __repr__(self):
        return f"{self.__class__.__name__}()"

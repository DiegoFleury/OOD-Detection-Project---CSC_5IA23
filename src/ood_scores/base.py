"""
Base class for OOD scorers
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class BaseOODScorer(ABC):
    """
    Abstract base class for all OOD scoring methods
    
    OOD score convention: Higher score = more likely to be OOD
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        """
        Args:
            model: trained model
            device: 'cuda' or 'cpu'
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    @abstractmethod
    def score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute OOD score for input batch
        
        Args:
            x: input batch [B, C, H, W]
        
        Returns:
            scores [B] - higher = more OOD
        """
        pass
    
    def score_loader(self, loader) -> np.ndarray:
        """
        Compute OOD scores for entire dataloader
        
        Args:
            loader: PyTorch DataLoader
        
        Returns:
            scores array [N]
        """
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

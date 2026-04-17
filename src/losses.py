"""
Loss Functions: Custom loss functions with regularization techniques.

This module provides:
    - Label-smoothed cross-entropy loss
    - Focal loss for handling class imbalance
    - InfoNCE contrastive loss
    - R-Drop dual-forward regularization (implemented in train.py)
"""

from typing import Callable

import torch
import torch.nn.functional as F


def get_label_smoothed_cross_entropy(smoothing: float) -> Callable:
    """
    Create a label-smoothed cross-entropy loss function.
    
    Label smoothing prevents the model from becoming overconfident on the
    training set by distributing probability mass to incorrect labels.
    
    Args:
        smoothing: Smoothing factor (0 = no smoothing, 0.1 = typical value)
    
    Returns:
        Loss function that takes (outputs, labels) and returns a scalar loss
        
    Reference:
        https://arxiv.org/abs/1906.02629 (Rethinking the Inception Architecture)
    """
    def loss_fn(outputs, labels, **_):
        logits = outputs.logits
        n_classes = logits.size(-1)
        
        # Create smoothed target distribution
        with torch.no_grad():
            smooth_targets = torch.zeros_like(logits).fill_(
                smoothing / (n_classes - 1)
            )
            smooth_targets.scatter_(1, labels.unsqueeze(1), 1.0 - smoothing)
        
        # Compute cross-entropy with smoothed targets
        return -(
            smooth_targets * F.log_softmax(logits, dim=-1)
        ).sum(dim=-1).mean()
    
    return loss_fn


def get_focal_loss(
    alpha: float = 1.0,
    gamma: float = 2.0,
    smoothing: float = 0.0
) -> Callable:
    """
    Create a focal loss function for handling class imbalance.
    
    Focal loss down-weights easy examples and focuses on hard negatives.
    Particularly effective when positive and negative examples are imbalanced.
    
    Args:
        alpha: Weighting factor for positive examples (0 < alpha < 1)
        gamma: Focusing parameter (controls how much to focus on hard negatives)
        smoothing: Optional label smoothing factor
    
    Returns:
        Loss function that takes (outputs, labels) and returns a scalar loss
        
    Formula: FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    
    Reference:
        https://arxiv.org/abs/1708.02002 (Focal Loss for Dense Object Detection)
    """
    def focal_loss(outputs, labels, **_):
        logits = outputs.logits
        
        if smoothing > 0:
            # Apply label smoothing
            n_classes = logits.size(-1)
            smooth_targets = torch.zeros_like(logits).fill_(
                smoothing / (n_classes - 1)
            )
            smooth_targets.scatter_(1, labels.unsqueeze(1), 1.0 - smoothing)
            ce = -(
                smooth_targets * F.log_softmax(logits, dim=-1)
            ).sum(dim=-1)
        else:
            ce = F.cross_entropy(logits, labels, reduction="none")
        
        # Compute focal weight: (1 - p_t)^gamma
        pt = torch.exp(-ce)
        loss = alpha * (1 - pt) ** gamma * ce
        return loss.mean()
    
    return focal_loss


def get_infonce_loss(
    temperature: float = 0.07,
    weight: float = 0.5,
    smoothing: float = 0.0
) -> Callable:
    """
    Create an InfoNCE contrastive loss function.
    
    InfoNCE treats each example in a batch as having a positive pair (itself)
    and negative pairs (all other examples). This encourages the model to
    learn discriminative representations.
    
    Args:
        temperature: Temperature parameter for scaling logits
        weight: Weight of InfoNCE loss relative to other losses
        smoothing: Label smoothing factor (not typically used with InfoNCE)
    
    Returns:
        Loss function that takes (outputs, labels) and returns a weighted scalar loss
        
    Reference:
        https://arxiv.org/abs/1807.03748 (Representation Learning with Contrastive Predictive Coding)
    """
    def infonce_loss(outputs, labels, **_):
        # Extract hidden representations from last layer
        reps = outputs.hidden_states[-1]
        reps = F.normalize(reps, dim=-1)
        
        # Compute similarity matrix with temperature scaling
        sim_matrix = torch.mm(reps, reps.t()) / temperature
        
        # Target: each example should be similar to itself
        target = torch.arange(reps.size(0), device=reps.device)
        
        # Cross-entropy loss
        loss = F.cross_entropy(sim_matrix, target)
        return weight * loss
    
    return infonce_loss

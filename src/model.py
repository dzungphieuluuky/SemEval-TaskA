"""
Model Layer: Loading and configuring pre-trained models.

This module handles:
    - Loading pretrained models from HuggingFace
    - Loading and configuring tokenizers
    - Freezing base model weights for transfer learning
"""

import logging
from typing import Tuple

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, PreTrainedTokenizer


logger = logging.getLogger(__name__)


def load_model_and_tokenizer(
    model_name: str,
    num_labels: int = 2,
    device: torch.device = None,
    logger: logging.Logger = None
) -> Tuple[nn.Module, PreTrainedTokenizer]:
    """
    Load pretrained model and tokenizer from HuggingFace.
    
    Args:
        model_name: HuggingFace model identifier (e.g., "microsoft/graphcodebert-base")
        num_labels: Number of classification labels
        device: PyTorch device to move model to
        logger: Logger instance for logging messages
    
    Returns:
        Tuple[nn.Module, PreTrainedTokenizer]: (model, tokenizer)
        
    Note:
        - Model is configured for single-label sequence classification
        - Dropout probabilities are set to 0.2 for regularization
        - Model is moved to device if provided
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Loading tokenizer from: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    logger.info(f"Loading model from: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="single_label_classification",
        hidden_dropout_prob=0.2,
        attention_probs_dropout_prob=0.2,
    )
    
    if device:
        model.to(device)
        logger.info(f"Model placed on {device}")
    
    logger.info("Model loaded successfully")
    logger.info(
        f"Model config - num_labels: {model.config.num_labels}, "
        f"hidden_size: {model.config.hidden_size}"
    )
    
    return model, tokenizer


def freeze_base_model(
    model: nn.Module,
    logger: logging.Logger = None
) -> None:
    """
    Freeze all base model parameters, keeping only classifier head trainable.
    
    For sequence classification models (e.g., CodeBERT based on RoBERTa),
    this freezes the encoder layers and allows only the classification head
    to be updated during training.
    
    Args:
        model: PyTorch model to freeze
        logger: Logger instance for logging messages
        
    Note:
        This is useful for transfer learning when you want to fine-tune
        only the task-specific head on top of the frozen pretrained encoder.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Freeze all parameters except classifier
    for name, param in model.named_parameters():
        if "classifier" not in name and "cls" not in name.lower():
            param.requires_grad = False
    
    # Ensure classifier is trainable
    for name, param in model.named_parameters():
        if "classifier" in name or "cls" in name.lower():
            param.requires_grad = True
    
    logger.info("Base model frozen - only classifier head is trainable")
    
    # Log trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Trainable parameters: {trainable_params:,} / {total_params:,} "
        f"({100 * trainable_params / total_params:.1f}%)"
    )

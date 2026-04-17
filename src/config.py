"""
Configuration classes and constants for Task A training pipeline.

Defines all configuration parameters as a single dataclass following
the orchestration pattern.
"""

import os
import torch
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainConfig:
    """
    Configuration for the training pipeline.
    
    Attributes:
        model_name: HuggingFace model identifier
        output_dir: Directory to save checkpoints and logs
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Initial learning rate
        max_length: Maximum token sequence length
        num_labels: Number of classification labels (2 for binary)
        use_wandb: Enable Weights & Biases logging
        freeze_base: Freeze base model parameters
        loss_type: Type of loss function ("ce", "focal", "infonce", "r-drop")
        focal_alpha: Focal loss alpha parameter
        focal_gamma: Focal loss gamma parameter
        r_drop_alpha: R-Drop KL divergence weight
        infonce_temperature: InfoNCE temperature parameter
        infonce_weight: Weight for InfoNCE loss
        seed: Random seed for reproducibility
        resume_from_checkpoint: Path to checkpoint for resuming training
        label_smoothing: Label smoothing factor (0 = disabled)
        adversarial_epsilon: FGM adversarial perturbation magnitude (0 = disabled)
        use_swa: Enable Stochastic Weight Averaging
        swa_start_epoch: Epoch to start SWA
        swa_lr: Learning rate for SWA
        data_augmentation: Enable code augmentation
        aug_rename_prob: Probability to rename identifiers in augmentation
        aug_format_prob: Probability to modify formatting in augmentation
        device: PyTorch device (auto-detected)
    """
    
    model_name: str = "microsoft/graphcodebert-base"
    output_dir: str = "./output_checkpoints/graphcodebert-robust"
    num_epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5
    max_length: int = 512
    num_labels: int = 2
    use_wandb: bool = False
    freeze_base: bool = False
    loss_type: str = "r-drop"
    focal_alpha: float = 1.0
    focal_gamma: float = 2.0
    r_drop_alpha: float = 4.0
    infonce_temperature: float = 0.07
    infonce_weight: float = 0.5
    seed: int = 42
    resume_from_checkpoint: Optional[str] = None
    
    # Regularization parameters
    label_smoothing: float = 0.1
    adversarial_epsilon: float = 0.5
    use_swa: bool = True
    swa_start_epoch: int = 2
    swa_lr: float = 1e-5
    
    # Augmentation parameters
    data_augmentation: bool = True
    aug_rename_prob: float = 0.3
    aug_format_prob: float = 0.3
    
    # Internally computed
    device: torch.device = field(
        init=False,
        default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )


# Constants for dataset and task
TASK_ID = "A"
TRAIN_DATASET_NAME = "DaniilOr/SemEval-2026-Task13"
INFERENCE_DATASET_NAME = "dzungpham/SemEval-2026-TaskA-dataset"

# Label mapping
LABEL_TO_ID = {"human": 0, "machine": 1}
ID_TO_LABEL = {0: "human", 1: "machine"}

# Number of classes
NUM_CLASSES = 2

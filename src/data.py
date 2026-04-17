"""
Data Layer: Loading, preprocessing, and tokenizing datasets.

This module handles:
    - Code preprocessing and normalization
    - Dataset loading from HuggingFace Hub
    - Tokenization with truncation and padding
    - Class balancing via upsampling
    - Evaluation metrics computation
"""

import os
import re
import random
import logging
import numpy as np
from typing import Dict, Tuple, Optional, Callable

from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import PreTrainedTokenizer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from .config import TASK_ID, TRAIN_DATASET_NAME


logger = logging.getLogger(__name__)


def preprocess_code(code_str: str) -> str:
    """
    Normalize code string with semantic-preserving perturbations.
    
    Transformations:
        - Remove BOM and zero-width characters
        - Normalize line endings
        - Randomly adjust indentation (spaces ↔ tabs)
        - Strip trailing whitespace
        - Collapse excessive blank lines
        - Collapse excessive whitespace
    
    Args:
        code_str: Raw code string
    
    Returns:
        Preprocessed code string
    """
    code_str = code_str.lstrip("\ufeff\u200b\u200c\u200d")
    code_str = re.sub(r"\r\n|\r", "\n", code_str)
    
    # 20% chance to replace 4-space indent with tabs
    if random.random() < 0.2:
        code_str = code_str.replace("    ", "\t")
    # 20% chance to replace tabs with 2-space indent
    elif random.random() < 0.2:
        code_str = code_str.replace("    ", "  ")
    
    code_str = "\n".join(line.rstrip() for line in code_str.split("\n"))
    code_str = re.sub(r"\n{3,}", "\n\n", code_str)
    code_str = re.sub(r"[ \t]+", " ", code_str)
    return code_str.strip()


def tokenize_function(
    examples: Dict,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512
) -> Dict:
    """
    Tokenize code examples.
    
    Args:
        examples: Dictionary with "code" key containing list of code strings
        tokenizer: PreTrainedTokenizer instance
        max_length: Maximum sequence length
    
    Returns:
        Dictionary with "input_ids", "attention_mask", and "token_type_ids"
    """
    codes = [preprocess_code(c) for c in examples["code"]]
    return tokenizer(
        codes,
        truncation=True,
        max_length=max_length,
        padding=False
    )


def upsample_dataset(dataset: Dataset) -> Dataset:
    """
    Upsample the minority class to balance the dataset.
    
    Args:
        dataset: Dataset with "label" column
    
    Returns:
        Balanced dataset with upsampled minority class
    """
    labels = np.array(dataset["label"])
    class_counts = np.bincount(labels)
    majority_class = np.argmax(class_counts)
    minority_class = np.argmin(class_counts)
    
    logger.info(
        f"Initial distribution: Class 0: {class_counts[0]:,}, "
        f"Class 1: {class_counts[1]:,}"
    )
    
    diff = class_counts[majority_class] - class_counts[minority_class]
    
    if diff > 0:
        minority_indices = np.where(labels == minority_class)[0]
        upsample_indices = np.random.choice(
            minority_indices,
            size=diff,
            replace=True
        )
        upsampled_data = dataset.select(upsample_indices)
        dataset = concatenate_datasets([dataset, upsampled_data])
        
        new_counts = np.bincount(np.array(dataset["label"]))
        logger.info(
            f"Balanced distribution: Class 0: {new_counts[0]:,}, "
            f"Class 1: {new_counts[1]:,}"
        )
    
    return dataset.shuffle(seed=42)


def load_datasets(
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
    aug: Optional[Callable] = None
) -> Tuple[Dataset, Dataset]:
    """
    Load, preprocess, and tokenize train and validation datasets.
    
    Args:
        tokenizer: PreTrainedTokenizer instance
        max_length: Maximum token sequence length
        aug: Optional augmentation function to apply before tokenization
    
    Returns:
        Tuple[Dataset, Dataset]: (train_dataset, val_dataset) ready for training
        
    Note:
        - Removes "code" and "language" columns after tokenization
        - Renames "label" to "labels" for HuggingFace Trainer compatibility
        - Prints dataset schemas before and after processing
    """
    from .utils import print_dataset_schema
    
    logger.info("Loading datasets from Hugging Face Hub...")
    
    train_dataset = load_dataset(TRAIN_DATASET_NAME, TASK_ID, split="train")
    val_dataset = load_dataset(TRAIN_DATASET_NAME, TASK_ID, split="validation")
    
    # Print schemas before processing
    print_dataset_schema(train_dataset, "Train Dataset (Before Processing)")
    print_dataset_schema(val_dataset, "Validation Dataset (Before Processing)")
    
    logger.info(f"Train samples: {len(train_dataset):,}")
    logger.info(f"Val samples: {len(val_dataset):,}")
    
    # Apply augmentation if provided
    if aug is not None:
        logger.info("Applying data augmentation...")
        def augment_batch(examples):
            examples["code"] = [aug(c) for c in examples["code"]]
            return examples
        
        train_dataset = train_dataset.map(
            augment_batch,
            batched=True,
            batch_size=512,
            desc="Augmenting train",
            num_proc=os.cpu_count(),
        )
    
    # Tokenize
    logger.info("Tokenizing datasets...")
    train_dataset = train_dataset.map(
        lambda ex: tokenize_function(ex, tokenizer, max_length),
        batched=True,
        batch_size=512,
        remove_columns=[
            col for col in ["code", "language"]
            if col in train_dataset.column_names
        ],
        desc="Tokenizing train",
        num_proc=os.cpu_count(),
    )
    
    val_dataset = val_dataset.map(
        lambda ex: tokenize_function(ex, tokenizer, max_length),
        batched=True,
        batch_size=512,
        remove_columns=[
            col for col in ["code", "language"]
            if col in val_dataset.column_names
        ],
        desc="Tokenizing val",
        num_proc=os.cpu_count(),
    )
    
    # Rename label column for HuggingFace Trainer
    train_dataset = train_dataset.rename_column("label", "labels")
    val_dataset = val_dataset.rename_column("label", "labels")
    
    # Print schemas after processing
    print_dataset_schema(train_dataset, "Train Dataset (After Tokenization)")
    print_dataset_schema(val_dataset, "Validation Dataset (After Tokenization)")
    
    return train_dataset, val_dataset


def compute_metrics(eval_pred) -> Dict[str, float]:
    """
    Compute evaluation metrics: accuracy, precision, recall, macro F1.
    
    Optimizes for macro F1-score as specified in the task requirements.
    
    Args:
        eval_pred: Tuple of (predictions, labels) from model.predict()
    
    Returns:
        Dictionary with "accuracy", "precision", "recall", "macro_f1"
    """
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="macro",
        zero_division=0
    )
    acc = accuracy_score(labels, predictions)
    
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "macro_f1": f1
    }

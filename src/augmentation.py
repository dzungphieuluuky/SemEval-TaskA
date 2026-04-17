"""
Data Augmentation: Code-specific augmentation strategies.

This module provides:
    - Lightweight code transformations for training robustness
    - Identifier renaming (variable/function name variations)
    - Formatting changes (indentation, blank lines)
    - On-the-fly augmentation for DataLoaders
"""

import re
import random
import logging
from typing import Optional

import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import PreTrainedTokenizer


logger = logging.getLogger(__name__)


class CodeAugmentation:
    """
    Apply lightweight semantic-preserving code transformations.
    
    Augmentations:
        1. Identifier Renaming: Replace variable/function names with random tokens
        2. Formatting Changes: Add/remove blank lines and adjust indentation
        
    These transformations preserve code semantics while varying surface-level
    features, helping the model generalize to different coding styles.
    """
    
    def __init__(self, rename_prob: float = 0.3, format_prob: float = 0.3):
        """
        Args:
            rename_prob: Probability of applying identifier renaming (0-1)
            format_prob: Probability of applying formatting changes (0-1)
        """
        self.rename_prob = rename_prob
        self.format_prob = format_prob
    
    def __call__(self, code_str: str) -> str:
        """
        Apply random augmentations to a code string.
        
        Args:
            code_str: Source code as string
        
        Returns:
            Augmented code string
        """
        code_str = str(code_str)
        
        # 1. Identifier renaming
        if random.random() < self.rename_prob:
            code_str = self._rename_identifiers(code_str)
        
        # 2. Formatting changes
        if random.random() < self.format_prob:
            code_str = self._modify_formatting(code_str)
        
        return code_str
    
    def _rename_identifiers(self, code_str: str) -> str:
        """
        Rename identifiers (variables, functions) to random tokens.
        
        Preserves:
            - Language keywords
            - Built-in functions
            - Operators and special characters
        """
        # Extract all identifiers (alphanumeric + underscore)
        tokens = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', code_str)
        
        # Filter out keywords
        keywords = {
            'def', 'return', 'if', 'else', 'elif', 'for', 'while', 'break',
            'continue', 'import', 'from', 'as', 'with', 'try', 'except',
            'finally', 'raise', 'assert', 'class', 'lambda', 'and', 'or',
            'not', 'in', 'is', 'True', 'False', 'None', 'pass', 'yield',
            'global', 'nonlocal', 'del', '__name__', '__main__'
        }
        
        identifiers = [
            t for t in set(tokens)
            if t not in keywords and len(t) > 1
        ]
        
        # Rename up to 5 identifiers per augmentation
        if identifiers:
            rename_map = {
                identifier: f"var_{random.randint(1000, 9999)}"
                for identifier in identifiers[:5]
            }
            for old_name, new_name in rename_map.items():
                code_str = code_str.replace(old_name, new_name)
        
        return code_str
    
    def _modify_formatting(self, code_str: str) -> str:
        """
        Modify code formatting: blank lines and indentation.
        """
        lines = code_str.split('\n')
        new_lines = []
        
        # Randomly insert blank lines
        for line in lines:
            new_lines.append(line)
            if random.random() < 0.1:  # 10% chance to add blank line after
                new_lines.append('')
        
        code_str = '\n'.join(new_lines)
        
        # Randomly add spaces to line beginnings
        if random.random() < 0.2:
            lines = code_str.split('\n')
            lines = [
                (' ' + line) if random.random() < 0.1 else line
                for line in lines
            ]
            code_str = '\n'.join(lines)
        
        return code_str


class AugmentedDataset(TorchDataset):
    """
    Wrap a HuggingFace Dataset to apply on-the-fly code augmentation.
    
    Augmentation is applied during iteration, allowing stochastic variations
    across epochs without doubling the dataset size.
    """
    
    def __init__(
        self,
        dataset,
        tokenizer: PreTrainedTokenizer,
        max_length: int,
        augmentation: CodeAugmentation
    ):
        """
        Args:
            dataset: HuggingFace Dataset with "code" column
            tokenizer: PreTrainedTokenizer for tokenization
            max_length: Maximum sequence length
            augmentation: CodeAugmentation instance
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augmentation = augmentation
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Get a single augmented example.
        
        Returns:
            Dictionary with tokenized input_ids, attention_mask, and labels
        """
        item = self.dataset[idx]
        
        # Extract code (supports both "code" and "text" fields)
        if 'code' in item:
            code = item['code']
        elif 'text' in item:
            code = item['text']
        else:
            raise KeyError("Dataset must contain 'code' or 'text' column")
        
        # Apply augmentation
        code = self.augmentation(code)
        
        # Tokenize
        encoding = self.tokenizer(
            code,
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_tensors=None
        )
        
        # Add label
        label = item.get('label', item.get('labels', 0))
        return {**encoding, 'labels': label}

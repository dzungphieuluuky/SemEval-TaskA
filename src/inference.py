"""
Inference Pipeline: Standalone inference with metrics computation.

This module provides:
    - Model loading for inference
    - Dataset loading with fallback for schema mismatches
    - Batch prediction with optional metrics
    - Results export to CSV
"""

import os
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm.auto import tqdm
from datasets import load_dataset, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)

from .config import INFERENCE_DATASET_NAME
from .utils import setup_logger, log_model_architecture, print_dataset_schema


logger = logging.getLogger(__name__)


def run_standalone_inference(
    checkpoint_path: str,
    output_dir: str = "./",
    output_csv: str = "submission.csv",
    batch_size: int = 32,
    max_length: int = 512,
    dataset_name: str = INFERENCE_DATASET_NAME,
    split: str = "test",
    calculate_metrics: bool = False,
) -> None:
    """
    Run standalone inference on a test dataset.
    
    Pipeline:
        1. Load model and tokenizer from checkpoint
        2. Load test dataset (with fallback for schema mismatches)
        3. Tokenize and create DataLoader
        4. Run inference on all batches
        5. (Optional) Calculate evaluation metrics
        6. Save predictions to CSV
    
    Args:
        checkpoint_path: Path to saved model checkpoint
        output_dir: Directory to save outputs
        output_csv: Filename for predictions CSV
        batch_size: Inference batch size
        max_length: Maximum token sequence length
        dataset_name: HuggingFace dataset identifier
        split: Dataset split to use ("test", "validation", etc.)
        calculate_metrics: Compute metrics if labels are available
        
    Note:
        Handles schema mismatches in inference datasets by trying
        multiple loading strategies.
    """
    # Setup
    inference_logger = setup_logger("inference", output_dir, "inference.log")
    inference_logger.info(f"Loading model from: {checkpoint_path}")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
    log_model_architecture(model, tokenizer, inference_logger)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Load dataset with fallback
    inference_logger.info(f"Loading dataset: {dataset_name} (split={split})")
    test_ds = None
    
    try:
        # First attempt: standard loading
        test_ds = load_dataset(dataset_name, split=split)
        inference_logger.info("✅ Loaded with default config")
    except Exception as e:
        inference_logger.warning(f"Default loading failed: {e}")
        inference_logger.info("Retrying with data_files fallback...")
        
        try:
            # Second attempt: use data_files pattern matching
            test_ds = load_dataset(
                dataset_name,
                data_files={split: f"*{split}*"},
                split=split
            )
            inference_logger.info("✅ Loaded with data_files fallback")
        except Exception as e2:
            inference_logger.error(f"Failed to load dataset: {e2}")
            raise
    
    # Print schema for debugging
    print_dataset_schema(test_ds, f"Test Dataset (Raw)")
    
    # Tokenization
    def tokenize_fn(examples):
        return tokenizer(
            examples["code"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
    
    inference_logger.info("Tokenizing dataset...")
    
    # Safely remove non-essential columns
    cols_to_remove = [
        c for c in test_ds.column_names
        if c not in ["input_ids", "attention_mask"]
        and c not in ["id", "ID", "label"]
    ]
    inference_logger.info(f"Removing columns: {cols_to_remove}")
    
    tokenized_ds = test_ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=cols_to_remove,
        desc="Tokenizing"
    )
    
    # Print schema after tokenization
    print_dataset_schema(tokenized_ds, "Test Dataset (After Tokenization)")
    
    tokenized_ds.set_format(
        type="torch",
        columns=["input_ids", "attention_mask"]
    )
    
    # DataLoader
    test_loader = torch.utils.data.DataLoader(
        tokenized_ds,
        batch_size=batch_size,
        shuffle=False,
    )
    
    # Inference
    inference_logger.info(f"Running inference on {len(test_ds):,} examples...")
    all_logits = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            inputs = {
                k: v.to(device)
                for k, v in batch.items()
                if k in ["input_ids", "attention_mask"]
            }
            outputs = model(**inputs)
            all_logits.append(outputs.logits.cpu())
    
    # Combine logits and get predictions
    logits = torch.cat(all_logits, dim=0).numpy()
    pred_labels = logits.argmax(axis=-1)
    
    # Optional: Calculate metrics
    if calculate_metrics and "label" in test_ds.column_names:
        inference_logger.info("Calculating classification metrics...")
        true_labels = np.array(test_ds["label"])
        
        acc = accuracy_score(true_labels, pred_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels,
            pred_labels,
            average='macro'
        )
        
        inference_logger.info("-" * 50)
        inference_logger.info(f"EVALUATION METRICS (split={split})")
        inference_logger.info("-" * 50)
        inference_logger.info(f"Accuracy:  {acc:.4f}")
        inference_logger.info(f"Precision: {precision:.4f}")
        inference_logger.info(f"Recall:    {recall:.4f}")
        inference_logger.info(f"Macro F1:  {f1:.4f}")
        inference_logger.info("-" * 50)
        
        cm = confusion_matrix(true_labels, pred_labels)
        inference_logger.info(f"Confusion Matrix:\n{cm}")
    else:
        inference_logger.warning(
            "No labels found in dataset. Skipping metric calculation."
        )
    
    # Save predictions
    os.makedirs(output_dir, exist_ok=True)
    csv_path = Path(output_dir) / output_csv
    
    # Get IDs (try different column names)
    if "id" in test_ds.column_names:
        ids = test_ds["id"]
    elif "ID" in test_ds.column_names:
        ids = test_ds["ID"]
    else:
        ids = list(range(len(pred_labels)))
    
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("id,label\n")
        for idx, label in zip(ids, pred_labels):
            f.write(f"{idx},{label}\n")
    
    inference_logger.info(f"✅ Predictions saved to {csv_path}")
    
    return test_ds, pred_labels, logits

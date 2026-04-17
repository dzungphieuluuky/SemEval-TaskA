"""
Utility functions for environment setup, logging, and checkpoint management.

This module handles:
    - Environment path configuration (Colab, Kaggle, Local)
    - Secret/API key loading from different environments
    - System information reporting
    - Logging setup for training and inference
    - Model architecture logging
    - Checkpoint downloads from HuggingFace Hub
    - Dataset schema printing for debugging
"""

import os
import sys
import logging
from pathlib import Path
from typing import Tuple, Optional

import torch
from datasets import Dataset


def configure_environment_paths() -> Tuple[str, str, str]:
    """
    Detect execution environment and configure paths accordingly.
    
    Supports: Google Colab, Kaggle Notebooks, and Local/HPC environments.
    
    Returns:
        Tuple[str, str, str]: (input_path, output_path, environment_name)
    """
    try:
        from IPython import get_ipython
        ipy = get_ipython()
        
        if "google.colab" in str(ipy):
            print("✅ Environment: Google Colab")
            base_data_path = "/content/"
            base_output_path = "/content/"
            environment_name = "colab"
        elif os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
            print("✅ Environment: Kaggle")
            base_data_path = "/kaggle/input/"
            base_output_path = "/kaggle/working/"
            environment_name = "kaggle"
        else:
            print("⚠️  Environment: Local/Unknown")
            base_data_path = "./data/"
            base_output_path = "./output/"
            environment_name = "local"
    except NameError:
        print("⚠️  Non-interactive session. Using local paths.")
        base_data_path = "./data/"
        base_output_path = "./output/"
        environment_name = "local"
    
    os.makedirs(base_output_path, exist_ok=True)
    print(f"📂 Data Path: {base_data_path}")
    print(f"📦 Output Path: {base_output_path}")
    
    return base_data_path, base_output_path, environment_name


def load_secret(key_name: str, env_name: str) -> Optional[str]:
    """
    Load API keys and secrets from environment-specific sources.
    
    Args:
        key_name: Name of the secret to load
        env_name: Environment name ("colab", "kaggle", or "local")
    
    Returns:
        The secret value if found, None otherwise
    """
    secret_value = None
    print(f"Attempting to load secret '{key_name}' from '{env_name}' environment...")
    
    try:
        if env_name == "colab":
            from google.colab import userdata
            secret_value = userdata.get(key_name)
        elif env_name == "kaggle":
            from kaggle_secrets import UserSecretsClient
            user_secrets = UserSecretsClient()
            secret_value = user_secrets.get_secret(key_name)
        else:
            secret_value = os.getenv(key_name)
        
        if not secret_value:
            print(f"⚠️  Secret '{key_name}' not found in {env_name} environment.")
            return None
        
        print(f"✅ Successfully loaded secret '{key_name}'.")
        return secret_value
    except Exception as e:
        print(f"❌ Error loading secret '{key_name}': {e}")
        return None


def print_system_info() -> None:
    """Print system information (Python, PyTorch, CUDA)."""
    print("\n🔧 System Information")
    print(f"Python version: {sys.version.split()[0]}")
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("CUDA not available (using CPU)")
    except ImportError:
        print("PyTorch not installed")


def setup_logger(
    logger_name: str,
    output_dir: str,
    log_filename: str = "training.log"
) -> logging.Logger:
    """
    Create and configure a logger with both console and file handlers.
    
    Args:
        logger_name: Name for the logger
        output_dir: Directory to save log files
        log_filename: Name of the log file
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(ch)
    
    # File handler
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, log_filename)
    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    )
    logger.addHandler(fh)
    
    logger.info(f"Logging to {log_path}")
    return logger


def log_model_architecture(
    model: torch.nn.Module,
    tokenizer,
    logger: logging.Logger
) -> None:
    """
    Log model architecture, parameter counts, and tokenizer information.
    
    Args:
        model: PyTorch model to log
        tokenizer: HuggingFace tokenizer instance
        logger: Logger instance
    """
    logger.info("===== Model Architecture =====")
    logger.info("\n" + model.__repr__())
    
    # Parameter counting
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total_params = trainable_params + non_trainable_params
    
    logger.info("===== Parameter Summary =====")
    logger.info(f"Total Parameters:         {total_params:,}")
    logger.info(f"Trainable Parameters:     {trainable_params:,}")
    logger.info(f"Non-trainable Parameters: {non_trainable_params:,}")
    
    # Tokenizer info
    logger.info("===== Tokenizer Summary =====")
    logger.info(
        f"Vocab size: {len(tokenizer)} | "
        f"Special tokens: {tokenizer.all_special_tokens}"
    )
    logger.info("===== End of Architecture Log =====")


def print_dataset_schema(dataset: Dataset, name: str) -> None:
    """
    Print dataset schema, size, and sample for debugging.
    
    Useful for verifying column names and data types match expectations.
    
    Args:
        dataset: HuggingFace Dataset instance
        name: Human-readable name for the dataset
    """
    print(f"\n{'='*70}")
    print(f"📊 DATASET SCHEMA: {name}")
    print(f"{'='*70}")
    print(f"Columns:     {dataset.column_names}")
    print(f"Num rows:    {len(dataset):,}")
    print(f"Features:    {dataset.features}")
    print(f"\n📝 Sample row (index 0):")
    sample = dataset[0]
    for key, value in sample.items():
        if isinstance(value, str):
            value_preview = value[:100] + "..." if len(value) > 100 else value
        else:
            value_preview = value
        print(f"  {key}: {value_preview}")
    print(f"{'='*70}\n")


def download_from_hf(
    repo_id: str,
    local_dir: str = "checkpoints",
    allow_patterns: Optional[list] = None,
    force_download: bool = False,
    repo_type: str = "model"
) -> None:
    """
    Download model checkpoint from HuggingFace Hub.
    
    Args:
        repo_id: HuggingFace repository ID (e.g., "dzungpham/model-name")
        local_dir: Local directory to save files
        allow_patterns: List of file patterns to download (e.g., ["*.safetensors"])
        force_download: Force re-download even if files exist
        repo_type: Repository type ("model" or "dataset")
    """
    if allow_patterns is None:
        allow_patterns = ["*.safetensors", "*.json"]
    
    print(f"📥 Downloading from {repo_id} to '{local_dir}'...\n")
    
    try:
        from huggingface_hub import snapshot_download
        
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            repo_type=repo_type,
            allow_patterns=allow_patterns,
            force_download=force_download,
        )
        
        print("\n✅ Download complete!")
        print(f"\n📂 Files in {local_dir}/:")
        for file in os.listdir(local_dir):
            file_path = os.path.join(local_dir, file)
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path) / (1024**2)
                print(f"  ✓ {file} ({size:.2f} MB)")
    except Exception as e:
        print(f"❌ Download failed: {e}")
        raise

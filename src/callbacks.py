"""
Custom callbacks for training pipeline.

Provides callbacks for logging configurations, hyperparameters, and metrics
to checkpoint directories during training.
"""

import os
import json
import logging
from typing import Dict, Any

from transformers import TrainerCallback, TrainerState, TrainerControl
from transformers.integrations import TensorBoardCallback

from .config import TrainConfig


logger = logging.getLogger(__name__)


class ConfigLoggingCallback(TrainerCallback):
    """
    Callback to save hyperparameters and configuration to each checkpoint.
    
    At each save_steps interval, writes a comprehensive config JSON file to
    the checkpoint directory containing:
        - All TrainConfig parameters
        - All TrainingArguments parameters
        - Training state (epoch, step, best_metric)
    """
    
    def __init__(self, train_config: TrainConfig):
        """
        Args:
            train_config: TrainConfig instance with all hyperparameters
        """
        self.train_config = train_config
    
    def on_save(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """
        Called after a checkpoint is saved.
        
        Logs all configs to config.json in the checkpoint directory.
        """
        checkpoint_dir = os.path.join(
            args.output_dir,
            f"checkpoint-{state.global_step}"
        )
        
        if not os.path.exists(checkpoint_dir):
            # Fallback in case directory structure is different
            logger.warning(
                f"Expected checkpoint directory not found: {checkpoint_dir}. "
                "Config logging may be incomplete."
            )
            return
        
        # Prepare config dictionary
        config_dict = self._prepare_config_dict(args, state)
        
        # Save to config.json
        config_path = os.path.join(checkpoint_dir, "config_hyperparams.json")
        try:
            with open(config_path, "w") as f:
                json.dump(config_dict, f, indent=2, default=str)
            logger.info(f"Saved hyperparameters to {config_path}")
        except Exception as e:
            logger.error(f"Failed to save config to {config_path}: {e}")
    
    def _prepare_config_dict(
        self,
        training_args,
        state: TrainerState
    ) -> Dict[str, Any]:
        """
        Prepare a comprehensive config dictionary with all parameters.
        
        Returns:
            Dictionary containing:
                - train_config: All TrainConfig parameters
                - training_arguments: Subset of TrainingArguments
                - training_state: Current training progress
        """
        # Extract TrainConfig as dictionary
        train_cfg_dict = {
            "model_name": self.train_config.model_name,
            "num_epochs": self.train_config.num_epochs,
            "batch_size": self.train_config.batch_size,
            "learning_rate": self.train_config.learning_rate,
            "max_length": self.train_config.max_length,
            "num_labels": self.train_config.num_labels,
            "loss_type": self.train_config.loss_type,
            "focal_alpha": self.train_config.focal_alpha,
            "focal_gamma": self.train_config.focal_gamma,
            "r_drop_alpha": self.train_config.r_drop_alpha,
            "infonce_temperature": self.train_config.infonce_temperature,
            "infonce_weight": self.train_config.infonce_weight,
            "label_smoothing": self.train_config.label_smoothing,
            "adversarial_epsilon": self.train_config.adversarial_epsilon,
            "use_swa": self.train_config.use_swa,
            "swa_start_epoch": self.train_config.swa_start_epoch,
            "swa_lr": self.train_config.swa_lr,
            "data_augmentation": self.train_config.data_augmentation,
            "aug_rename_prob": self.train_config.aug_rename_prob,
            "aug_format_prob": self.train_config.aug_format_prob,
            "freeze_base": self.train_config.freeze_base,
            "seed": self.train_config.seed,
            "use_wandb": self.train_config.use_wandb,
        }
        
        # Extract key TrainingArguments
        training_args_dict = {
            "output_dir": training_args.output_dir,
            "num_train_epochs": training_args.num_train_epochs,
            "per_device_train_batch_size": training_args.per_device_train_batch_size,
            "per_device_eval_batch_size": training_args.per_device_eval_batch_size,
            "learning_rate": training_args.learning_rate,
            "warmup_steps": training_args.warmup_steps,
            "weight_decay": training_args.weight_decay,
            "logging_steps": training_args.logging_steps,
            "eval_steps": training_args.eval_steps,
            "save_steps": training_args.save_steps,
            "metric_for_best_model": training_args.metric_for_best_model,
            "greater_is_better": training_args.greater_is_better,
            "save_total_limit": training_args.save_total_limit,
            "fp16": training_args.fp16,
            "seed": training_args.seed,
        }
        
        # Extract training state
        training_state_dict = {
            "global_step": state.global_step,
            "epoch": state.epoch,
            "total_flos": state.total_flos,
            "best_metric": state.best_metric,
            "best_model_checkpoint": state.best_model_checkpoint,
        }
        
        # Combine all sections
        config_dict = {
            "train_config": train_cfg_dict,
            "training_arguments": training_args_dict,
            "training_state": training_state_dict,
        }
        
        return config_dict

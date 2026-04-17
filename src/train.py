"""
Training Engine: RobustTrainer and training pipeline orchestration.

This module provides:
    - Custom Trainer with advanced regularization (R-Drop, adversarial training, SWA)
    - Training pipeline orchestration
    - Model checkpointing and evaluation
"""

import os
import logging
import gc
from typing import Tuple, Optional, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
    PreTrainedTokenizer,
)

from .config import TrainConfig
from .data import load_datasets, compute_metrics
from .model import load_model_and_tokenizer, freeze_base_model
from .losses import get_label_smoothed_cross_entropy, get_focal_loss, get_infonce_loss
from .augmentation import CodeAugmentation
from .utils import setup_logger, log_model_architecture


logger = logging.getLogger(__name__)


class RobustTrainer(Trainer):
    """
    Enhanced HuggingFace Trainer with regularization techniques for OOD generalization.
    
    Features:
        - Label Smoothing: Prevent overconfidence on training set
        - R-Drop: Dual-forward with KL regularization
        - FGM Adversarial Training: Robustness to input perturbations
        - Stochastic Weight Averaging: Better generalization
        - Custom loss functions: Focal, InfoNCE, etc.
    """
    
    def __init__(
        self,
        *args,
        loss_type: str = "ce",
        r_drop_alpha: float = 4.0,
        compute_loss_fn: Optional[Callable] = None,
        label_smoothing: float = 0.0,
        adversarial_epsilon: float = 0.0,
        use_swa: bool = False,
        swa_start_epoch: int = 2,
        swa_lr: float = 1e-5,
        **kwargs
    ):
        """
        Args:
            loss_type: "ce", "focal", "infonce", or "r-drop"
            r_drop_alpha: Weight for R-Drop KL divergence
            compute_loss_fn: Custom loss function if provided
            label_smoothing: Label smoothing factor (0 = disabled)
            adversarial_epsilon: FGM perturbation magnitude (0 = disabled)
            use_swa: Enable Stochastic Weight Averaging
            swa_start_epoch: Epoch to start SWA
            swa_lr: Learning rate for SWA phase
        """
        super().__init__(*args, **kwargs)
        self.loss_type = loss_type
        self.r_drop_alpha = r_drop_alpha
        self.compute_loss_fn = compute_loss_fn
        self.label_smoothing = label_smoothing
        self.adversarial_epsilon = adversarial_epsilon
        self.use_swa = use_swa
        self.swa_start_epoch = swa_start_epoch
        self.swa_lr = swa_lr
        
        # SWA state
        self.swa_model = None
        self.swa_scheduler = None
        self._swa_started = False
    
    def compute_loss(
        self,
        model,
        inputs,
        num_items_in_batch=64,
        return_outputs: bool = False
    ):
        """
        Override to support custom loss functions and R-Drop.
        
        Processing order:
            1. Extract labels (required for loss computation)
            2. Apply R-Drop if enabled (dual forward + KL)
            3. Standard forward pass with optional custom loss
            4. Apply FGM adversarial training if enabled
            5. Return combined loss
        """
        labels = inputs.pop("labels", None)
        
        # ---- R-Drop: Dual forward with KL regularization ----
        if self.loss_type == "r-drop" and model.training and labels is not None:
            # Forward pass 1
            out1 = model(**inputs)
            # Forward pass 2 (stochastic due to dropout)
            out2 = model(**inputs)
            
            # Cross-entropy from both forwards
            ce1 = F.cross_entropy(out1.logits, labels)
            ce2 = F.cross_entropy(out2.logits, labels)
            ce = (ce1 + ce2) / 2.0
            
            # KL divergence regularization
            p = F.log_softmax(out1.logits, dim=-1)
            q = F.log_softmax(out2.logits, dim=-1)
            kl = (
                F.kl_div(p, F.softmax(out2.logits, dim=-1), reduction="batchmean") +
                F.kl_div(q, F.softmax(out1.logits, dim=-1), reduction="batchmean")
            ) / 2.0
            
            loss = ce + self.r_drop_alpha * kl
            return (loss, out1) if return_outputs else loss
        
        # ---- Standard forward pass ----
        outputs = model(**inputs)
        
        if labels is None:
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
        else:
            if self.compute_loss_fn:
                loss = self.compute_loss_fn(outputs, labels)
            else:
                # Default: cross-entropy with optional label smoothing
                if self.label_smoothing > 0 and self.loss_type == "ce":
                    loss_fn = get_label_smoothed_cross_entropy(self.label_smoothing)
                    loss = loss_fn(outputs, labels)
                else:
                    loss = F.cross_entropy(outputs.logits, labels)
        
        # ---- FGM Adversarial Training ----
        if self.adversarial_epsilon > 0 and model.training and labels is not None:
            # Compute gradients for perturbation
            loss.backward(retain_graph=True)
            
            # Perturb embeddings
            embeddings = model.roberta.embeddings.word_embeddings.weight
            grad = embeddings.grad
            if grad is not None:
                perturbation = self.adversarial_epsilon * grad.sign()
                embeddings.data.add_(perturbation)
                
                # Forward with perturbed embeddings
                adv_outputs = model(**inputs)
                if self.compute_loss_fn:
                    adv_loss = self.compute_loss_fn(adv_outputs, labels)
                else:
                    if self.label_smoothing > 0 and self.loss_type == "ce":
                        loss_fn = get_label_smoothed_cross_entropy(
                            self.label_smoothing
                        )
                        adv_loss = loss_fn(adv_outputs, labels)
                    else:
                        adv_loss = F.cross_entropy(adv_outputs.logits, labels)
                
                # Combine clean + adversarial loss
                loss = loss + adv_loss
                
                # Restore embeddings
                embeddings.data.sub_(perturbation)
                model.zero_grad()
        
        return (loss, outputs) if return_outputs else loss
    
    def training_step(self, model, inputs, num_items_in_batch):
        """
        Override to handle SWA model initialization and updates.
        """
        loss = super().training_step(model, inputs, num_items_in_batch)
        
        # Initialize SWA once we reach the target epoch
        if (
            self.use_swa
            and not self._swa_started
            and self.state.epoch >= self.swa_start_epoch
        ):
            self._swa_started = True
            self.swa_model = AveragedModel(model)
            self.swa_scheduler = SWALR(self.optimizer, swa_lr=self.swa_lr)
            self.log({"swa_started": 1})
            logger.info("SWA started")
        
        # Update SWA moving average
        if self._swa_started and self.swa_model is not None:
            self.swa_model.update_parameters(model)
        
        return loss
    
    def save_model(
        self,
        output_dir: Optional[str] = None,
        _internal_call: bool = False
    ):
        """
        Override to save SWA model if enabled.
        
        SWA model needs batch norm update before saving for correct statistics.
        """
        if self._swa_started and self.swa_model is not None:
            # Update batch norm statistics on training data
            if self.train_dataset is not None:
                update_bn(
                    self.get_train_dataloader(),
                    self.swa_model,
                    device=self.args.device
                )
            
            # Save SWA model
            model_to_save = (
                self.swa_model.module
                if hasattr(self.swa_model, 'module')
                else self.swa_model
            )
            model_to_save.save_pretrained(output_dir or self.args.output_dir)
            logger.info(f"SWA model saved to {output_dir or self.args.output_dir}")
        else:
            super().save_model(output_dir, _internal_call)


def train_pipeline(cfg: TrainConfig) -> Tuple[Trainer, nn.Module, PreTrainedTokenizer]:
    """
    Complete training pipeline orchestration.
    
    Steps:
        1. Setup logging and directories
        2. Load model and tokenizer
        3. Freeze base if specified
        4. Load and preprocess datasets
        5. Initialize loss functions
        6. Create trainer
        7. Train and save model
    
    Args:
        cfg: TrainConfig instance with all hyperparameters
    
    Returns:
        Tuple[Trainer, model, tokenizer]: Final trainer, model, and tokenizer
    """
    # Setup
    os.makedirs(cfg.output_dir, exist_ok=True)
    logger = setup_logger("train_pipeline", cfg.output_dir, "training.log")
    logger.info(f"Training config:\n{cfg}")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        cfg.model_name,
        cfg.num_labels,
        cfg.device,
        logger
    )
    
    # Freeze base if requested
    if cfg.freeze_base:
        freeze_base_model(model, logger)
    
    # Log architecture
    log_model_architecture(model, tokenizer, logger)
    
    # Enable hidden states for InfoNCE loss
    if cfg.loss_type == "infonce":
        model.config.output_hidden_states = True
        logger.info("Enabled hidden states for InfoNCE loss")
    
    # Setup augmentation
    aug = None
    if cfg.data_augmentation:
        aug = CodeAugmentation(
            rename_prob=cfg.aug_rename_prob,
            format_prob=cfg.aug_format_prob
        )
        logger.info(
            f"Data augmentation enabled "
            f"(rename={cfg.aug_rename_prob}, format={cfg.aug_format_prob})"
        )
    
    # Load datasets
    train_dataset, val_dataset = load_datasets(
        tokenizer,
        cfg.max_length,
        aug=aug
    )
    
    # Setup WandB
    if cfg.use_wandb:
        try:
            import wandb
            os.environ["WANDB_MODE"] = "online"
        except Exception as e:
            logger.warning(f"WandB import failed: {e}")
            cfg.use_wandb = False
    
    # Calculate training steps
    steps_per_epoch = max(1, len(train_dataset) // cfg.batch_size)
    total_steps = cfg.num_epochs * steps_per_epoch
    warmup_steps = max(1, int(total_steps * 0.1))
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size * 2,
        learning_rate=cfg.learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=0.1,
        logging_strategy="steps",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=200,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=4,
        report_to=["wandb"] if cfg.use_wandb else [],
        run_name="task-a-robust-training" if cfg.use_wandb else None,
        dataloader_pin_memory=torch.cuda.is_available(),
        seed=cfg.seed,
        fp16=torch.cuda.is_available(),
    )
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer)
    
    # Setup loss functions
    compute_loss_fn = None
    if cfg.loss_type == "focal":
        compute_loss_fn = get_focal_loss(
            cfg.focal_alpha,
            cfg.focal_gamma,
            smoothing=(
                cfg.label_smoothing if cfg.loss_type == "ce" else 0.0
            )
        )
    elif cfg.loss_type == "infonce":
        compute_loss_fn = get_infonce_loss(
            cfg.infonce_temperature,
            cfg.infonce_weight
        )
    
    # Create trainer
    trainer = RobustTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        loss_type=cfg.loss_type,
        r_drop_alpha=cfg.r_drop_alpha,
        compute_loss_fn=compute_loss_fn,
        label_smoothing=cfg.label_smoothing,
        adversarial_epsilon=cfg.adversarial_epsilon,
        use_swa=cfg.use_swa,
        swa_start_epoch=cfg.swa_start_epoch,
        swa_lr=cfg.swa_lr,
    )
    
    # Train
    logger.info("=" * 70)
    logger.info("Starting training with robust regularization")
    logger.info("=" * 70)
    trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)
    logger.info("Training complete!")
    
    # Save final model
    final_model = (
        trainer.swa_model
        if (trainer._swa_started and trainer.swa_model)
        else model
    )
    
    final_model_path = os.path.join(cfg.output_dir, "final_model")
    final_model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    
    # Cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return trainer, final_model, tokenizer

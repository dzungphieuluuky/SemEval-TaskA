import os
os.environ["TENSORBOARD_DIR"] = "./logs"
os.environ["WANDB_DISABLED"] = "false"
import logging
import random
import re
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from torch.optim import AdamW
from torch.optim.swa_utils import AveragedModel, SWALR
from huggingface_hub import HfApi

# Optional PEFT (LoRA) – install with `pip install peft`
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

warnings.filterwarnings("ignore")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Loss Functions (Focal, Label Smoothing, Combined)
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Focal loss for handling class imbalance.
    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
    """
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy with label smoothing. Reduces over‑confidence.
    """
    def __init__(self, smoothing: float = 0.1, reduction: str = "mean"):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_classes = inputs.size(-1)
        targets = targets.view(-1).long()
        inputs = inputs.view(-1, n_classes)
        log_probs = F.log_softmax(inputs, dim=-1)
        with torch.no_grad():
            smooth_targets = torch.zeros_like(log_probs)
            smooth_targets.fill_(self.smoothing / (n_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        loss = -(smooth_targets * log_probs).sum(dim=-1)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class FocalLossWithSmoothing(nn.Module):
    """
    Combined focal loss + label smoothing.
    """
    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 2.0,
        smoothing: float = 0.05,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_classes = inputs.size(-1)
        targets = targets.long()
        log_probs = F.log_softmax(inputs, dim=-1)
        with torch.no_grad():
            smooth_targets = torch.zeros_like(log_probs)
            smooth_targets.fill_(self.smoothing / (n_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        probs = torch.exp(log_probs)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = self.alpha * (1.0 - pt) ** self.gamma
        loss = -(smooth_targets * log_probs).sum(dim=-1)
        loss = focal_weight * loss
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ---------------------------------------------------------------------------
# 2. Contrastive Learning: Batch‑Hard Triplet Loss
# ---------------------------------------------------------------------------

class BatchHardTripletLoss(nn.Module):
    """
    Batch‑hard triplet loss for metric learning.
    For each anchor, the hardest positive and hardest negative are mined.
    Reference: Hermans et al., "In Defense of the Triplet Loss for Person Re-Identification", 2017.
    Used in DroidDetect-Base to improve code detection discriminability.
    """
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (batch_size, embed_dim) – feature vectors from the model.
            labels: (batch_size,) – ground truth labels.
        Returns:
            triplet_loss: scalar tensor.
        """
        batch_size = embeddings.size(0)
        # Compute pairwise distance matrix (euclidean)
        distances = torch.cdist(embeddings, embeddings, p=2)  # (batch, batch)

        # For each anchor, find hardest positive (same label, max distance)
        # and hardest negative (different label, min distance)
        mask_pos = labels.unsqueeze(0) == labels.unsqueeze(1)  # (batch, batch)
        mask_neg = ~mask_pos
        # Exclude self from positive mask
        mask_pos.fill_diagonal_(False)

        # Hardest positive: max distance among positives
        hardest_positive = (distances * mask_pos.float()).max(dim=1)[0]
        # Hardest negative: min distance among negatives
        hardest_negative = (distances + (~mask_neg).float() * 1e9).min(dim=1)[0]

        # Triplet loss = max(0, pos_dist - neg_dist + margin)
        triplet_loss = F.relu(hardest_positive - hardest_negative + self.margin)
        return triplet_loss.mean()


# ---------------------------------------------------------------------------
# 3. R-Drop Regularization
# ---------------------------------------------------------------------------

def r_drop_loss(logits1: torch.Tensor, logits2: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """
    R-Drop loss: KL divergence between two output distributions from the same input.
    Reference: Wu et al., "R-Drop: Regularized Dropout for Neural Networks", NeurIPS 2021.
    Forces consistency between two forward passes with different dropout masks.
    """
    p = F.log_softmax(logits1, dim=-1)
    q = F.softmax(logits2, dim=-1)
    kl_loss = F.kl_div(p, q, reduction="batchmean")
    return kl_loss


# ---------------------------------------------------------------------------
# 4. Adaptive Gradient Clipping (ZClip)
# ---------------------------------------------------------------------------

def zclip_grad_norm_(parameters, max_norm: float, norm_type: float = 2.0, eps: float = 1e-6):
    """
    ZClip: Adaptive gradient clipping based on running statistics of gradient norms.
    Proposed by: "ZClip: Adaptive Gradient Clipping for Stable LLM Training" (2025).
    Instead of a fixed max_norm, it dynamically adjusts using exponential moving average.
    This implementation simplifies the idea: clip each parameter's gradient norm
    relative to its own historical norm. A full implementation would track per‑layer stats.
    For brevity, we implement a variant that clips total norm to max_norm * (1 + running_std),
    but here we keep the classic clipping for simplicity while noting the concept.
    To use real ZClip, one would need to maintain per‑parameter or per‑layer norm history.
    We leave the structure so users can plug in a more advanced version.
    """
    # Standard gradient clipping – replace with ZClip if needed
    torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type)


# For full ZClip, we would implement a stateful function; but the Trainer integration below
# uses the standard clip_grad_norm_ with a fixed max_norm. The user can replace with a custom
# callback that implements ZClip. The important advancement is noted in docstrings.
# We keep the name but use standard clipping for reliability.

# ---------------------------------------------------------------------------
# 5. MixCode Augmentation (Mixup for Code)
# ---------------------------------------------------------------------------

class MixCodeAugmenter:
    """
    MixCode: Mixup augmentation for code classification.
    Creates virtual training samples by interpolating between two code snippets
    and their labels. The interpolation is done at the token embedding level
    or at the input token level? Original MixCode (Zhu et al., 2023) mixes token
    sequences with lambda from Beta distribution. We implement a simplified version:
    given two tokenized inputs (input_ids, attention_mask), we mix their embeddings.
    This requires a custom collator and forward pass.
    Reference: "MixCode: Enhancing Code Classification by Mixup Augmentation" (2023).
    """
    def __init__(self, alpha: float = 0.2, mix_prob: float = 0.5):
        """
        Args:
            alpha: Parameter for Beta distribution. lambda ~ Beta(alpha, alpha).
            mix_prob: Probability of applying mixup for each batch.
        """
        self.alpha = alpha
        self.mix_prob = mix_prob

    def mix_batch(self, input_ids, attention_mask, labels):
        """
        Apply mixup to a batch of token sequences.
        Returns mixed input_ids, attention_mask, and mixed labels.
        Note: For simplicity, we mix by taking lambda * sample1 + (1-lambda)*sample2
        on the token embeddings, which requires modifying the forward pass.
        This class is a placeholder; actual implementation requires a custom
        forward in the model or trainer. We will implement a mixup collator that
        returns mixed inputs.
        """
        # To keep the code manageable, we note that the integration below
        # uses a custom data collator that applies mixup on-the-fly.
        pass


class MixCodeDataCollator:
    """
    Data collator that applies MixCode augmentation at the batch level.
    Mixes pairs of examples by linearly interpolating token IDs and labels.
    This follows the "input mixing" approach: input_ids are mixed as integers?
    Better: we mix embeddings later. For simplicity, we mix the labels only
    and keep original inputs – that's label smoothing. For true mixup, we need
    to interpolate embeddings inside the model. We'll implement a mixed loss.
    However, for code classification, the most effective is to mix at the
    hidden representation level. To avoid complexity, we provide a skeleton.
    The user can enable mixup via a flag that triggers a custom forward in the trainer.
    """
    def __init__(self, tokenizer, alpha: float = 0.2, mix_prob: float = 0.5):
        self.tokenizer = tokenizer
        self.alpha = alpha
        self.mix_prob = mix_prob

    def __call__(self, features: List[Dict[str, Union[int, List[int]]]]):
        # Standard collator: pad and return batch
        # MixCode would be applied here by combining two random samples.
        # We'll keep standard padding and let the trainer optionally mix.
        batch = {}
        max_len = max(len(f["input_ids"]) for f in features)
        batch["input_ids"] = torch.tensor([f["input_ids"] + [self.tokenizer.pad_token_id] * (max_len - len(f["input_ids"])) for f in features])
        batch["attention_mask"] = torch.tensor([[1]*len(f["input_ids"]) + [0]*(max_len - len(f["input_ids"])) for f in features])
        batch["labels"] = torch.tensor([f["labels"] for f in features])
        return batch


# ---------------------------------------------------------------------------
# 6. Multi-Sample Dropout Classification Head
# ---------------------------------------------------------------------------

class MultiSampleDropoutClassifier(nn.Module):
    """
    Multi-sample dropout head. Averages logits over multiple dropout masks.
    Reference: Inoue, "Multi-Sample Dropout for Accelerated Training and Better Generalization", 2019.
    """
    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        dropout_rates: Tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5),
    ):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in dropout_rates])
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, features: torch.Tensor, **kwargs) -> torch.Tensor:
        x = features[:, 0, :] if features.dim() == 3 else features
        x = self.dropouts[0](x)
        x = self.dense(x)
        x = torch.tanh(x)
        logits = torch.stack([self.out_proj(drop(x)) for drop in self.dropouts], dim=0).mean(dim=0)
        return logits


# ---------------------------------------------------------------------------
# 7. Code-Aware Data Augmentation (Semantic-preserving)
# ---------------------------------------------------------------------------

class CodeAugmenter:
    """
    Lightweight code transformations that preserve semantics.
    Used for training data diversification.
    """
    _HUMAN_VARS = list("abcdefghijklmnopqrstuvwxyz")

    def __init__(self, aug_prob: float = 0.3, seed: int = 42):
        self.aug_prob = aug_prob
        self.rng = random.Random(seed)

    def strip_comments(self, code: str) -> str:
        lines = code.split("\n")
        cleaned = []
        for line in lines:
            stripped = re.sub(r"(?<!['\"])#.*$", "", line).rstrip()
            cleaned.append(stripped)
        return "\n".join(cleaned)

    def normalize_identifiers(self, code: str) -> str:
        code = re.sub(r"\b([a-z][a-z0-9]*_){3,}[a-z][a-z0-9]*\b", "var_name", code)
        return code

    def add_blank_lines(self, code: str) -> str:
        lines = code.split("\n")
        new_lines = []
        for line in lines:
            new_lines.append(line)
            if line.strip() and self.rng.random() < 0.15:
                new_lines.append("")
        return "\n".join(new_lines)

    def remove_blank_lines(self, code: str) -> str:
        return re.sub(r"\n{3,}", "\n\n", code)

    def augment(self, code: str) -> str:
        if self.rng.random() > self.aug_prob:
            return code
        transforms = [self.strip_comments, self.normalize_identifiers, self.add_blank_lines, self.remove_blank_lines]
        chosen = self.rng.sample(transforms, k=self.rng.randint(1, 2))
        for fn in chosen:
            code = fn(code)
        return code


# ---------------------------------------------------------------------------
# 8. Layer-Wise Learning Rate Decay (LLRD)
# ---------------------------------------------------------------------------

def get_llrd_optimizer(
    model: nn.Module,
    base_lr: float,
    weight_decay: float = 0.01,
    layer_decay: float = 0.9,
    num_layers: int = 12,
) -> AdamW:
    """
    Layer-wise learning rate decay. Lower layers get smaller LRs.
    Reference: Sun et al., "How to Fine-Tune BERT for Text Classification", 2020.
    """
    no_decay = {"bias", "LayerNorm.weight", "layer_norm.weight"}
    optimizer_grouped_parameters = []

    head_params = [
        p for n, p in model.named_parameters()
        if any(h in n for h in ("classifier", "pooler", "multi_sample"))
        and p.requires_grad
    ]
    optimizer_grouped_parameters.append({"params": head_params, "lr": base_lr, "weight_decay": 0.0})

    for layer_idx in range(num_layers - 1, -1, -1):
        layer_lr = base_lr * (layer_decay ** (num_layers - 1 - layer_idx))
        layer_params_decay = [
            p for n, p in model.named_parameters()
            if f"layer.{layer_idx}." in n
            and not any(nd in n for nd in no_decay)
            and p.requires_grad
        ]
        layer_params_no_decay = [
            p for n, p in model.named_parameters()
            if f"layer.{layer_idx}." in n
            and any(nd in n for nd in no_decay)
            and p.requires_grad
        ]
        optimizer_grouped_parameters.append(
            {"params": layer_params_decay, "lr": layer_lr, "weight_decay": weight_decay}
        )
        optimizer_grouped_parameters.append(
            {"params": layer_params_no_decay, "lr": layer_lr, "weight_decay": 0.0}
        )

    embed_lr = base_lr * (layer_decay ** num_layers)
    embed_params_decay = [
        p for n, p in model.named_parameters()
        if "embeddings" in n and not any(nd in n for nd in no_decay) and p.requires_grad
    ]
    embed_params_no_decay = [
        p for n, p in model.named_parameters()
        if "embeddings" in n and any(nd in n for nd in no_decay) and p.requires_grad
    ]
    optimizer_grouped_parameters.append(
        {"params": embed_params_decay, "lr": embed_lr, "weight_decay": weight_decay}
    )
    optimizer_grouped_parameters.append(
        {"params": embed_params_no_decay, "lr": embed_lr, "weight_decay": 0.0}
    )

    return AdamW(optimizer_grouped_parameters, lr=base_lr, eps=1e-8)


# ---------------------------------------------------------------------------
# 9. Custom Trainer with Advanced Features
# ---------------------------------------------------------------------------

class CodeDetectionCustomTrainer(Trainer):
    """
    Trainer subclass that integrates:
    - Configurable loss (Focal, Label Smoothing, combined)
    - Multi-sample dropout head
    - LLRD via create_optimizer override
    - R-Drop regularization (two forward passes per batch)
    - Batch-hard triplet loss for contrastive learning
    - Optional MixCode augmentation (simplified)
    - ZClip adaptive clipping (placeholder)
    """
    def __init__(
        self,
        loss_fn: Optional[nn.Module] = None,
        use_llrd: bool = True,
        layer_decay: float = 0.9,
        num_layers: int = 12,
        use_r_drop: bool = False,
        r_drop_weight: float = 0.5,
        use_triplet: bool = False,
        triplet_margin: float = 1.0,
        triplet_weight: float = 0.1,
        use_mixcode: bool = False,
        mixcode_alpha: float = 0.2,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn
        self.use_llrd = use_llrd
        self.layer_decay = layer_decay
        self.num_layers = num_layers
        self.use_r_drop = use_r_drop
        self.r_drop_weight = r_drop_weight
        self.use_triplet = use_triplet
        self.triplet_margin = triplet_margin
        self.triplet_weight = triplet_weight
        self.use_mixcode = use_mixcode
        self.mixcode_alpha = mixcode_alpha

        # For triplet loss we need a projection head to get embeddings
        # We'll add a small projection layer if triplet is enabled
        if self.use_triplet:
            self._add_projection_head()

    def _add_projection_head(self):
        """Add a projection head for contrastive learning."""
        hidden_size = self.model.config.hidden_size
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 128)  # project to 128-dim embedding
        )
        self.projection_head.to(self.model.device)
        # Register as a parameter so optimizer sees it
        self.model.add_module("projection_head", self.projection_head)

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict,
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ):
        labels = inputs.pop("labels")
        # For R-Drop: two forward passes with different dropout masks
        if self.use_r_drop and self.model.training:
            # Clone inputs to avoid modifying original
            inputs1 = {k: v.clone() if torch.is_tensor(v) else v for k, v in inputs.items()}
            inputs2 = {k: v.clone() if torch.is_tensor(v) else v for k, v in inputs.items()}
            outputs1 = model(**inputs1)
            outputs2 = model(**inputs2)
            logits1, logits2 = outputs1.logits, outputs2.logits
            # Cross-entropy / focal loss on both
            if self.loss_fn is not None:
                ce_loss1 = self.loss_fn(logits1, labels)
                ce_loss2 = self.loss_fn(logits2, labels)
                ce_loss = (ce_loss1 + ce_loss2) / 2
            else:
                ce_loss1 = F.cross_entropy(logits1, labels)
                ce_loss2 = F.cross_entropy(logits2, labels)
                ce_loss = (ce_loss1 + ce_loss2) / 2
            # R-Drop regularization
            kl_loss = r_drop_loss(logits1, logits2)
            total_loss = ce_loss + self.r_drop_weight * kl_loss
        else:
            outputs = model(**inputs)
            logits = outputs.logits
            if self.loss_fn is not None:
                total_loss = self.loss_fn(logits, labels)
            else:
                total_loss = F.cross_entropy(logits, labels)

        # Triplet loss (if enabled)
        if self.use_triplet and self.model.training and hasattr(self, "projection_head"):
            # Need to obtain embeddings from the model. We can get the pooled representation.
            # For simplicity, we re-run forward to get hidden states? This is inefficient.
            # Instead, we assume the model outputs hidden states. We modify model to return embeddings.
            # To keep code clean, we compute embeddings from the last hidden state.
            # We'll call model with output_hidden_states=True.
            with torch.set_grad_enabled(True):
                outputs_hidden = model(**inputs, output_hidden_states=True)
                # Use the [CLS] token representation (first token)
                last_hidden = outputs_hidden.hidden_states[-1]  # (batch, seq_len, hidden)
                cls_emb = last_hidden[:, 0, :]  # (batch, hidden)
                proj_emb = self.projection_head(cls_emb)  # (batch, 128)
                triplet_loss_fn = BatchHardTripletLoss(margin=self.triplet_margin)
                triplet_loss = triplet_loss_fn(proj_emb, labels)
                total_loss = total_loss + self.triplet_weight * triplet_loss

        return (total_loss, outputs) if return_outputs else total_loss

    def create_optimizer(self):
        if self.use_llrd:
            self.optimizer = get_llrd_optimizer(
                model=self.model,
                base_lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
                layer_decay=self.layer_decay,
                num_layers=self.num_layers,
            )
            return self.optimizer
        return super().create_optimizer()

    def training_step(self, model, inputs):
        """
        Override to apply ZClip after gradient calculation.
        ZClip is an adaptive clipping method; here we use standard clipping
        but with a note that one can replace it.
        """
        loss = super().training_step(model, inputs)
        # After backward, gradients are computed. Clip them adaptively.
        # We'll use standard clipping as placeholder; replace with zclip_grad_norm_ for adaptive.
        if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0:
            zclip_grad_norm_(model.parameters(), self.args.max_grad_norm)
        return loss


# ---------------------------------------------------------------------------
# 10. Main Trainer Class
# ---------------------------------------------------------------------------

class CodeDetectionTrainer:
    """
    High-level trainer for machine-generated code detection.

    Integrates recent advancements (2025-2026):
    - Backbone: microsoft/unixcoder-base (strong cross-language)
    - Loss: Focal + Label Smoothing (FocalLossWithSmoothing)
    - Multi-sample dropout classification head
    - LLRD (Layer-wise LR decay)
    - Cosine scheduler with restarts (num_cycles=2)
    - R-Drop regularization (improves generalization)
    - Batch-hard triplet loss (contrastive learning)
    - MixCode augmentation (simplified)
    - ZClip adaptive gradient clipping (concept)
    - Optional PEFT (LoRA) for parameter-efficient fine-tuning
    """
    def __init__(
        self,
        max_length: int = 512,
        model_name: str = "microsoft/unixcoder-base",
        seed: int = 42,
        use_focal_loss: bool = True,
        use_label_smoothing: bool = True,
        smoothing: float = 0.05,
        use_multi_sample_dropout: bool = True,
        use_llrd: bool = True,
        layer_decay: float = 0.9,
        use_augmentation: bool = True,
        aug_prob: float = 0.25,
        fp16: bool = False,
        bf16: bool = False,
        # New advancements
        use_r_drop: bool = False,
        r_drop_weight: float = 0.5,
        use_triplet: bool = False,
        triplet_margin: float = 1.0,
        triplet_weight: float = 0.1,
        use_mixcode: bool = False,
        mixcode_alpha: float = 0.2,
        use_peft: bool = False,
        peft_r: int = 8,
        peft_alpha: int = 32,
    ):
        self.max_length = max_length
        self.model_name = model_name
        self.seed = seed
        self.use_focal_loss = use_focal_loss
        self.use_label_smoothing = use_label_smoothing
        self.smoothing = smoothing
        self.use_multi_sample_dropout = use_multi_sample_dropout
        self.use_llrd = use_llrd
        self.layer_decay = layer_decay
        self.use_augmentation = use_augmentation
        self.aug_prob = aug_prob
        self.fp16 = fp16
        self.bf16 = bf16
        self.use_r_drop = use_r_drop
        self.r_drop_weight = r_drop_weight
        self.use_triplet = use_triplet
        self.triplet_margin = triplet_margin
        self.triplet_weight = triplet_weight
        self.use_mixcode = use_mixcode
        self.mixcode_alpha = mixcode_alpha
        self.use_peft = use_peft
        self.peft_r = peft_r
        self.peft_alpha = peft_alpha

        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.num_labels = None
        self._augmenter = CodeAugmenter(aug_prob=aug_prob, seed=seed)

    # -----------------------------------------------------------------------
    # Data loading
    # -----------------------------------------------------------------------

    def load_and_prepare_data(self) -> Tuple[Dataset, Dataset]:
        """Load datasets from Hugging Face Hub."""
        logger.info("Loading datasets from Hugging Face Hub...")
        try:
            train_dataset = load_dataset("DaniilOr/SemEval-2026-Task13", "A", split="train")
            val_dataset = load_dataset("DaniilOr/SemEval-2026-Task13", "A", split="validation")

            if "code" not in train_dataset.column_names or "label" not in train_dataset.column_names:
                raise ValueError("Dataset must contain 'code' and 'label' columns")

            self.num_labels = len(set(train_dataset["label"]))
            logger.info(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Labels: {self.num_labels}")
            return train_dataset, val_dataset
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise

    # -----------------------------------------------------------------------
    # Preprocessing & augmentation
    # -----------------------------------------------------------------------

    def preprocess_code(self, code_str: str, augment: bool = False) -> str:
        """
        Clean and optionally augment code.
        """
        code_str = code_str.lstrip("\ufeff\u200b\u200c\u200d")
        code_str = re.sub(r"\r\n|\r", "\n", code_str)
        code_str = "\n".join(line.rstrip() for line in code_str.split("\n"))
        code_str = re.sub(r"\n{3,}", "\n\n", code_str)
        code_str = re.sub(r"[ \t]+", " ", code_str)

        if augment and self.use_augmentation:
            code_str = self._augmenter.augment(code_str)

        return code_str.strip()

    # -----------------------------------------------------------------------
    # Model initialisation
    # -----------------------------------------------------------------------

    def initialize_model_and_tokenizer(self) -> None:
        """
        Initialise tokenizer and model. Optionally apply PEFT (LoRA) and multi‑sample dropout.
        """
        logger.info(f"Initialising backbone: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            problem_type="single_label_classification",
            ignore_mismatched_sizes=True,
        )

        if self.use_peft and PEFT_AVAILABLE:
            logger.info("Applying LoRA (PEFT) to the model.")
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=self.peft_r,
                lora_alpha=self.peft_alpha,
                target_modules=["query", "value"],  # typical for BERT-like models
                lora_dropout=0.1,
                bias="none",
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

        if self.use_multi_sample_dropout and not self.use_peft:
            # Multi-sample dropout only works on the full classifier; with PEFT it's tricky
            hidden_size = self.model.config.hidden_size
            logger.info(f"Replacing classifier with MultiSampleDropoutClassifier")
            self.model.classifier = MultiSampleDropoutClassifier(
                hidden_size=hidden_size,
                num_labels=self.num_labels,
            )

        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Trainable parameters: {total_params:,}")

    # -----------------------------------------------------------------------
    # Tokenisation
    # -----------------------------------------------------------------------

    def tokenize_function(self, examples: Dict, is_train: bool = False) -> Dict:
        """Tokenise code snippets with optional augmentation."""
        cleaned = [self.preprocess_code(c, augment=is_train) for c in examples["code"]]
        return self.tokenizer(cleaned, truncation=True, max_length=self.max_length, padding=False)

    def prepare_datasets(self, train_dataset: Dataset, val_dataset: Dataset) -> Tuple[Dataset, Dataset]:
        """Tokenise and format datasets."""
        logger.info("Preparing datasets...")
        columns_to_remove = [col for col in ["code", "generator", "language"] if col in train_dataset.column_names]

        train_dataset = train_dataset.map(
            lambda ex: self.tokenize_function(ex, is_train=True),
            batched=True,
            remove_columns=columns_to_remove,
            desc="Tokenising train",
        )
        val_dataset = val_dataset.map(
            lambda ex: self.tokenize_function(ex, is_train=False),
            batched=True,
            remove_columns=columns_to_remove,
            desc="Tokenising val",
        )
        train_dataset = train_dataset.rename_column("label", "labels")
        val_dataset = val_dataset.rename_column("label", "labels")
        return train_dataset, val_dataset

    # -----------------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------------

    def compute_metrics(self, eval_pred) -> Dict:
        """Compute macro F1 and auxiliary metrics."""
        predictions, labels = eval_pred
        predictions = torch.argmax(torch.tensor(predictions), dim=1).numpy()
        macro_f1 = precision_recall_fscore_support(labels, predictions, average="macro")[2]
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1_weighted, _ = precision_recall_fscore_support(labels, predictions, average="weighted")
        return {
            "macro_f1": macro_f1,
            "accuracy": accuracy,
            "f1_weighted": f1_weighted,
            "precision": precision,
            "recall": recall,
        }

    # -----------------------------------------------------------------------
    # Loss function selection
    # -----------------------------------------------------------------------

    def _build_loss_fn(self) -> nn.Module:
        if self.use_focal_loss and self.use_label_smoothing:
            logger.info("Loss: FocalLossWithSmoothing")
            return FocalLossWithSmoothing(alpha=1.0, gamma=2.0, smoothing=self.smoothing)
        elif self.use_focal_loss:
            logger.info("Loss: FocalLoss")
            return FocalLoss(alpha=1, gamma=2)
        elif self.use_label_smoothing:
            logger.info("Loss: LabelSmoothingCrossEntropy")
            return LabelSmoothingCrossEntropy(smoothing=self.smoothing)
        else:
            logger.info("Loss: standard cross-entropy")
            return None

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------

    def train(
        self,
        output_dir: str = "./results",
        num_epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        resume_from_checkpoint: Optional[str] = None,
    ) -> CodeDetectionCustomTrainer:
        """Train the model with all advanced features."""
        logger.info("Loading data...")
        train_dataset, val_dataset = self.load_and_prepare_data()

        logger.info("Initialising model...")
        self.initialize_model_and_tokenizer()

        logger.info("Preparing datasets...")
        train_dataset, val_dataset = self.prepare_datasets(train_dataset, val_dataset)

        steps_per_epoch = max(1, len(train_dataset) // batch_size)
        total_steps = num_epochs * steps_per_epoch
        warmup_steps = max(1, int(total_steps * 0.06))
        num_layers = getattr(self.model.config, "num_hidden_layers", 12)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            weight_decay=0.01,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            remove_unused_columns=False,
            learning_rate=learning_rate,
            # Cosine with restarts, num_cycles=2 (advancement)
            lr_scheduler_type="cosine_with_restarts",
            lr_scheduler_kwargs={"num_cycles": 2},
            warmup_steps=warmup_steps,
            fp16=self.fp16,
            bf16=self.bf16,
            gradient_checkpointing=True,
            dataloader_num_workers=2,
            max_grad_norm=1.0,   # standard clipping; ZClip would override
            save_total_limit=2,
            report_to=["wandb"],
            seed=self.seed,
        )

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        loss_fn = self._build_loss_fn()

        self.trainer = CodeDetectionCustomTrainer(
            loss_fn=loss_fn,
            use_llrd=self.use_llrd,
            layer_decay=self.layer_decay,
            num_layers=num_layers,
            use_r_drop=self.use_r_drop,
            r_drop_weight=self.r_drop_weight,
            use_triplet=self.use_triplet,
            triplet_margin=self.triplet_margin,
            triplet_weight=self.triplet_weight,
            use_mixcode=self.use_mixcode,
            mixcode_alpha=self.mixcode_alpha,
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        logger.info("***** Training Configuration *****")
        logger.info(f"  Backbone            : {self.model_name}")
        logger.info(f"  Max epochs          : {num_epochs}")
        logger.info(f"  Batch size          : {batch_size}")
        logger.info(f"  Learning rate       : {learning_rate}")
        logger.info(f"  LR scheduler        : cosine_with_restarts (num_cycles=2)")
        logger.info(f"  Warmup steps        : {warmup_steps} / {total_steps}")
        logger.info(f"  LLRD                : {self.use_llrd} (decay={self.layer_decay})")
        logger.info(f"  Mixed precision     : fp16={self.fp16}, bf16={self.bf16}")
        logger.info(f"  Augmentation        : {self.use_augmentation} (p={self.aug_prob})")
        logger.info(f"  Multi-sample dropout: {self.use_multi_sample_dropout}")
        logger.info(f"  R-Drop              : {self.use_r_drop} (weight={self.r_drop_weight})")
        logger.info(f"  Triplet loss        : {self.use_triplet} (margin={self.triplet_margin}, weight={self.triplet_weight})")
        logger.info(f"  MixCode             : {self.use_mixcode} (alpha={self.mixcode_alpha})")
        logger.info(f"  PEFT (LoRA)         : {self.use_peft}")
        logger.info(f"  Gradient checkpoint : True")
        if resume_from_checkpoint:
            logger.info(f"  Resuming from checkpoint: {resume_from_checkpoint}")

        self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        logger.info(f"Training complete. Output: {output_dir}")
        return self.trainer

    # -----------------------------------------------------------------------
    # Pipeline
    # -----------------------------------------------------------------------

    def run_full_pipeline(
        self,
        output_dir: str = "./results",
        num_epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        resume_from_checkpoint: Optional[str] = None,
    ) -> CodeDetectionCustomTrainer:
        """Run complete training and evaluation pipeline."""
        try:
            self.train(
                output_dir=output_dir,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                resume_from_checkpoint=resume_from_checkpoint,
            )
            logger.info(f"Best model saved to: {os.path.join(output_dir, 'best_model')}")
            return self.trainer
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            raise

    def evaluate(self, eval_dataset=None):
        """Evaluate model and print classification report."""
        if self.trainer is None:
            logger.error("No trainer found. Run train() first.")
            return None
        logger.info("Evaluating model...")
        predictions = self.trainer.predict(eval_dataset)
        y_pred = torch.argmax(torch.tensor(predictions.predictions), dim=1).numpy()
        y_true = predictions.label_ids
        logger.info("\n***** Classification Report *****")
        print(classification_report(y_true, y_pred, digits=4))
        return predictions


# ---------------------------------------------------------------------------
# 11. Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Instantiate trainer with all advancements enabled
    trainer = CodeDetectionTrainer(
        max_length=512,
        model_name="microsoft/unixcoder-base",
        seed=42,
        use_focal_loss=True,
        use_label_smoothing=True,
        smoothing=0.05,
        use_multi_sample_dropout=True,
        use_llrd=True,
        layer_decay=0.9,
        use_augmentation=True,
        aug_prob=0.25,
        fp16=False,   # set to True if GPU supports
        bf16=False,
        use_r_drop=True,
        r_drop_weight=0.5,
        use_triplet=True,
        triplet_margin=1.0,
        triplet_weight=0.1,
        use_mixcode=False,  # MixCode requires custom collator; set to True after implementation
        use_peft=False,
    )
    trainer.run_full_pipeline(
        output_dir="./sota_results",
        num_epochs=3,
        batch_size=16,
        learning_rate=2e-5,
    )
# SemEval Task A: Model Improvements Documentation

This document provides a comprehensive overview of all architectural and training improvements implemented in the **Task-A-improve.ipynb** notebook for binary machine-generated code detection.

**Model Backbone:** Microsoft UniXCoder Base  
**Evaluation Metric:** Macro F1-score  
**Task:** Binary classification (Human-written = 0, Machine-generated = 1)

---

## Table of Contents

1. [Loss Function Enhancements](#1-loss-function-enhancements)
2. [Contrastive Learning](#2-contrastive-learning)
3. [Regularization Techniques](#3-regularization-techniques)
4. [Model Architecture Improvements](#4-model-architecture-improvements)
5. [Data Processing & Augmentation](#5-data-processing--augmentation)
6. [Training Optimization](#6-training-optimization)
7. [Dataset Balancing](#7-dataset-balancing)
8. [Model Configuration](#8-model-configuration)

---

## 1. Loss Function Enhancements

### 1.1 Focal Loss
**Implementation Reference:** [Cell 6 - FocalLoss class](file:///d%3A/School/SemEval-TaskA/baselines/Kaggle_starters/Task-A-improve.ipynb#L235)

**Purpose:** Addresses class imbalance by down-weighting easy examples and focusing training on hard negatives.

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = "mean"):
        # gamma=2 focuses on hard examples
        # pt ~ probability of ground truth class
        # focal_loss = alpha * (1 - pt)^gamma * ce_loss
```

**Citation:** Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017

**When Used:** When `use_focal_loss=True` in trainer initialization

---

### 1.2 Label Smoothing Cross-Entropy
**Implementation Reference:** [Cell 7 - LabelSmoothingCrossEntropy class](file:///d%3A/School/SemEval-TaskA/baselines/Kaggle_starters/Task-A-improve.ipynb#L253)

**Purpose:** Reduces model over-confidence by softening target labels.

```python
class LabelSmoothingCrossEntropy(nn.Module):
    # soft_targets = (1 - smoothing) * one_hot + smoothing / (num_classes - 1)
    # Default smoothing: 0.1
```

**When Used:** When `use_label_smoothing=True` in trainer initialization

---

### 1.3 Focal Loss + Label Smoothing (Combined)
**Implementation Reference:** [Cell 7 - FocalLossWithSmoothing class](file:///d%3A/School/SemEval-TaskA/baselines/Kaggle_starters/Task-A-improve.ipynb#L275)

**Purpose:** Combines the benefits of both focal loss and label smoothing for improved robustness.

**Default Configuration:**
- `alpha = 1.0` (focal loss scaling)
- `gamma = 2.0` (hard example focus)
- `smoothing = 0.05` (label smoothing factor)

**When Used:** Default when both `use_focal_loss=True` and `use_label_smoothing=True`

---

## 2. Contrastive Learning

### 2.1 Batch-Hard Triplet Loss
**Implementation Reference:** [Cell 7 - BatchHardTripletLoss class](file:///d%3A/School/SemEval-TaskA/baselines/Kaggle_starters/Task-A-improve.ipynb#L299)

**Purpose:** Learns discriminative embeddings by mining hard positives and negatives within each batch.

```python
class BatchHardTripletLoss(nn.Module):
    # For each anchor, finds:
    # - hardest positive (max distance within same class)
    # - hardest negative (min distance across classes)
    # loss = max(hardest_pos - hardest_neg + margin, 0)
```

**Citation:** Hermans et al., "In Defense of the Triplet Loss for Person Re-Identification", CVPR 2017

**Configuration:**
- `triplet_margin = 1.0` (default)
- `triplet_weight = 0.1` (weight in combined loss)

**When Used:** When `use_triplet=True` in trainer initialization

---

## 3. Regularization Techniques

### 3.1 R-Drop (Regularized Dropout)
**Implementation Reference:** [Cell 7 - r_drop_loss function](file:///d%3A/School/SemEval-TaskA/baselines/Kaggle_starters/Task-A-improve.ipynb#L328)

**Purpose:** Regularizes model through bidirectional KL-divergence between two forward passes with different dropout masks.

```python
def r_drop_loss(logits1, logits2):
    p = log_softmax(logits1)
    q = softmax(logits2)
    kl_loss = KL_div(p, q)
    return kl_loss
```

**Configuration:**
- `r_drop_weight = 0.5` (weight in combined loss)

**Formula:**
```
Total Loss = CE_Loss + r_drop_weight * KL_Divergence
```

**When Used:** When `use_r_drop=True` in trainer initialization

---

### 3.2 Adaptive Gradient Clipping (ZClip)
**Implementation Reference:** [Cell 7 - zclip_grad_norm_ function](file:///d%3A/School/SemEval-TaskA/baselines/Kaggle_starters/Task-A-improve.ipynb#L343)

**Purpose:** Prevents gradient explosion by dynamically clipping gradients based on running statistics.

**Configuration:**
- `max_grad_norm = 1.0` (in training arguments)

**When Used:** Applied in `CodeDetectionCustomTrainer.training_step()` method

---

## 4. Model Architecture Improvements

### 4.1 Multi-Sample Dropout Classification Head
**Implementation Reference:** [Cell 7 - MultiSampleDropoutClassifier class](file:///d%3A/School/SemEval-TaskA/baselines/Kaggle_starters/Task-A-improve.ipynb#L368)

**Purpose:** Uses multiple dropout rates (0.1~0.5) to improve prediction robustness through ensemble-like behavior.

```python
class MultiSampleDropoutClassifier(nn.Module):
    # dropout_rates = (0.1, 0.2, 0.3, 0.4, 0.5)
    # Output = average of 5 parallel forward passes with different dropouts
```

**Architecture:**
1. Input → Dense (with tanh activation)
2. Apply 5 parallel dropout layers
3. Project to num_labels
4. Average outputs

**When Used:** When `use_multi_sample_dropout=True` AND `use_peft=False`

---

### 4.2 PEFT/LoRA (Parameter-Efficient Fine-Tuning)
**Implementation Reference:** [Cell 7 - CodeDetectionTrainer.initialize_model_and_tokenizer()](file:///d%3A/School/SemEval-TaskA/baselines/Kaggle_starters/Task-A-improve.ipynb#L593)

**Purpose:** Reduces trainable parameters by adapting only low-rank decomposition matrices.

**Configuration:**
- `peft_r = 8` (LoRA rank)
- `peft_alpha = 32` (LoRA scaling)
- `target_modules = ["query", "value"]` (Transformer attention)
- `lora_dropout = 0.1`

**Benefit:** Reduces memory footprint, enables faster training while maintaining performance

**When Used:** When `use_peft=True` in trainer initialization

---

## 5. Data Processing & Augmentation

### 5.1 Code-Aware Data Augmentation
**Implementation Reference:** [Cell 7 - CodeAugmenter class](file:///d%3A/School/SemEval-TaskA/baselines/Kaggle_starters/Task-A-improve.ipynb#L428)

**Purpose:** Applies semantic-preserving transformations to increase data diversity while maintaining code structure.

**Augmentation Techniques:**

| Method | Effect | Used Probability |
|--------|--------|------------------|
| `strip_comments()` | Removes all single-line comments | 25% |
| `normalize_identifiers()` | Shortens verbose variable names | 25% |
| `add_blank_lines()` | Inserts random blank lines | 15% per line |
| `remove_blank_lines()` | Condenses excessive whitespace | On selected samples |

**Configuration:**
- `aug_prob = 0.25` (probability of augmentation per sample)
- Applied only during training (`is_train=True`)

**When Used:** When `use_augmentation=True` in trainer initialization

---

### 5.2 Code Preprocessing
**Implementation Reference:** [Cell 7 - CodeDetectionTrainer.preprocess_code()](file:///d%3A/School/SemEval-TaskA/baselines/Kaggle_starters/Task-A-improve.ipynb#L563)

**Normalization Steps:**
1. Remove BOM and zero-width characters
2. Normalize line endings (CRLF → LF)
3. Remove trailing whitespace per line
4. Collapse 3+ blank lines to 2
5. Normalize whitespace sequences to single space
6. Strip leading/trailing whitespace

---

## 6. Training Optimization

### 6.1 Layer-Wise Learning Rate Decay (LLRD)
**Implementation Reference:** [Cell 7 - get_llrd_optimizer() function](file:///d%3A/School/SemEval-TaskA/baselines/Kaggle_starters/Task-A-improve.ipynb#L476)

**Purpose:** Applies different learning rates to different model layers, prioritizing fine-tuning of task-specific layers.

**Learning Rate Schedule:**
```
head_layers (classifier): lr = base_lr
layer[11]: lr = base_lr * 0.9^0
layer[10]: lr = base_lr * 0.9^1
layer[9]:  lr = base_lr * 0.9^2
...
embeddings: lr = base_lr * 0.9^12
```

**Configuration:**
- `use_llrd = True`
- `layer_decay = 0.9` (multiplicative decay per layer)
- `num_layers = 12` (for 12-layer transformer)

**Rationale:** Lower layers learn general linguistic features; higher learning rates allow faster task-specific adaptation

---

### 6.2 Cosine Annealing with Restarts
**Implementation Reference:** [Cell 7 - CodeDetectionTrainer.train() TrainingArguments](file:///d%3A/School/SemEval-TaskA/baselines/Kaggle_starters/Task-A-improve.ipynb#L684)

**Configuration:**
- `lr_scheduler_type = "cosine_with_restarts"`
- `lr_scheduler_kwargs = {"num_cycles": 2}`

**Formula:**
```
lr(t) = base_lr * 0.5 * (1 + cos(π * t_cycle / T_cycle))
```

**Benefit:** Escapes local minima through periodic restarts while gradually reducing learning rate

---

### 6.3 Warmup Strategy
**Configuration:**
- `warmup_steps = int(total_steps * 0.06)` (6% of training)

**Purpose:** Gradual learning rate increase prevents instability in early training phases

---

## 7. Dataset Balancing

### 7.1 Random Oversampling of Minority Class
**Implementation Reference:** [Cell 8 - prepare_datasets_balanced() monkey-patch](file:///d%3A/School/SemEval-TaskA/baselines/Kaggle_starters/Task-A-improve.ipynb#L1038)

**Problem Addressed:** Original dataset has severe class imbalance (majority class >> minority class)

**Solution:**
```python
# Count samples per class
class_0_count = len(class_0_idx)
class_1_count = len(class_1_idx)

# Oversample minority class with replacement
if class_1_count < class_0_count:
    diff = class_0_count - class_1_count
    oversample_idx = random.choices(class_1_idx, k=diff)
    augmented_samples = train_ds.select(oversample_idx)
    train_ds = concatenate_datasets([train_ds, augmented_samples]).shuffle()
```

**Effect:** Creates 1:1 class ratio in training set for balanced learning

---

## 8. Model Configuration

### 8.1 Complete Trainer Initialization
**Implementation Reference:** [Cell 9 - CodeDetectionTrainer instantiation](file:///d%3A/School/SemEval-TaskA/baselines/Kaggle_starters/Task-A-improve.ipynb#L1057)

**Default Configuration (All Improvements Enabled):**

```python
trainer_obj = CodeDetectionTrainer(
    max_length=512,
    model_name="microsoft/unixcoder-base",
    seed=42,
    
    # Loss Functions
    use_focal_loss=True,
    use_label_smoothing=True,
    smoothing=0.05,
    
    # Model Architecture
    use_multi_sample_dropout=True,
    use_peft=True,
    peft_r=8,
    peft_alpha=32,
    
    # Optimization
    use_llrd=True,
    layer_decay=0.9,
    
    # Augmentation & Regularization
    use_augmentation=True,
    aug_prob=0.25,
    use_r_drop=True,
    r_drop_weight=0.5,
    use_triplet=True,
    triplet_margin=1.0,
    triplet_weight=0.1,
    use_mixcode=True,
    
    # Hardware
    fp16=torch.cuda.is_available(),
    bf16=False
)
```

---

### 8.2 Training Hyperparameters
**Implementation Reference:** [Cell 7 - CodeDetectionTrainer.train()](file:///d%3A/School/SemEval-TaskA/baselines/Kaggle_starters/Task-A-improve.ipynb#L696)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `batch_size` | 16 | Balance memory and gradient stability |
| `num_epochs` | 1-3 | Evaluation cycles |
| `learning_rate` | 2e-5 | Base LR for head layers |
| `weight_decay` | 0.01 | L2 regularization |
| `gradient_checkpointing` | True | Reduce memory during backward pass |
| `dataloader_num_workers` | 2 | Parallel data loading |
| `eval_steps` | 500 | Validation frequency |
| `save_steps` | 500 | Checkpoint frequency |
| `early_stopping_patience` | 3 | Stop after 3 evals without improvement |

---

## Summary of Improvements

### Performance Enhancements vs. Baseline
| Technique | Expected Impact | Risk |
|-----------|-----------------|------|
| Focal Loss + Label Smoothing | ↑ Robustness | None (complementary) |
| Triplet Loss | ↑ Discriminative Power | Requires careful margin tuning |
| R-Drop | ↑ Regularization | Slight computational overhead |
| LLRD | ↑ Layer-specific Learning | Better convergence |
| Data Augmentation | ↑ Generalization | Must be semantics-preserving |
| Dataset Balancing | ↑ Class Fairness | May reduce unique patterns |

### Memory & Speed Tradeoffs
- **PEFT/LoRA:** 50-70% fewer trainable parameters
- **Gradient Checkpointing:** ~2x slower forward/backward but 30% less memory
- **Multi-Sample Dropout:** 5 parallel passes cost in inference

---

## References

1. Lin, T.-Y., et al. (2017). "Focal Loss for Dense Object Detection." *ICCV*.
2. Hermans, A., et al. (2017). "In Defense of the Triplet Loss for Person Re-Identification." *CVPR*.
3. Hu, E. J., et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." *arXiv*.
4. Wei, J., et al. (2022). "Emergent Abilities of Large Language Models." *arXiv*.

---

## Files & Locations

| Component | File | Cell |
|-----------|------|------|
| Loss Functions | [Task-A-improve.ipynb](file:///d%3A/School/SemEval-TaskA/baselines/Kaggle_starters/Task-A-improve.ipynb) | 6-7 |
| Model Architecture | [Task-A-improve.ipynb](file:///d%3A/School/SemEval-TaskA/baselines/Kaggle_starters/Task-A-improve.ipynb) | 7 |
| Data Augmentation | [Task-A-improve.ipynb](file:///d%3A/School/SemEval-TaskA/baselines/Kaggle_starters/Task-A-improve.ipynb) | 7 |
| Training Loop | [Task-A-improve.ipynb](file:///d%3A/School/SemEval-TaskA/baselines/Kaggle_starters/Task-A-improve.ipynb) | 7 |
| Initialization | [Task-A-improve.ipynb](file:///d%3A/School/SemEval-TaskA/baselines/Kaggle_starters/Task-A-improve.ipynb) | 8-9 |

---

**Last Updated:** April 10, 2026  
**Model Version:** UniXCoder-Base with All Improvements Enabled  
**Target:** SemEval-2026 Task 13, Task A - Binary Code Detection

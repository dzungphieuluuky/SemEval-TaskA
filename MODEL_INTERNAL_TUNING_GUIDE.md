# Model Internal Tuning Guide (GraphCodeBERT + RobustTrainer)

This guide explains how to tune internal network behavior in your training notebook, not just top-level hyperparameters.

Scope:
- Notebook pipeline in Cell 9 and Cell 15.
- Task A binary detection objective with Macro F1 focus.
- Internal controls in model, loss, augmentation, trainer logic, and checkpoint fusion.

## 1) Mental Model: Where Parameters Actually Live

There are 4 layers of control in your notebook:

1. Data and augmentation behavior
- Functions/classes: `preprocess_code`, `CodeAugmentation`, `load_datasets`
- Effects: input distribution, token patterns, robustness signals

2. Model graph and embeddings
- Objects: `AutoModelForSequenceClassification`, `model.roberta.embeddings.word_embeddings`
- Effects: representational capacity and what gradients modify

3. Trainer loss mechanics
- Class: `RobustTrainer.compute_loss`
- Effects: decision boundary shape (R-Drop, MixCode, frequency consistency, smoothing)

4. Optimization and schedule behavior
- Objects: `TrainingArguments`, SWA/FGM in `RobustTrainer.training_step`
- Effects: training trajectory, stability, convergence speed

If you want to "touch internals", most power is in (3) and (4).

## 2) Fast Map of the Most Important Knobs

Primary knobs you already sweep:
- `batch_size`
- `learning_rate`
- `r_drop_alpha`
- `label_smoothing`
- `aug_rename_prob`
- `aug_format_prob`
- `mixup_alpha`
- `low_pass_keep_ratio`
- `freq_consistency_weight`

How they act internally:
- `r_drop_alpha`: weight on KL consistency between two stochastic forwards.
- `label_smoothing`: softens target distribution in CE-style losses.
- `mixup_alpha`: Beta distribution shape for embedding-space interpolation.
- `low_pass_keep_ratio`: how aggressively high-frequency embedding components are removed.
- `freq_consistency_weight`: KL penalty between original logits and low-pass logits.

## 3) Exact Internal Flow During One Training Step

In your `RobustTrainer.compute_loss`:

1. Labels are removed from `inputs`.
2. If `loss_type == "r-drop"` and train mode:
- Two forward passes
- CE average + symmetric KL term scaled by `r_drop_alpha`
3. Else if MixCode enabled (`mixup_alpha > 0`):
- Build mixed embeddings from shuffled batch
- Train with soft mixed labels
4. Else standard forward + selected base loss
5. If frequency consistency enabled:
- Build low-pass embeddings via FFT
- Forward again
- Add KL penalty scaled by `freq_consistency_weight`

In `training_step`:
- Base backward pass
- Optional FGM adversarial perturbation on embedding weights
- Optional SWA updates

This is the core place to customize behavior.

## 4) How to Add a New Internal Parameter Correctly

Use this exact propagation path whenever adding a new tuning parameter:

1. Add field to `TrainConfig`
2. Pass it into `RobustTrainer(...)`
3. Store it in `RobustTrainer.__init__`
4. Use it in `compute_loss` or `training_step`
5. Add it to checkpoint config logging (`ConfigLoggingCallback`)
6. Add candidate values in `PARAM_GRID`

If you miss step 5 or 6, experiment traceability becomes weak.

## 5) Touching Weights Directly (When You Need Low-Level Control)

Read or inspect parameters:

```python
for name, p in model.named_parameters():
    if "classifier" in name:
        print(name, p.shape, p.requires_grad, p.data.norm().item())
```

Freeze or unfreeze selective blocks:

```python
for name, p in model.named_parameters():
    p.requires_grad = ("classifier" in name or "layer.11" in name)
```

Apply manual parameter scaling (advanced):

```python
with torch.no_grad():
    for name, p in model.named_parameters():
        if "classifier" in name:
            p.mul_(0.95)
```

Important:
- Do direct weight edits only before optimizer step boundaries.
- Keep a checkpoint before manual interventions.

## 6) Safe Pattern for New Loss Terms

When adding a custom term, follow this formula:

```python
total_loss = base_loss + lambda_new * new_term
```

Best practice checklist:
- Keep `new_term` differentiable.
- Put its weight in config (do not hardcode).
- Log all components (`base_loss`, `new_term`, `total_loss`) for debugging.
- Start with very small weight, then scale.

## 7) Tuning Strategy That Usually Works

Phase A: stabilize
- Lower LR
- Moderate smoothing
- Disable extra penalties first

Phase B: regularize
- Enable one mechanism at a time: R-Drop, then MixCode, then frequency consistency
- Increase penalty weights gradually

Phase C: combine
- Run constrained grid (you already cap under 300)
- Keep seed fixed for fair comparisons
- Compare by Macro F1 and per-language behavior

## 8) Understanding Your LMC Fusion Cell

Your Cell 15 does linear weight interpolation across trained checkpoints:

- Start from model A state dict
- For each next model B:
  - `W <- (1 - alpha) * W + alpha * W_B`
- Save fused model

`INTERPOLATION_CONTROL` controls alpha.

Practical advice:
- Alpha near 0.3 to 0.7 is usually the first search band.
- Evaluate fused checkpoint separately; fusion is not guaranteed to beat best single run.

## 9) Common Failure Modes and Fixes

1. Training becomes unstable
- Lower `learning_rate`
- Reduce `r_drop_alpha`
- Reduce `freq_consistency_weight`

2. Loss drops but Macro F1 does not improve
- Lower smoothing
- Rebalance augmentation probabilities
- Check whether mixup is too aggressive

3. Fused model underperforms
- Fuse only top-K runs by validation Macro F1
- Tune `INTERPOLATION_CONTROL`
- Ensure same architecture and tokenizer across runs

## 10) Suggested Next Upgrades

1. Add per-run validation after 100 steps and store results in a CSV.
2. Rank runs by Macro F1 before LMC fusion.
3. Fuse only top-N runs instead of all runs.
4. Add interpolation sweep over alpha values for the final fused model.

---

If you want, I can add a follow-up notebook cell that automatically:
- evaluates each run on validation,
- selects top-K checkpoints,
- performs alpha sweep for LMC,
- outputs one final best fused model path.

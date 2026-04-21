# Experiment Design Guide: Binary Machine-Generated Code Detection (Task A)

This guide outlines a structured approach to designing, executing, and reporting experiments for your NeurIPS/SemEval-2026 paper, based on your `GraphCodeBERT` implementation.

## 1. Baseline Establishment
Before demonstrating the effectiveness of your advanced techniques, you need a solid baseline.
* **Experiment 1.0 (Naive Baseline)**: Train `microsoft/graphcodebert-base` using standard Cross-Entropy loss without any augmentations, SWA, R-Drop, FGM, or FFT.
* **Experiment 1.1 (Frozen vs. Unfrozen Base)**: Compare freezing the base model (your current approach) vs. full fine-tuning to validate your claim that freezing prevents catastrophic forgetting of pre-trained structural knowledge.

## 2. Ablation Studies
To prove that each component of your complex pipeline adds value, perform a leave-one-out or add-one-in ablation study. Track the **Macro F1-score** for each.
* **+ Structural Data Augmentation**: Add formatting and renaming permutations.
* **+ MixCode**: Add soft-labeling and mixup embeddings.
* **+ FGM (Adversarial Training)**: Add gradient perturbations ($\epsilon = 0.5$).
* **+ R-Drop**: Add symmetric KL-divergence ($\alpha = 6.0$).
* **+ FFT Low-Pass Filter**: Add frequency-domain consistency.
* **+ SWA**: Apply Stochastic Weight Averaging starting at Epoch 2.

*Recommendation*: Create a table in `report.tex` showing the incremental gain in Macro F1-score as each technique is added.

## 3. Generalization Testing (Crucial for Task A)
The task specifically requires generalizing from training languages (C++, Python, Java) to unseen languages (Go, PHP, C#, C, JS) and domains.
* **In-Distribution Evaluation**: Evaluate on a validation split containing only C++, Python, and Java.
* **Zero-Shot Cross-Lingual Evaluation**: Evaluate the frozen-base model on the unseen languages. Compare the drop in Macro F1-score between your robust pipeline and the naive baseline to prove your model generalizes better.

## 4. Hyperparameter Sensitivity (Optional but strong)
Show how sensitive your robust framework is to specific hyperparameters.
* **FFT Keep Ratio**: Test `keep_ratio` at $0.1, 0.25, 0.5, 0.75$. Does keeping too much high-frequency noise hurt generalization?
* **R-Drop Alpha**: Test $\alpha \in \{3.0, 4.0, 6.0, 8.0\}$.
* **Mixup Alpha**: Test different Beta distribution parameters for MixCode.

## 5. Integrating Results into `report.tex`

### Tables to Create
1. **Main Results Table**: Compare your final robust model against baselines (like standard RoBERTa, CodeBERT, or Kaggle starters) on the Macro F1 metric.
2. **Ablation Table**: A checklist table (using checkmarks $\checkmark$) showing components (MixCode, FGM, R-Drop, FFT, SWA) and the resulting Macro F1-score.
3. **Cross-Lingual Generalization Table**: Columns for each unseen language (Go, PHP, etc.) and rows for Baseline vs. Your Model.

### Figures to Create
1. **Loss/F1 Curves**: Show the validation Macro F1 curve with and without SWA to demonstrate how SWA stabilizes and flattens the minima.
2. **Feature Space t-SNE (Optional)**: A scatter plot showing how your FFT low-pass filter clusters AI-generated vs. human-written code more cleanly in the embedding space compared to the baseline.

## Checklist for Execution
- [ ] Ensure all experiments strictly optimize for **Macro F1-score**.
- [ ] Ensure NO external data or specialized AI-detection LLMs are used (Strict SemEval rules).
- [ ] Save validation logs and checkpoint configs (your `ConfigLoggingCallback` already handles this!) to populate the tables accurately.
# SemEval-2026 Task 13 Context: Task A

When interacting with this workspace, please adhere to the following constraints and context:

1. **Task A Focus Only**: This workspace is exclusively dedicated to **Task A (Binary Machine-Generated Code Detection)**. Ignore any context, data, or requirements related to Task B (Multi-class Authorship) and Task C (Hybrid Code Detection).
2. **Objective**: Classify code snippets as either fully human-written (label `0`) or fully machine-generated (label `1`).
3. **Evaluation Metric**: Optimize all training and validation pipelines for **Macro F1-score**. 
4. **Generalization Requirement**: Models are trained on C++, Python, and Java, but must generalize to unseen languages (Go, PHP, C#, C, JS) and domains (Research, Production). Factor zero-shot cross-lingual abilities into architectural decisions.
5. **Strict Rules**: 
   - DO NOT use external/additional training data.
   - DO NOT use models specifically pre-trained for AI-generated code detection. General-purpose code models (e.g., CodeBERT, StarCoder) are allowed.
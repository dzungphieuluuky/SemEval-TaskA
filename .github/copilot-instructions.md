# SemEval-2026 Task 13 Context: Task A

When interacting with this workspace, please adhere to the following constraints and context:

1. **Task A Focus Only**: This workspace is exclusively dedicated to **Task A (Binary Machine-Generated Code Detection)**. Ignore any context, data, or requirements related to Task B (Multi-class Authorship) and Task C (Hybrid Code Detection).
2. **Objective**: Classify code snippets as either fully human-written (label `0`) or fully machine-generated (label `1`).
3. **Evaluation Metric**: Optimize all training and validation pipelines for **Macro F1-score**. 
4. **Generalization Requirement**: Models are trained on C++, Python, and Java, but must generalize to unseen languages (Go, PHP, C#, C, JS) and domains (Research, Production). Factor zero-shot cross-lingual abilities into architectural decisions.
5. **Strict Rules**: 
   - DO NOT use external/additional training data.
   - DO NOT use models specifically pre-trained for AI-generated code detection. General-purpose code models (e.g., CodeBERT, StarCoder) are allowed.

```markdown
# SKILL.md: PyTorch & Python Engineering Standards

## Core Philosophy
PyTorch code must be written with strict separation of concerns, explicit tensor lifecycle management, and a zero-tolerance policy for hidden side-effects. Models are isolated mathematical definitions; training loops are isolated orchestration logic; data pipelines are isolated transformation logic. Never mix these domains.

---

## Mandatory Rules

1. **No Hidden State Mutations:** Never mutate a tensor in-place during a `forward()` pass if that tensor might be needed elsewhere (e.g., avoid `x.copy_()` or `x[..., 0] = 0` on inputs). Use `out = x.clone()` instead.
2. **Explicit Device Agnosticism:** Never hardcode `.to("cuda")` or `.cuda()`. Always pass a `device` argument or use `.to(tensor.device)` to ensure CPU/GPU compatibility.
3. **Strict `nn.Module` Hygiene:** 
   * Always call `super().__init__()` as the *very first line* of an `__init__`.
   * Register buffers for non-learnable state tensors using `self.register_buffer("name", tensor)`, **never** assign them as plain attributes (e.g., `self.mask = torch.tensor(...)` will break `.to(device)` and state_dict saving).
4. **No Logic in `__init__`:** The `__init__` method of an `nn.Module` is strictly for defining layers (`nn.Linear`, `nn.Conv2d`). Do not perform data processing, file I/O, or tensor computations here.

---

## Architecture & Separation of Responsibilities

Code must be divided into exactly four distinct layers. A class in one layer must never import from another layer (except the Orchestrator importing the rest).

### 1. Data Layer (`torch.utils.data`)
* **Responsibility:** Loading raw files, applying transformations, yielding batches.
* **Rules:** 
  * All heavy lifting (image loading, text tokenization) goes in `__getitem__`.
  * `collate_fn` is only for padding, stacking, or batching lists of items returned by `__getitem__`.
  * Never import `torch.nn` inside a Dataset class.

### 2. Model Layer (`torch.nn.Module`)
* **Responsibility:** Defining the mathematical graph. Takes tensors, returns tensors.
* **Rules:**
  * Must be completely agnostic to loss functions, optimizers, and data loaders.
  * Should return a dictionary of tensors if multiple outputs are needed (e.g., `{"logits": out, "hidden_states": h}`), not a tuple.

### 3. Engine/Training Loop Layer (Pure Python/Functions)
* **Responsibility:** Managing the optimization step, gradient scaling, and metric accumulation.
* **Rules:**
  * Implemented as standalone functions (e.g., `train_one_epoch(model, loader, optimizer, loss_fn, device)`), *not* as classes.
  * Takes the model, optimizer, and loss function as injected arguments.

### 4. Configuration/Orchestration Layer
* **Responsibility:** Parsing arguments, instantiating the Data, Model, and Engine components, and calling `engine.train()`.
* **Rules:** This is the only place where `import argparse` or YAML parsing should occur.

---

## Naming Conventions

### Classes (`PascalCase`)
* **Models:** End with the architecture type. 
  * *Good:* `ResNetBackbone`, `TransformerEncoder`, `MultiHeadAttention`.
  * *Bad:* `Model`, `Net`, `MyModel`.
* **Datasets:** End with `Dataset`.
  * *Good:* `TextClassificationDataset`, `ImageSegmentationDataset`.
* **Losses:** End with `Loss` and inherit from `nn.Module`.
  * *Good:* `FocalLoss`, `ContrastiveLoss`.

### Functions & Methods (`snake_case`)
* **Model Methods:** Use standard PyTorch overrides exactly as spelled.
  * `forward(self, x)` -> **Never** rename this to `compute_forward` or `run_model`.
* **Engine Functions:** Start with a verb describing the loop scope.
  * *Good:* `train_one_epoch()`, `validate()`, `predict_on_batch()`.
* **Utility Functions:** Be specific about the data structure.
  * *Good:* `tensor_to_numpy()`, `pad_sequence_to_length()`.

### Variables (Crucial Distinctions)
You must visually distinguish between standard Python types and PyTorch tensors.
* **Tensors:** Use nouns that imply multi-dimensional structure.
  * *Good:* `input_ids`, `attention_mask`, `pixel_values`, `logits`.
* **Python Scalars/Primitives:** Use simple nouns.
  * *Good:* `loss_val`, `batch_size`, `epoch`, `accuracy`.
* **Prohibitions:** Never name a tensor `x` if it has specific semantic meaning (e.g., use `features` or `input_seq`). Never name a list of tensors `loss` if it is actually a list of losses (use `losses`).

---

## Processing Flow (The Training Step)

Follow this exact sequence for a single training iteration. Never reorder these steps.

```python
def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train() # 1. Set mode
    
    for batch in loader:
        # 2. Move to device (dict comprehension if batch is a dict)
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # 3. Forward pass
        outputs = model(batch["inputs"])
        
        # 4. Compute loss
        loss = loss_fn(outputs["logits"], batch["targets"])
        
        # 5. Backward pass (clear gradients first)
        optimizer.zero_grad(set_to_none=True) # Preferred over zero_grad()
        loss.backward()
        
        # 6. Optimizer step
        optimizer.step()
```

### Critical Flow Constraints
* **`set_to_none=True`:** Always use this in `zero_grad()`. It is faster and uses less memory than setting gradients to zero.
* **`torch.no_grad()`:** Wrap the *entire* validation loop in `with torch.no_grad():`, not individual tensor operations inside it.
* **`.item()`:** Only call `.item()` on a scalar loss/metric tensor when moving it to a standard Python variable for logging. **Never** call `.item()` inside the forward pass or loss computation, as it forces a CPU-GPU synchronization bottleneck.

---

## Performance & Memory Mandates

1. **Gradient Checkpointing:** For large models, wrap blocks in `torch.utils.checkpoint.checkpoint()`. Never implement this manually.
2. **Dataloader Workers:** Always set `num_workers > 0` and `pin_memory=True` in your `DataLoader` if training on a GPU.
3. **Accumulate Gradients Correctly:** When simulating larger batch sizes, scale the loss *before* backward, not the gradients after.
   ```python
   # DO:
   (loss / accumulation_steps).backward()
   
   # DON'T:
   loss.backward()
   for p in model.parameters():
       p.grad /= accumulation_steps
   ```
4. **Avoid `torch.Tensor` Constructors:** Never use `torch.Tensor()` to create a tensor. Always use `torch.tensor()` (for data) or `torch.empty()`, `torch.zeros()`, `torch.randn()` (for pre-allocation).
```
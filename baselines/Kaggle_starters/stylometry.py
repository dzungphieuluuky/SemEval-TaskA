"""
stylometry.py — Transformer-Based Code Stylometry for LLM Authorship Attribution
==================================================================================

Implements the CodeT5-Authorship / CodeT5-JSA architecture family, as introduced in:

  • Bisztray et al. (2025) "I Know Which LLM Wrote Your Code Last Summer"
    ACM AISec 2025. arXiv:2506.17323
    → Binary / multi-class attribution of C programs to specific LLMs.
    → CodeT5+ encoder-only + 2-layer GELU head → 97.56% binary, 95.40% 5-class.

  • Gurioli et al. (2024) "Is This You, LLM? Recognizing AI-written Programs
    with Multilingual Code Stylometry". arXiv:2412.14611
    → Single classifier across 10 programming languages → 84.1% accuracy.
    → CodeT5+ encoder + [CLS] token + ReLU head (Linear → ReLU → Dropout → Linear).

  • Dubniczky et al. (2025) "The Hidden DNA of LLM-Generated JavaScript"
    arXiv:2510.10493
    → CodeT5-JSA variant on 20 LLMs, 50k Node.js programs.
    → 95.8% (5-class) / 94.6% (10-class) / 88.5% (20-class).

Architecture summary
--------------------
  CodeT5+ (encoder-decoder, 220M or 770M)
        │  decoder discarded; only encoder weights loaded
        ↓
  Encoder hidden states  [batch, seq_len, hidden]
        │  take [CLS] / first token  → [batch, hidden]
        ↓
  StylometryHead  (two-layer MLP)
     Linear(hidden → hidden//2)
     Activation  (GELU for Bisztray variant; ReLU for Gurioli variant)
     Dropout(0.20)
     Linear(hidden//2 → num_classes)
        ↓
  logits  [batch, num_classes]

Two task modes
--------------
  "binary"       – human vs. AI  (num_classes = 2)
  "attribution"  – which LLM produced this code  (num_classes = N generators)

Both modes use the same architecture; only the label set and loss differ.

Usage example
-------------
    trainer = CodeStylometryTrainer(
        task_mode="attribution",       # "binary" | "attribution"
        author_names=["gpt-4o", "gpt-4.1", "claude-3.5-haiku",
                      "gemini-2.5-flash", "llama-3.3"],
        encoder_name="Salesforce/codet5p-220m",   # or codet5p-770m for JSA
        activation="gelu",             # "gelu" (Bisztray) | "relu" (Gurioli)
        preserve_comments=True,        # set False for ablation
    )
    trainer.run_full_pipeline(output_dir="./stylometry_results")
"""

from __future__ import annotations

import logging
import os
import re
import warnings
from typing import Dict, List, Optional, Tuple

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
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    PreTrainedModel,
    T5EncoderModel,
    Trainer,
    TrainingArguments,
)

warnings.filterwarnings("ignore")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Classification head
# ──────────────────────────────────────────────────────────────────────────────

class StylometryHead(nn.Module):
    """
    Two-layer MLP classification head, as used in both CodeT5-Authorship
    (GELU activation) and the Gurioli multilingual variant (ReLU activation).

    Architecture (from the paper's Figure 3):
        Linear(hidden_size → hidden_size // 2)
        Activation (GELU or ReLU)
        Dropout(dropout_rate)
        Linear(hidden_size // 2 → num_classes)
    """

    def __init__(
        self,
        hidden_size: int,
        num_classes: int,
        dropout_rate: float = 0.20,
        activation: str = "gelu",   # "gelu" | "relu"
    ):
        super().__init__()
        intermediate = hidden_size // 2
        self.fc1 = nn.Linear(hidden_size, intermediate)
        self.act = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(intermediate, num_classes)

        # Initialise weights (Glorot uniform, bias zero)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, cls_repr: torch.Tensor) -> torch.Tensor:
        x = self.act(self.fc1(cls_repr))
        x = self.dropout(x)
        return self.fc2(x)


# ──────────────────────────────────────────────────────────────────────────────
# Encoder-only CodeT5+ model
# ──────────────────────────────────────────────────────────────────────────────

class CodeT5StyleModel(nn.Module):
    """
    CodeT5+ encoder-only model for code stylometry.

    Loads only the encoder layers from a CodeT5+ checkpoint, discarding the
    decoder entirely (as done in CodeT5-Authorship and CodeT5-JSA).  The
    representation of the first token ([CLS] / <s>) is fed to a two-layer
    classification head.

    Supported checkpoints (HuggingFace):
        "Salesforce/codet5p-220m"   – 220M, fast, good for 10-class tasks
        "Salesforce/codet5p-770m"   – 770M, best accuracy (JSA variant)
    """

    def __init__(
        self,
        encoder_name: str,
        num_classes: int,
        dropout_rate: float = 0.20,
        activation: str = "gelu",
    ):
        super().__init__()
        logger.info(f"Loading encoder from {encoder_name} (decoder discarded)...")
        # T5EncoderModel loads only the encoder stack
        self.encoder = T5EncoderModel.from_pretrained(encoder_name)
        hidden_size = self.encoder.config.d_model
        self.head = StylometryHead(
            hidden_size=hidden_size,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            activation=activation,
        )
        self.num_classes = num_classes
        logger.info(
            f"CodeT5StyleModel ready: hidden={hidden_size}, "
            f"classes={num_classes}, activation={activation}"
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # Encode; output shape: [batch, seq_len, hidden]
        enc_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Take the first token representation (CLS / <s>)
        cls_repr = enc_out.last_hidden_state[:, 0, :]   # [batch, hidden]
        logits = self.head(cls_repr)                      # [batch, num_classes]

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return {"loss": loss, "logits": logits}


# ──────────────────────────────────────────────────────────────────────────────
# Custom Trainer (wraps HuggingFace Trainer for our non-standard model)
# ──────────────────────────────────────────────────────────────────────────────

class StylometryTrainerHF(Trainer):
    """
    Thin Trainer subclass that correctly handles the dict output of
    CodeT5StyleModel (which is not a HuggingFace ModelOutput).
    """

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict,
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ):
        labels = inputs.pop("labels")
        outputs = model(**inputs, labels=labels)
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None,
    ):
        """Override so that evaluation correctly extracts logits from dict."""
        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            if has_labels:
                labels = inputs["labels"]
                outputs = model(**inputs)
                loss = outputs["loss"].mean().detach()
                logits = outputs["logits"].detach()
            else:
                labels = None
                outputs = model(**inputs)
                loss = None
                logits = outputs["logits"].detach()

        if prediction_loss_only:
            return (loss, None, None)

        labels = labels.detach() if labels is not None else None
        return (loss, logits, labels)


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing / comment ablation
# ──────────────────────────────────────────────────────────────────────────────

class StylometryPreprocessor:
    """
    Code preprocessing for stylometry experiments.

    Gurioli et al. (2024) showed that snippet length is a noticeable
    distinguishing feature between AI and human code.
    Bisztray et al. (2025) showed comment phrasing is the richest single
    stylometric signal (removing comments drops accuracy by 7.2 pp.).

    This class supports:
      - Standard normalisation (always applied)
      - Comment preservation / stripping (ablation flag)
      - Language tag injection (prepend `# lang: python` etc.) to help the
        single multilingual classifier condition on language identity
    """

    # Single-line comment markers per language
    _COMMENT_MARKERS: Dict[str, str] = {
        "python": "#",
        "javascript": "//",
        "typescript": "//",
        "java": "//",
        "c": "//",
        "cpp": "//",
        "go": "//",
        "kotlin": "//",
        "ruby": "#",
        "rust": "//",
        "php": "//",
        "swift": "//",
        "scala": "//",
    }

    def __init__(
        self,
        preserve_comments: bool = True,
        inject_language_tag: bool = True,
    ):
        self.preserve_comments = preserve_comments
        self.inject_language_tag = inject_language_tag

    def strip_comments(self, code: str, language: str) -> str:
        """Remove single-line comments for a given language."""
        marker = self._COMMENT_MARKERS.get(language.lower(), "#")
        pattern = rf"(?m)(?<!['\"])({re.escape(marker)}.*)$"
        return re.sub(pattern, "", code)

    def strip_multiline_comments(self, code: str) -> str:
        """Remove /* … */ and /** … */ style block comments."""
        return re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)

    def normalize(self, code: str) -> str:
        """Baseline normalisation: line endings, excess whitespace."""
        code = code.lstrip("\ufeff\u200b")
        code = re.sub(r"\r\n|\r", "\n", code)
        code = "\n".join(line.rstrip() for line in code.split("\n"))
        code = re.sub(r"\n{3,}", "\n\n", code)
        code = re.sub(r"[ \t]+", " ", code)
        return code.strip()

    def process(self, code: str, language: str = "python") -> str:
        code = self.normalize(code)
        if not self.preserve_comments:
            code = self.strip_comments(code, language)
            code = self.strip_multiline_comments(code)
        if self.inject_language_tag:
            # Prepend a language hint so the multilingual model can condition
            # on it (consistent with the H-AIRosettaMP experimental setup)
            code = f"# lang: {language.lower()}\n" + code
        return code


# ──────────────────────────────────────────────────────────────────────────────
# High-level trainer orchestrating everything
# ──────────────────────────────────────────────────────────────────────────────

class CodeStylometryTrainer:
    """
    End-to-end trainer for transformer-based code stylometry.

    Implements the CodeT5-Authorship / CodeT5-JSA pipeline:

    Task modes
    ──────────
    "binary"
        Human-written vs. AI-generated detection.
        label 0 = human, label 1 = AI (or derive from dataset 'label' column).

    "attribution"
        Model-level attribution: predict *which* LLM generated the code.
        Requires `author_names` list; each entry is a generator name that maps
        to an integer class index (alphabetically sorted for reproducibility).

    Key design decisions (from the papers)
    ────────────────────────────────────────
    • Encoder-only CodeT5+: decoder discarded (Bisztray 2025, Fig. 3).
    • CLS / first-token representation fed to the head.
    • Two-layer MLP head with 20% dropout.
    • GELU for fine-grained attribution (Bisztray); ReLU for binary/multilingual
      (Gurioli). Configurable via `activation`.
    • Slanted triangular (warmup + cosine) LR schedule, gradual weight decay,
      consistent with best practices for encoder fine-tuning (Howard & Ruder 2018).
    • Language tag injection for the multilingual setting (Gurioli 2024).
    • Comment ablation supported as an experimental switch (Bisztray 2025).
    """

    TASK_MODES = {"binary", "attribution"}

    def __init__(
        self,
        # ── Task ──────────────────────────────────────────────────────────
        task_mode: str = "attribution",
        author_names: Optional[List[str]] = None,  # required for "attribution"
        # ── Model ─────────────────────────────────────────────────────────
        encoder_name: str = "Salesforce/codet5p-220m",
        activation: str = "gelu",          # "gelu" (Bisztray) | "relu" (Gurioli)
        dropout_rate: float = 0.20,
        max_length: int = 512,
        # ── Preprocessing ─────────────────────────────────────────────────
        preserve_comments: bool = True,    # set False for ablation study
        inject_language_tag: bool = True,  # multilingual conditioning
        # ── Training ──────────────────────────────────────────────────────
        seed: int = 42,
        fp16: bool = False,
        bf16: bool = False,
    ):
        if task_mode not in self.TASK_MODES:
            raise ValueError(f"task_mode must be one of {self.TASK_MODES}")
        if task_mode == "attribution" and not author_names:
            raise ValueError("author_names must be provided for attribution mode")

        self.task_mode = task_mode
        self.author_names = sorted(author_names) if author_names else None
        self.encoder_name = encoder_name
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.max_length = max_length
        self.seed = seed
        self.fp16 = fp16
        self.bf16 = bf16

        self.preprocessor = StylometryPreprocessor(
            preserve_comments=preserve_comments,
            inject_language_tag=inject_language_tag,
        )

        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[CodeT5StyleModel] = None
        self.trainer: Optional[StylometryTrainerHF] = None
        self._author2id: Optional[Dict[str, int]] = None
        self.num_classes: Optional[int] = None

    # ──────────────────────────────────────────────────────────────────────
    # Label mapping helpers
    # ──────────────────────────────────────────────────────────────────────

    def _build_author_map(self, dataset: Dataset) -> None:
        """Build author→id mapping from dataset or from author_names list."""
        if self.task_mode == "attribution":
            if self.author_names:
                self._author2id = {name: i for i, name in enumerate(self.author_names)}
            else:
                # Infer from dataset
                names = sorted(set(dataset["generator"]))
                self._author2id = {name: i for i, name in enumerate(names)}
                self.author_names = names
            self.num_classes = len(self._author2id)
            logger.info(f"Attribution map ({self.num_classes} classes): {self._author2id}")
        else:
            # Binary: use existing 'label' column (0/1)
            self.num_classes = 2
            logger.info("Binary mode: 0=human, 1=AI")

    def _encode_label(self, example: Dict) -> Dict:
        """Map generator string → integer class for attribution mode."""
        if self.task_mode == "attribution":
            gen = example.get("generator", "unknown")
            example["labels"] = self._author2id.get(gen, -1)
        else:
            example["labels"] = int(example["label"])
        return example

    # ──────────────────────────────────────────────────────────────────────
    # Data loading
    # ──────────────────────────────────────────────────────────────────────

    def load_and_prepare_data(self) -> Tuple[Dataset, Dataset]:
        """Load SemEval-2026-Task13 (or any dataset with 'code'/'label' columns)."""
        logger.info("Loading datasets...")
        train_dataset = load_dataset("DaniilOr/SemEval-2026-Task13", "A", split="train")
        val_dataset = load_dataset("DaniilOr/SemEval-2026-Task13", "A", split="validation")

        required = {"code", "label"}
        missing = required - set(train_dataset.column_names)
        if missing:
            raise ValueError(f"Dataset missing required columns: {missing}")

        self._build_author_map(train_dataset)
        logger.info(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")
        return train_dataset, val_dataset

    # ──────────────────────────────────────────────────────────────────────
    # Tokenisation
    # ──────────────────────────────────────────────────────────────────────

    def _tokenize(self, examples: Dict) -> Dict:
        """Preprocess and tokenise a batch of code snippets."""
        lang_col = examples.get("language", ["python"] * len(examples["code"]))
        cleaned = [
            self.preprocessor.process(code, lang)
            for code, lang in zip(examples["code"], lang_col)
        ]
        return self.tokenizer(
            cleaned,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )

    def prepare_datasets(
        self, train_ds: Dataset, val_ds: Dataset
    ) -> Tuple[Dataset, Dataset]:
        """Label-encode and tokenise both splits."""
        logger.info("Encoding labels...")
        train_ds = train_ds.map(self._encode_label)
        val_ds = val_ds.map(self._encode_label)

        # Remove examples with unmapped labels (-1)
        train_ds = train_ds.filter(lambda x: x["labels"] >= 0)
        val_ds = val_ds.filter(lambda x: x["labels"] >= 0)

        cols_to_remove = [
            c for c in ["code", "generator", "language", "label"]
            if c in train_ds.column_names
        ]
        logger.info("Tokenising...")
        train_ds = train_ds.map(
            self._tokenize, batched=True,
            remove_columns=cols_to_remove, desc="Tokenising train"
        )
        val_ds = val_ds.map(
            self._tokenize, batched=True,
            remove_columns=cols_to_remove, desc="Tokenising val"
        )
        return train_ds, val_ds

    # ──────────────────────────────────────────────────────────────────────
    # Metrics  (macro F1 as primary, accuracy for comparability with papers)
    # ──────────────────────────────────────────────────────────────────────

    def compute_metrics(self, eval_pred) -> Dict:
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        macro_f1 = precision_recall_fscore_support(labels, preds, average="macro")[2]
        accuracy = accuracy_score(labels, preds)
        weighted_f1 = precision_recall_fscore_support(labels, preds, average="weighted")[2]
        return {
            "macro_f1": macro_f1,
            "accuracy": accuracy,
            "weighted_f1": weighted_f1,
        }

    # ──────────────────────────────────────────────────────────────────────
    # Model initialisation
    # ──────────────────────────────────────────────────────────────────────

    def initialize_model_and_tokenizer(self) -> None:
        logger.info(f"Initialising tokenizer from {self.encoder_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.encoder_name)
        logger.info(f"Building CodeT5StyleModel ({self.num_classes} classes, "
                    f"activation={self.activation})...")
        self.model = CodeT5StyleModel(
            encoder_name=self.encoder_name,
            num_classes=self.num_classes,
            dropout_rate=self.dropout_rate,
            activation=self.activation,
        )
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Trainable parameters: {n_params:,}")

    # ──────────────────────────────────────────────────────────────────────
    # Training
    # ──────────────────────────────────────────────────────────────────────

    def train(
        self,
        output_dir: str = "./stylometry_results",
        num_epochs: int = 5,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
    ) -> StylometryTrainerHF:
        logger.info("=" * 60)
        logger.info(f"  Task mode  : {self.task_mode}")
        logger.info(f"  Num classes: {self.num_classes}")
        logger.info(f"  Encoder    : {self.encoder_name}")
        logger.info(f"  Activation : {self.activation}")
        logger.info("=" * 60)

        train_ds, val_ds = self.load_and_prepare_data()
        self.initialize_model_and_tokenizer()
        train_ds, val_ds = self.prepare_datasets(train_ds, val_ds)

        steps_per_epoch = max(1, len(train_ds) // batch_size)
        total_steps = num_epochs * steps_per_epoch
        # Slanted triangular schedule: ~6% warmup (Howard & Ruder 2018)
        warmup_steps = max(1, int(total_steps * 0.06))

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            learning_rate=learning_rate,
            weight_decay=0.01,
            warmup_steps=warmup_steps,
            lr_scheduler_type="cosine",
            # ── Evaluation & checkpointing ─────────────────────────────
            eval_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            save_total_limit=2,
            # ── Efficiency ────────────────────────────────────────────
            gradient_checkpointing=True,
            fp16=self.fp16,
            bf16=self.bf16,
            max_grad_norm=1.0,
            dataloader_num_workers=2,
            remove_unused_columns=False,
            report_to=[],
            seed=self.seed,
            logging_steps=20,
        )

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        self.trainer = StylometryTrainerHF(
            model=self.model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        logger.info(f"  Total steps  : {total_steps}")
        logger.info(f"  Warmup steps : {warmup_steps}")
        logger.info(f"  LR schedule  : cosine")
        logger.info("***** Starting stylometry training *****")
        self.trainer.train()
        logger.info(f"Training complete → {output_dir}")
        return self.trainer

    # ──────────────────────────────────────────────────────────────────────
    # Evaluation
    # ──────────────────────────────────────────────────────────────────────

    def evaluate(self, eval_dataset: Optional[Dataset] = None):
        """Evaluate and print per-class report with author names."""
        if self.trainer is None:
            logger.error("No trainer found. Run train() first.")
            return None

        logger.info("Evaluating stylometry model...")
        pred_out = self.trainer.predict(eval_dataset)
        y_pred = np.argmax(pred_out.predictions, axis=1)
        y_true = pred_out.label_ids

        target_names = self.author_names if self.author_names else None
        logger.info("\n***** Stylometry Classification Report *****")
        print(classification_report(y_true, y_pred, target_names=target_names, digits=4))

        # Per-class accuracy for fine-grained attribution analysis
        if self.task_mode == "attribution" and self.author_names:
            logger.info("\n***** Per-Author Attribution Accuracy *****")
            for cls_idx, name in enumerate(self.author_names):
                mask = y_true == cls_idx
                if mask.sum() == 0:
                    continue
                per_cls_acc = (y_pred[mask] == cls_idx).mean() * 100
                logger.info(f"  {name:30s} : {per_cls_acc:.1f}%  (n={mask.sum()})")

        return pred_out

    # ──────────────────────────────────────────────────────────────────────
    # Full pipeline
    # ──────────────────────────────────────────────────────────────────────

    def run_full_pipeline(
        self,
        output_dir: str = "./stylometry_results",
        num_epochs: int = 5,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
    ) -> StylometryTrainerHF:
        """Train + evaluate in one call."""
        try:
            self.train(output_dir, num_epochs, batch_size, learning_rate)
            logger.info(f"Best model saved to: {os.path.join(output_dir, 'best_model')}")
            return self.trainer
        except Exception as e:
            logger.error(f"Stylometry pipeline error: {e}")
            raise


# ──────────────────────────────────────────────────────────────────────────────
# Comment-ablation experiment (reproduces Bisztray 2025 Table 5 analysis)
# ──────────────────────────────────────────────────────────────────────────────

class CommentAblationExperiment:
    """
    Runs paired experiments WITH and WITHOUT comments to measure their
    contribution as a stylometric signal.

    Bisztray et al. (2025) found that removing comments from BERT dropped
    accuracy by 7.2 pp in multi-class attribution — making it the single
    richest stylometric feature.

    Usage:
        exp = CommentAblationExperiment(
            author_names=["gpt-4.1", "gpt-4o", "claude-3.5-haiku",
                          "gemini-2.5-flash", "llama-3.3"],
            encoder_name="Salesforce/codet5p-220m",
        )
        results = exp.run()
        # {'with_comments': {'accuracy': ..., 'macro_f1': ...},
        #  'without_comments': {'accuracy': ..., 'macro_f1': ...},
        #  'comment_delta_accuracy_pp': ...}
    """

    def __init__(
        self,
        author_names: List[str],
        encoder_name: str = "Salesforce/codet5p-220m",
        num_epochs: int = 5,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        seed: int = 42,
    ):
        self.author_names = author_names
        self.encoder_name = encoder_name
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed

    def _run_single(self, preserve_comments: bool, output_dir: str) -> Dict:
        trainer = CodeStylometryTrainer(
            task_mode="attribution",
            author_names=self.author_names,
            encoder_name=self.encoder_name,
            activation="gelu",
            preserve_comments=preserve_comments,
            seed=self.seed,
        )
        trainer.run_full_pipeline(
            output_dir=output_dir,
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
        )
        # Return final eval metrics
        metrics = trainer.trainer.evaluate()
        return {
            "accuracy": metrics.get("eval_accuracy", float("nan")),
            "macro_f1": metrics.get("eval_macro_f1", float("nan")),
        }

    def run(self) -> Dict:
        """Run both conditions and report delta."""
        logger.info("=" * 60)
        logger.info("Comment Ablation Experiment")
        logger.info("=" * 60)

        logger.info("Condition 1: WITH comments (preserve_comments=True)")
        with_res = self._run_single(
            preserve_comments=True, output_dir="./ablation_with_comments"
        )

        logger.info("Condition 2: WITHOUT comments (preserve_comments=False)")
        without_res = self._run_single(
            preserve_comments=False, output_dir="./ablation_without_comments"
        )

        delta = (with_res["accuracy"] - without_res["accuracy"]) * 100
        results = {
            "with_comments": with_res,
            "without_comments": without_res,
            "comment_delta_accuracy_pp": round(delta, 2),
        }
        logger.info("\n***** Ablation Results *****")
        logger.info(f"  With comments    : accuracy={with_res['accuracy']:.4f}")
        logger.info(f"  Without comments : accuracy={without_res['accuracy']:.4f}")
        logger.info(f"  Delta (pp)       : {delta:+.2f}")
        return results
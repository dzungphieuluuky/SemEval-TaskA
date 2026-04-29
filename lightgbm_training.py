"""
LightGBM Training Script for Machine-Generated Code Detection (Task A)
Features: ~20 handcrafted features based on code structure, style, and patterns
"""

import os
import re
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_fscore_support
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# 1. FEATURE EXTRACTION LAYER
# =============================================================================

class CodeFeatureExtractor:
    """Extract handcrafted features from code snippets."""
    
    def __init__(self):
        # Regex patterns
        self.comment_patterns = [
            r'#.*$',                      # Python comments
            r'//.*$',                     # C++/Java/JS comments
            r'/\*.*?\*/',                 # C-style block comments
        ]
        self.function_patterns = [
            r'\bdef\s+\w+\s*\(',          # Python def
            r'\bfunction\s+\w+\s*\(',     # JS function
            r'\w+\s+\w+\s*\([^)]*\)\s*{', # C++/Java function
        ]
        self.class_patterns = [
            r'\bclass\s+\w+',             # class definition
        ]
        self.control_keywords = [
            'if', 'else', 'elif', 'for', 'while', 'switch', 'case', 'break', 'continue',
            'try', 'except', 'catch', 'finally', 'throw', 'return'
        ]
        self.operators = ['+', '-', '*', '/', '%', '==', '!=', '<', '>', '<=', '>=', 
                         '&&', '||', '&', '|', '^', '<<', '>>', '=']
        
    def extract_features(self, code: str) -> Dict[str, float]:
        """Extract all features from a single code snippet."""
        try:
            return {
                # Basic structural features
                'line_count': self._line_count(code),
                'avg_line_length': self._avg_line_length(code),
                'max_line_length': self._max_line_length(code),
                'empty_line_ratio': self._empty_line_ratio(code),
                
                # Comment features
                'comment_count': self._count_comments(code),
                'comment_ratio': self._comment_ratio(code),
                
                # Function and class features
                'function_count': self._count_functions(code),
                'class_count': self._count_classes(code),
                
                # Token/Lexical features
                'token_count': self._token_count(code),
                'unique_tokens': self._unique_token_count(code),
                'token_diversity': self._token_diversity(code),
                'avg_token_length': self._avg_token_length(code),
                
                # Control flow features
                'control_flow_count': self._control_flow_count(code),
                'operator_count': self._operator_count(code),
                
                # Pattern features
                'has_docstring': self._has_docstring(code),
                'has_type_hints': self._has_type_hints(code),
                'import_count': self._import_count(code),
                
                # Stylistic features
                'indentation_consistency': self._indentation_consistency(code),
                'bracket_nesting_depth': self._bracket_nesting_depth(code),
                'code_entropy': self._code_entropy(code),
                
                # Length-based features
                'code_length': self._code_length(code),
                'code_length_normalized': self._code_length_normalized(code),
            }
        except Exception as e:
            # Return zeros if feature extraction fails
            return {k: 0.0 for k in [
                'line_count', 'avg_line_length', 'max_line_length', 'empty_line_ratio',
                'comment_count', 'comment_ratio', 'function_count', 'class_count',
                'token_count', 'unique_tokens', 'token_diversity', 'avg_token_length',
                'control_flow_count', 'operator_count', 'has_docstring', 'has_type_hints',
                'import_count', 'indentation_consistency', 'bracket_nesting_depth',
                'code_entropy', 'code_length', 'code_length_normalized'
            ]}
    
    # =========== Basic Structural Features ===========
    def _line_count(self, code: str) -> float:
        """Count total number of lines."""
        return float(len(code.split('\n')))
    
    def _avg_line_length(self, code: str) -> float:
        """Average length of non-empty lines."""
        lines = [l for l in code.split('\n') if l.strip()]
        if not lines:
            return 0.0
        return np.mean([len(l) for l in lines])
    
    def _max_line_length(self, code: str) -> float:
        """Maximum line length."""
        lines = code.split('\n')
        if not lines:
            return 0.0
        return float(max([len(l) for l in lines]))
    
    def _empty_line_ratio(self, code: str) -> float:
        """Ratio of empty lines."""
        lines = code.split('\n')
        if not lines:
            return 0.0
        empty = sum(1 for l in lines if not l.strip())
        return empty / len(lines)
    
    # =========== Comment Features ===========
    def _count_comments(self, code: str) -> float:
        """Count comment lines."""
        count = 0
        for line in code.split('\n'):
            stripped = line.strip()
            if stripped.startswith('#') or stripped.startswith('//'):
                count += 1
        return float(count)
    
    def _comment_ratio(self, code: str) -> float:
        """Ratio of comment lines to total lines."""
        total = len(code.split('\n'))
        if total == 0:
            return 0.0
        comments = self._count_comments(code)
        return comments / total
    
    # =========== Function/Class Features ===========
    def _count_functions(self, code: str) -> float:
        """Count function definitions."""
        count = 0
        for pattern in self.function_patterns:
            count += len(re.findall(pattern, code, re.MULTILINE | re.IGNORECASE))
        # Remove duplicates by being more specific for each language
        count = max(
            len(re.findall(r'\bdef\s+\w+\s*\(', code)),  # Python
            len(re.findall(r'\bfunction\s+\w+\s*\(', code)),  # JS
            len(re.findall(r'\w+\s+\w+\s*\([^)]*\)\s*{', code))  # C++/Java
        )
        return float(count)
    
    def _count_classes(self, code: str) -> float:
        """Count class definitions."""
        count = len(re.findall(r'\bclass\s+\w+', code, re.IGNORECASE))
        return float(count)
    
    # =========== Lexical/Token Features ===========
    def _tokenize(self, code: str) -> List[str]:
        """Simple tokenization: split by non-alphanumeric + underscore."""
        # Replace common operators and punctuation with spaces
        code = re.sub(r'[(){}\[\],.;:=+\-*/%<>!&|^~]', ' ', code)
        # Split by whitespace
        tokens = code.split()
        return [t for t in tokens if t]
    
    def _token_count(self, code: str) -> float:
        """Total number of tokens."""
        return float(len(self._tokenize(code)))
    
    def _unique_token_count(self, code: str) -> float:
        """Number of unique tokens."""
        tokens = self._tokenize(code)
        return float(len(set(tokens)))
    
    def _token_diversity(self, code: str) -> float:
        """Unique tokens / total tokens."""
        tokens = self._tokenize(code)
        if not tokens:
            return 0.0
        return len(set(tokens)) / len(tokens)
    
    def _avg_token_length(self, code: str) -> float:
        """Average length of tokens."""
        tokens = self._tokenize(code)
        if not tokens:
            return 0.0
        return np.mean([len(t) for t in tokens])
    
    # =========== Control Flow & Operators ===========
    def _control_flow_count(self, code: str) -> float:
        """Count control flow keywords."""
        count = 0
        for keyword in self.control_keywords:
            # Use word boundary to avoid partial matches
            count += len(re.findall(r'\b' + keyword + r'\b', code, re.IGNORECASE))
        return float(count)
    
    def _operator_count(self, code: str) -> float:
        """Count operators."""
        count = 0
        for op in self.operators:
            # Escape special regex characters
            escaped_op = re.escape(op)
            count += len(re.findall(escaped_op, code))
        return float(count)
    
    # =========== Pattern Features ===========
    def _has_docstring(self, code: str) -> float:
        """Check for docstrings."""
        triple_double = code.count('"""')
        triple_single = 0
        # Check if there are at least 2 of either type (opening and closing pair)
        has_docstring = (triple_double >= 2) or (triple_single >= 2)
        return 1.0 if has_docstring else 0.0
    
    def _has_type_hints(self, code: str) -> float:
        """Check for type hints."""
        # Python: func(x: int) or x: int =
        # Java/C++: int x, String s, etc.
        type_patterns = [
            r':\s*(int|str|float|bool|list|dict|tuple|set|Any)',  # Python
            r'\b(int|String|char|long|float|double|boolean)\s+\w+',  # Java/C++
        ]
        count = 0
        for pattern in type_patterns:
            count += len(re.findall(pattern, code, re.IGNORECASE))
        return float(min(1, count))
    
    def _import_count(self, code: str) -> float:
        """Count import statements."""
        imports = len(re.findall(r'\b(import|from|include|require)\b', code, re.IGNORECASE))
        return float(imports)
    
    # =========== Stylistic Features ===========
    def _indentation_consistency(self, code: str) -> float:
        """Measure indentation consistency."""
        lines = code.split('\n')
        indents = []
        for line in lines:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                indents.append(indent)
        
        if not indents or len(indents) < 2:
            return 1.0
        
        # Check consistency of indentation levels
        indent_diffs = np.diff(indents)
        # Preferred indentation is 4 or 2 or 1
        valid_steps = [0, 1, 2, 4, 8]
        valid = sum(1 for d in indent_diffs if abs(d) in valid_steps)
        return valid / len(indent_diffs) if indent_diffs.size > 0 else 1.0
    
    def _bracket_nesting_depth(self, code: str) -> float:
        """Calculate average bracket nesting depth."""
        max_depth = 0
        current_depth = 0
        for char in code:
            if char in '({[':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char in ')}]':
                current_depth = max(0, current_depth - 1)
        return float(max_depth)
    
    def _code_entropy(self, code: str) -> float:
        """Calculate Shannon entropy of the code."""
        if not code:
            return 0.0
        # Count character frequencies
        freq = Counter(code)
        total = len(code)
        entropy = -sum((count / total) * np.log2(count / total) for count in freq.values())
        return float(entropy)
    
    # =========== Length Features ===========
    def _code_length(self, code: str) -> float:
        """Total character count."""
        return float(len(code))
    
    def _code_length_normalized(self, code: str) -> float:
        """Code length normalized by log."""
        return float(np.log1p(len(code)))


# =============================================================================
# 2. DATA LAYER
# =============================================================================

def load_and_extract_features(parquet_file: str) -> pd.DataFrame:
    """Load parquet file and extract features."""
    print(f"Loading {parquet_file}...")
    df = pd.read_parquet(parquet_file)
    
    print(f"Dataset shape: {df.shape}")
    print(f"First few samples:")
    print(df.head())
    if "label" in df.columns:
        print(f"\nLabel distribution:\n{df['label'].value_counts()}")
    
    print("\nExtracting features from code snippets...")
    extractor = CodeFeatureExtractor()
    
    features_list = []
    for idx, row in df.iterrows():
        if (idx + 1) % 50000 == 0:
            print(f"  Processed {idx + 1}/{len(df)} samples")
        
        code = str(row['code'])
        features = extractor.extract_features(code)
        if "label" in row:
            features['label'] = row['label']
        if 'language' in row:
            features['language'] = row['language']
        features_list.append(features)
    
    features_df = pd.DataFrame(features_list)
    print(f"\nExtracted features shape: {features_df.shape}")
    print(f"Feature columns: {[c for c in features_df.columns if c != 'label']}")
    
    return features_df


# =============================================================================
# 3. TRAINING ENGINE
# =============================================================================

def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    num_rounds: int = 500,
) -> lgb.Booster:
    """Train LightGBM model with macro F1 optimization."""
    
    print("\n" + "="*70)
    print("TRAINING LIGHTGBM")
    print("="*70)
    
    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # Parameters optimized for macro F1 (binary classification)
    params = {
        'objective': 'binary',
        'metric': ['binary_logloss', 'auc'],
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'min_child_samples': 20,
        'num_threads': -1,
        'verbose': -1,
    }
    
    print(f"Training parameters: {params}")
    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Validation set: {X_val.shape[0]} samples")
    
    # Train with early stopping
    evals_result = {}
    model = lgb.train(
        params,
        train_data,
        num_boost_round=num_rounds,
        valid_sets=[val_data],
        valid_names=['validation'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=50)
        ],
        evals_result=evals_result,
    )
    
    # Print training info
    best_iteration = model.best_iteration
    best_score = model.best_score['validation']['binary_logloss']
    print(f"\n✅ Training complete!")
    print(f"   Best iteration: {best_iteration}")
    print(f"   Best validation loss: {best_score:.6f}")
    
    return model


def evaluate_model(model: lgb.Booster, X: pd.DataFrame, y: pd.Series, split_name: str = "Validation"):
    """Evaluate model and print metrics."""
    y_pred_proba = model.predict(X)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    macro_f1 = f1_score(y, y_pred, average='macro')
    weighted_f1 = f1_score(y, y_pred, average='weighted')
    
    precision, recall, f1_dict, _ = precision_recall_fscore_support(
        y, y_pred, average=None, zero_division=0
    )
    
    print(f"\n{'='*70}")
    print(f"{split_name.upper()} METRICS")
    print(f"{'='*70}")
    print(f"Macro F1:     {macro_f1:.4f}")
    print(f"Weighted F1:  {weighted_f1:.4f}")
    print(f"\nClass-wise metrics:")
    print(f"  Class 0 (Human-written):")
    print(f"    Precision: {precision[0]:.4f}")
    print(f"    Recall:    {recall[0]:.4f}")
    print(f"    F1:        {f1_dict[0]:.4f}")
    print(f"  Class 1 (Machine-generated):")
    print(f"    Precision: {precision[1]:.4f}")
    print(f"    Recall:    {recall[1]:.4f}")
    print(f"    F1:        {f1_dict[1]:.4f}")
    
    cm = confusion_matrix(y, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"  TN={cm[0,0]}, FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}, TP={cm[1,1]}")
    
    return {
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'precision': precision,
        'recall': recall,
        'f1': f1_dict,
        'confusion_matrix': cm,
    }


def print_feature_importance(model: lgb.Booster, feature_names: List[str], top_k: int = 20):
    """Print feature importance in descending order."""
    importances = model.feature_importance(importance_type='gain')
    
    # Create importance dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances,
    }).sort_values('importance', ascending=False)
    
    print(f"\n{'='*70}")
    print(f"TOP {top_k} FEATURE IMPORTANCES (Gain)")
    print(f"{'='*70}")
    for idx, row in importance_df.head(top_k).iterrows():
        print(f"{row['feature']:30s}: {row['importance']:10.2f}")
    
    return importance_df


# =============================================================================
# 4. ORCHESTRATION LAYER
# =============================================================================

def main():
    # Configuration
    DATASET_DIR = "dataset"
    OUTPUT_DIR = "lgbm_output"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    print("\n" + "="*70)
    print("LOADING AND PREPARING DATA")
    print("="*70)
    
    train_features = load_and_extract_features(f"{DATASET_DIR}/train.parquet")
    val_features = load_and_extract_features(f"{DATASET_DIR}/validation.parquet")
    test_features = load_and_extract_features(f"{DATASET_DIR}/test.parquet")
    
    # Prepare training data (remove language column if exists)
    feature_cols = [col for col in train_features.columns if col not in ['label', 'language']]
    
    X_train = train_features[feature_cols].copy()
    y_train = train_features['label'].copy()
    
    X_val = val_features[feature_cols].copy()
    y_val = val_features['label'].copy()
    
    X_test = test_features[feature_cols].copy()
    y_test = test_features['label'].copy()
    
    # Handle any missing values
    X_train = X_train.fillna(0)
    X_val = X_val.fillna(0)
    X_test = X_test.fillna(0)
    
    print(f"\nFeatures shape: {X_train.shape}")
    print(f"Feature names: {feature_cols}")
    print(f"\nClass distribution in training set:")
    print(y_train.value_counts())
    print(f"\nClass distribution in validation set:")
    print(y_val.value_counts())
    
    # Train model
    model = train_lightgbm(X_train, y_train, X_val, y_val, num_rounds=500)
    
    # Evaluate on all splits
    train_metrics = evaluate_model(model, X_train, y_train, "Training")
    val_metrics = evaluate_model(model, X_val, y_val, "Validation")
    test_metrics = evaluate_model(model, X_test, y_test, "Test")
    
    # Feature importance
    importance_df = print_feature_importance(model, feature_cols, top_k=20)
    
    # Save model
    model.save_model(f"{OUTPUT_DIR}/lightgbm_model.txt")
    print(f"\n✅ Model saved to {OUTPUT_DIR}/lightgbm_model.txt")
    
    # Run inference and create submission
    print("\n" + "="*70)
    print("RUNNING INFERENCE ON TEST SET")
    print("="*70)
    
    y_test_pred_proba = model.predict(X_test)
    y_test_pred = (y_test_pred_proba > 0.5).astype(int)
    
    # Create submission CSV similar to notebook format
    test_df = pd.read_parquet(f"{DATASET_DIR}/test.parquet")
    submission = pd.DataFrame({
        'id': range(len(y_test_pred)),  # or use actual IDs if available
        'label': y_test_pred
    })
    
    submission_path = f"{OUTPUT_DIR}/submission.csv"
    submission.to_csv(submission_path, index=False)
    print(f"✅ Submission saved to {submission_path}")
    print(f"Submission shape: {submission.shape}")
    print(f"Submission preview:")
    print(submission.head(10))
    
    # Save metrics and feature importance
    metrics_summary = {
        'train': {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in train_metrics.items()},
        'validation': {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in val_metrics.items()},
        'test': {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in test_metrics.items()},
    }
    
    import json
    with open(f"{OUTPUT_DIR}/metrics.json", 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    
    importance_df.to_csv(f"{OUTPUT_DIR}/feature_importance.csv", index=False)
    print(f"✅ Feature importance saved to {OUTPUT_DIR}/feature_importance.csv")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"  - Model: lightgbm_model.txt")
    print(f"  - Submission: submission.csv")
    print(f"  - Metrics: metrics.json")
    print(f"  - Feature importance: feature_importance.csv")


if __name__ == "__main__":
    main()

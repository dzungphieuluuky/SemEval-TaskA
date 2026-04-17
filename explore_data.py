"""
Script to load parquet files from dataset folder and display their schemas.
"""

import pandas as pd
from pathlib import Path

# Define dataset directory
DATASET_DIR = Path(__file__).parent / "dataset"

# Parquet files to explore
parquet_files = ["train.parquet", "validation.parquet", "test.parquet"]

def explore_parquet_files():
    """Load and display schema and basic info for all parquet files."""
    for file_name in parquet_files:
        file_path = DATASET_DIR / file_name
        
        if not file_path.exists():
            print(f"⚠️  File not found: {file_path}")
            continue
        
        print(f"\n{'='*80}")
        print(f"📄 File: {file_name}")
        print(f"{'='*80}")
        
        # Load parquet file
        df = pd.read_parquet(file_path)
        
        # Display schema (column info)
        print(f"\n📋 Schema:")
        print(df.dtypes)
        
        # Display shape
        print(f"\n📊 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        
        # Display first few rows
        print(f"\n🔍 First 5 rows:")
        print(df.head())
        
        # Display column statistics
        print(f"\n📈 Data Info:")
        df.info()
        
        # Display memory usage
        print(f"\n💾 Memory Usage:")
        print(df.memory_usage(deep=True))

if __name__ == "__main__":
    explore_parquet_files()

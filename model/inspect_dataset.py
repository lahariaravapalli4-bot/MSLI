import os
import glob
import pandas as pd

# Search for any .parquet files inside model/data/
search_path = os.path.join("model", "data", "**", "*.parquet")
parquet_files = glob.glob(search_path, recursive=True)

print("=" * 60)
print(f"Total Parquet files found: {len(parquet_files)}")
print("=" * 60)

if not parquet_files:
    print("No .parquet files found! Checking contents of model/data:")
    for root, dirs, files in os.walk(os.path.join("model", "data")):
        print(f"Directory: {root}")
        print(f"Subdirectories: {dirs[:5]}")
        print(f"Sample files: {files[:5]}")
        break
else:
    sample_file = parquet_files[0]
    print(f"\nInspecting sample file:\n{sample_file}\n")
    
    df = pd.read_parquet(sample_file)
    
    print("--- 1. df.columns ---")
    print(df.columns.tolist())
    
    print("\n--- 2. df.shape ---")
    print(df.shape)
    
    print("\n--- 3. df.dtypes ---")
    print(df.dtypes)
    
    print("\n--- 4. df.head(10) ---")
    print(df.head(10))
    
    print("\n--- 5. Unique values in non-coordinate columns ---")
    for col in df.columns:
        if col not in ['x', 'y', 'z', 'landmark_index']:
            print(f"{col}: {df[col].unique()[:10]}")
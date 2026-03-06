#!/usr/bin/env python3
"""Download MTSamples dataset from Kaggle."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config, get_data_paths
from src.data.loader import download_mtsamples_kaggle, load_mtsamples


def main():
    config = load_config()
    raw_dir, _ = get_data_paths(config)
    dataset = config["data"]["kaggle_dataset"]
    print(f"Downloading {dataset} to {raw_dir}...")
    download_mtsamples_kaggle(raw_dir, dataset)
    df = load_mtsamples(raw_dir=raw_dir, download_if_missing=False)
    print(f"Loaded {len(df)} samples, {df['medical_specialty'].nunique()} specialties.")


if __name__ == "__main__":
    main()

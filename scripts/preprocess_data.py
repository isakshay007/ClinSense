#!/usr/bin/env python3
"""
Preprocess MTSamples for classification. Saves cleaned data to data/processed/.

Run locally, then upload mtsamples_classification.csv to Colab for identical data.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, get_data_paths
from src.data.loader import load_mtsamples


def main():
    config = load_config()
    raw_dir, processed_dir = get_data_paths(config)
    min_samples = config["data"].get("min_samples_per_class", 100)
    top_n = config["data"].get("top_n_classes", 10)

    df = load_mtsamples(
        raw_dir=raw_dir,
        download_if_missing=False,
        min_samples_per_class=min_samples,
        top_n_classes=top_n,
    )

    out_path = processed_dir / "mtsamples_classification.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved {len(df)} samples, {df['medical_specialty'].nunique()} classes to {out_path}")


if __name__ == "__main__":
    main()

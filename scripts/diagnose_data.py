#!/usr/bin/env python3
"""
Diagnose data before training: class distribution, text lengths, sample previews.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, get_data_paths
from src.data.loader import load_mtsamples, prepare_classification_data


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="Path to mtsamples file (csv or parquet)")
    args = parser.parse_args()

    config = load_config()
    raw_dir, _ = get_data_paths(config)
    data_cfg = config["data"]

    path = Path(args.path) if args.path else raw_dir / "mtsamples.csv"
    if not path.exists() and not args.path:
        # Try parquet if csv missing
        path = raw_dir / "mtsamples.parquet"

    print(f"Loading data from: {path}")
    df = load_mtsamples(
        data_path=path,
        download_if_missing=False,
        min_samples_per_class=data_cfg.get("min_samples_per_class", 150),
        top_n_classes=data_cfg.get("top_n_classes", 8),
        specialties=data_cfg.get("specialties"),
    )

    text_col = "transcription"
    df["text_clean"] = df[text_col].fillna("").astype(str).str.strip()

    print("=" * 60)
    print("1. CLASS DISTRIBUTION (after filtering)")
    print("=" * 60)
    vc = df["medical_specialty"].value_counts()
    print(vc)
    print(f"\nTotal classes: {len(vc)}, Total samples: {len(df)}")
    print(f"Min samples/class: {vc.min()}, Max: {vc.max()}")

    print("\n" + "=" * 60)
    print("2. TEXT LENGTH STATS")
    print("=" * 60)
    lens = df["text_clean"].str.len()
    print(f"Min: {lens.min()}, Max: {lens.max()}, Mean: {lens.mean():.0f}")
    print(f"Empty (< 10 chars): {(lens < 10).sum()}")
    print(f"Short (< 50 chars): {(lens < 50).sum()}")

    print("\n" + "=" * 60)
    print("3. SAMPLE TEXTS BEFORE vs AFTER (first 5 rows)")
    print("=" * 60)
    for i in range(min(5, len(df))):
        raw = df[text_col].iloc[i]
        clean = df["text_clean"].iloc[i]
        print(f"\n--- Row {i} [{df['medical_specialty'].iloc[i]}] ---")
        print(f"BEFORE (len={len(str(raw))}): {str(raw)[:150]}...")
        print(f"AFTER  (len={len(clean)}): {clean[:150]}...")

    X_train, _, X_test, y_train, _, y_test, _ = prepare_classification_data(df)
    print("\n" + "=" * 60)
    print("4. TRAIN SPLIT VALUE_COUNTS")
    print("=" * 60)
    print(y_train.value_counts().sort_index())


if __name__ == "__main__":
    main()

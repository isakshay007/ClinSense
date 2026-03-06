#!/usr/bin/env python3
"""
Extract medical entities (drugs, diseases) from MTSamples using scispaCy.

Saves enriched CSV with entities and optionally logs stats to W&B.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, get_data_paths
from src.data.loader import load_mtsamples
from src.ner.scispacy_ner import add_entity_columns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/processed/mtsamples_entities.csv")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Limit rows for quick test")
    args = parser.parse_args()

    config = load_config()
    raw_dir, processed_dir = get_data_paths(config)

    df = load_mtsamples(
        raw_dir=raw_dir,
        download_if_missing=args.download,
        dataset=config["data"]["kaggle_dataset"],
    )
    if args.limit:
        df = df.head(args.limit)

    print("Extracting entities with scispaCy (en_ner_bc5cdr_md)...")
    df = add_entity_columns(df)

    out_path = Path(__file__).parent.parent / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save without raw entities (list of dicts) for smaller file
    cols = [c for c in df.columns if c != "entities_raw"]
    df[cols].to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
    print(f"Sample stats: {df['drugs'].apply(len).sum()} drugs, {df['diseases'].apply(len).sum()} diseases")


if __name__ == "__main__":
    main()
